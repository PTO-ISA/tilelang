"""Byte data and offset carriers through automatic Parallel lowering."""

import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from test_pto_parallel_gather_scatter import binding_values, calls, lower_case

pytestmark = pytest.mark.pto


def byte_offset_view_kernel(kind, explicit_widen=False, *, payload="float32", index_dtype="int8", origin=256, uniform_index=False):
    extent, index_stride = (1, 0) if uniform_index else (150, 1)

    @T.prim_func
    def main(A: T.Tensor((512,), payload), I: T.Tensor((256,), index_dtype), C: T.Tensor((512,), payload)):
        with T.Kernel(1):
            x = T.alloc_shared((512,), payload)
            idx = T.alloc_shared((256,), index_dtype)
            y = T.alloc_shared((512,), payload)
            base = x if kind == "gather" else y
            view = tvm.tirx.decl_buffer((256,), payload, data=base.data, elem_offset=origin, scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    offset = idx[i * index_stride]
                    if explicit_widen:
                        offset = T.cast(offset, "int32") + 128
                    if kind == "gather":
                        y[i] = view[offset]
                    else:
                        view[offset] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_byte_view_offsets_reinterpret_before_address_arithmetic(kind):
    _, captured = lower_case(f"byte_view_{kind}", byte_offset_view_kernel(kind))
    mod = captured["VectorizeParallelToPTO_after"]
    assert any(str(call.dtype) == "uint8x128" for call in calls(mod, "tl.vmi.vinterpret_cast"))
    assert len(calls(mod, f"tl.vmi.v{kind}")) == 1


@pytest.mark.parametrize("kind,uniform_index", [("gather", False), ("scatter", False), ("scatter", True)])
@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
def test_b16_view_keeps_byte_relative_offsets(kind, uniform_index, index_dtype):
    kernel = byte_offset_view_kernel(kind, payload="float16", index_dtype=index_dtype, origin=32, uniform_index=uniform_index)
    _, captured = lower_case(f"b16_byte_view_{kind}_{index_dtype}", kernel)
    (indexed,) = calls(captured["VectorizeParallelToPTO_after"], f"tl.vmi.v{kind}")
    base, offsets = indexed.args[1:3] if kind == "gather" else indexed.args[2:4]
    assert int(base) == 32
    assert str(offsets.dtype) == "uint8x128"


def version_slot_view_kernel(kind, scheduled, widen=False, origin=0):
    @T.prim_func
    def main(A: T.Tensor((4, 256), "float16"), I: T.Tensor((4, 128), "uint8"), C: T.Tensor((4, 256), "float16")):
        with T.Kernel(1):
            shape = (256,) if scheduled else (2, 256)
            x = T.alloc_shared(shape, "float16")
            y = T.alloc_shared(shape, "float16")
            idx = T.alloc_shared((128,), "uint8")
            owner = x if kind == "gather" else y
            view_shape = (1, 128) if scheduled else (2, 1, 128)
            view_strides = (128, 1) if scheduled else (256, 128, 1)
            view = tvm.tirx.decl_buffer(
                view_shape, "float16", data=owner.data, strides=view_strides, elem_offset=origin, scope="shared.dyn"
            )
            if scheduled:
                T.annotate_buffer_versions({x: 2, idx: 2, y: 2})
                for t in T.Pipelined(4, num_stages=2):
                    T.copy(A[t, :], x)
                    T.copy(I[t, :], idx)
                    T.copy(C[t, :], y)
                    with T.SimdVF(lanes=128):
                        for i in T.Parallel(128):
                            offset = T.cast(idx[i], "uint16") if widen else idx[i]
                            if kind == "gather":
                                y[i] = view[0, offset]
                            else:
                                view[0, offset] = x[i]
                    T.copy(y, C[t, :])
            else:
                T.annotate_manual_multi_buffer(x, y)
                for t in T.serial(4):
                    slot = t % 2
                    T.copy(A[t, :], x[slot, :])
                    T.copy(I[t, :], idx)
                    T.copy(C[t, :], y[slot, :])
                    with T.SimdVF(lanes=128):
                        for i in T.Parallel(128):
                            offset = T.cast(idx[i], "uint16") if widen else idx[i]
                            if kind == "gather":
                                y[slot, i] = view[slot, 0, offset]
                            else:
                                view[slot, 0, offset] = x[slot, i]
                    T.copy(y[slot, :], C[t, :])

    return main


@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("scheduled,widen,origin", [(True, False, 0), (True, True, 0), (False, False, 32)])
def test_aligned_version_view_uses_slot_descriptor(kind, scheduled, widen, origin):
    _, captured = lower_case("aligned_version_byte_view", version_slot_view_kernel(kind, scheduled, widen, origin))
    (indexed,) = calls(captured["VectorizeParallelToPTO_after"], f"tl.vmi.v{kind}")
    pointer, base, offsets = indexed.args[:3] if kind == "gather" else indexed.args[1:4]
    start, extent, _ = pointer.args
    assert not isinstance(start.indices[0], tvm.tirx.IntImm)
    assert int(extent) == 256
    assert int(base) == origin
    assert str(offsets.dtype) == ("uint16x128" if widen else "uint8x128")


def test_aligned_version_view_that_spans_slots_keeps_owner_descriptor():
    kernel = version_slot_view_kernel("gather", scheduled=False, origin=160)
    _, captured = lower_case("cross_aligned_slots", kernel)
    (indexed,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    start, extent, _ = indexed.args[0].args
    assert int(start.indices[0]) == 0
    assert int(extent) == 512


def cast_bind_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), C: T.Tensor((256,), "int32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            y = T.alloc_shared((256,), "int32")
            T.copy(A, x)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    value = T.cast(x[i], "int32")
                    y[i] = value + 1
            T.copy(y, C)

    return main


def byte_gather_kernel(dtype, index_dtype="uint16", widen=False):
    result_dtype = "int32" if widen else dtype

    @T.prim_func
    def main(A: T.Tensor((256,), dtype), I: T.Tensor((256,), index_dtype), C: T.Tensor((256,), result_dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            idx = T.alloc_shared((256,), index_dtype)
            y = T.alloc_shared((256,), result_dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    value = x[idx[i]]
                    if widen:
                        promoted = T.cast(value, "int32")
                        y[i] = promoted + 1
                    else:
                        y[i] = value
            T.copy(y, C)

    return main


def byte_scatter_kernel(dtype, index_dtype="uint16"):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), I: T.Tensor((256,), index_dtype), C: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            idx = T.alloc_shared((256,), index_dtype)
            y = T.alloc_shared((256,), dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    y[idx[i]] = x[i]
            T.copy(y, C)

    return main


def byte_gather_scatter_chain_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), I: T.Tensor((256,), "uint16"), J: T.Tensor((256,), "uint16"), C: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), dtype)
            idx = T.alloc_shared((256,), "uint16")
            dst = T.alloc_shared((256,), "uint16")
            y = T.alloc_shared((256,), dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(J, dst)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    y[dst[i]] = x[idx[i]]
            T.copy(y, C)

    return main


def explicitly_widened_byte_offset_kernel(index_dtype):
    owner_size = 65536 if index_dtype == "int8" else 256

    @T.prim_func
    def main(A: T.Tensor((owner_size,), "float16"), I: T.Tensor((128,), index_dtype), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((owner_size,), "float16")
            idx = T.alloc_shared((128,), index_dtype)
            y = T.alloc_shared((128,), "float16")
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    y[i] = x[T.cast(idx[i], "uint16")]
            T.copy(y, C)

    return main


@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
def test_byte_offset_keeps_explicit_widening(index_dtype):
    _, captured = lower_case("widen_byte_offset", explicitly_widened_byte_offset_kernel(index_dtype))
    (gather,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "uint16x128"


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_ordinary_byte_cast_bind_uses_value_classification(dtype):
    _, captured = lower_case(f"cast_bind_{dtype}", cast_bind_kernel(dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    assert len(calls(mod, "tl.vmi.vcvt")) == 1
    assert len(calls(mod, "tl.vmi.vadd")) == 1
    assert not calls(mod, "tl.vmi.vgather")


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("widen", [False, True])
def test_byte_gather_restores_source_before_consuming_bind(dtype, widen):
    _, captured = lower_case(f"byte_gather_{dtype}_{widen}", byte_gather_kernel(dtype, widen=widen))
    mod = captured["VectorizeParallelToPTO_after"]
    (gather,) = calls(mod, "tl.vmi.vgather")
    assert str(gather.dtype) == ("int16x128" if dtype == "int8" else "uint16x128")
    conversions = calls(mod, "tl.vmi.vcvt")
    assert str(conversions[0].dtype) == f"{dtype}x128"
    assert str(conversions[0].annotations["saturate"].value) == "NOSAT"
    if widen:
        assert str(conversions[1].dtype) == ("int32x128" if dtype == "int8" else "uint32x128")
        if dtype == "uint8":
            assert str(calls(mod, "tl.vmi.vinterpret_cast")[0].dtype) == "int32x128"
    offsets, mask = gather.args[-2:]
    bindings = binding_values(mod)
    assert bindings[offsets].op.name == "tl.vmi.vload"
    mask = bindings[mask]
    assert mask.op.name == "tl.vmi.create_mask" and isinstance(mask.args[0], tvm.tirx.BufferLoad)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_scatter_keeps_source_dtype(dtype):
    _, captured = lower_case(f"byte_scatter_{dtype}", byte_scatter_kernel(dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    (scatter,) = calls(mod, "tl.vmi.vscatter")
    # The scattered value keeps the source element dtype; a continuous byte
    # load feeds the scatter without any restore or widening step.
    assert str(scatter.args[0].dtype) == f"{dtype}x128"
    assert str(scatter.args[3].dtype) == "uint16x128"
    assert not calls(mod, "tl.vmi.vgather")
    assert not calls(mod, "tl.vmi.vcvt")


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_scatter_consumes_the_restored_gather(dtype):
    _, captured = lower_case(f"byte_chain_{dtype}", byte_gather_scatter_chain_kernel(dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    assert len(calls(mod, "tl.vmi.vgather")) == 1
    (scatter,) = calls(mod, "tl.vmi.vscatter")
    # The byte gather is restored to the source dtype before its scatter
    # consumer; the widened gather result never reaches the store value.
    conversions = calls(mod, "tl.vmi.vcvt")
    assert str(conversions[0].dtype) == f"{dtype}x128"
    assert str(conversions[0].annotations["saturate"].value) == "NOSAT"
    assert str(scatter.args[0].dtype) == f"{dtype}x128"
