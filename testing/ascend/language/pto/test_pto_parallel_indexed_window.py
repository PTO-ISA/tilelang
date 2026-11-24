"""B16 windows within allocations larger than ui16 offsets."""

import pytest

from test_pto_parallel_gather_scatter import T, assert_vector_axis, calls, lower_case, tvm


def outer_range_window_kernel(flat=False):
    @T.prim_func
    def main(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((2, 128), "float16")):
        with T.Kernel(2) as bx:
            x = T.alloc_shared((70000,), "float16")
            idx = T.alloc_shared((128,), "uint16")
            y = T.alloc_shared((128,), "float16")
            view = T.view(x, shape=(2, 35000))
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if flat:
                        k = idx[i]
                        idx[i] = 0
                        y[i] = x[35000 * bx + k]
                    else:
                        y[i] = view[bx, idx[i]]
            T.copy(y, C[bx, :])

    return main


@pytest.mark.pto
@pytest.mark.parametrize("flat", [False, True], ids=["view", "flat_saved"])
def test_b16_window_with_block_dependent_base(flat):
    _, captured = lower_case("outer_range_window", outer_range_window_kernel(flat))
    converted = captured["VectorizeParallelToPTO_after"]
    gather = calls(converted, "tl.vmi.vgather")[0]
    assert str(gather.args[2].dtype) == "uint16x128"
    assert "35000" in str(gather.args[1])


def alternate_axis_window_kernel(kind="gather"):
    source_shape = (128, 600) if kind == "gather" else (2, 128)
    output_shape = (2, 128) if kind == "gather" else (128, 600)

    @T.prim_func
    def main(A: T.Tensor(source_shape, "float16"), C: T.Tensor(output_shape, "float16")):
        with T.Kernel(1):
            x = T.alloc_shared(source_shape, "float16")
            y = T.alloc_shared(output_shape, "float16")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(2, 128):
                    if kind == "gather":
                        y[i, j] = x[j, 2 * i]
                    else:
                        y[j, 2 * i] = x[i, j]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_affine_offsets_choose_representable_alternate_axis(kind):
    # Along j the full physical window exceeds ui16. Along i a bounded
    # two-element relative window and the scalar row base are lowerable.
    _, captured = lower_case(f"alternate_axis_{kind}", alternate_axis_window_kernel(kind))
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 0)
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, "tl.vmi.vgather")) == 1
    assert len(calls(converted, "tl.vmi.vscatter")) == 1


def base_outside_owner_kernel():
    @T.prim_func
    def main(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            raw = T.alloc_shared((70000,), "float16")
            # The view's uniform origin sits exactly at the owner's end: its
            # relative window is representable, but the rebased scalar base
            # is provably outside the owner, and the diagnostic must name the
            # base contract instead of blaming the ui16 window.
            view = tvm.tirx.decl_buffer((256,), "float16", data=raw.data, elem_offset=70000, scope="shared.dyn")
            idx = T.alloc_shared((128,), "uint16")
            y = T.alloc_shared((128,), "float16")
            T.copy(A, raw)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    y[i] = view[idx[i]]
            T.copy(y, C)

    return main


@pytest.mark.pto
def test_rebased_base_outside_owner_reports_the_base_contract():
    with pytest.raises(Exception, match="scalar instruction base D.*inside the owner.*70000"):
        lower_case("rejected_base_outside_owner", base_outside_owner_kernel())


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("origin", [0, 32])
def test_native_index_can_address_a_larger_owner(kind, origin):
    size_in = 70000 if kind == "gather" else 128
    size_out = 70000 if kind == "scatter" else 128

    @T.prim_func
    def main(A: T.Tensor((size_in,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((size_out,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((size_in,), "float16")
            idx = T.alloc_shared((128,), "uint16")
            y = T.alloc_shared((size_out,), "float16")
            raw = x if kind == "gather" else y
            view = tvm.tirx.decl_buffer((70000 - origin,), "float16", data=raw.data, elem_offset=origin, scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if kind == "gather":
                        y[i] = view[idx[i]]
                    else:
                        view[idx[i]] = x[i]
            T.copy(y, C)

    _, captured = lower_case("large_owner_" + kind, main)
    (indexed,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.v" + kind)
    base, offsets = indexed.args[1:3] if kind == "gather" else indexed.args[2:4]
    assert int(base) == origin and str(offsets.dtype) == "uint16x128"


def affine_large_window_kernel(kind, address_form, index_mode=None, *, origin=66000, stride=2):
    size_in, size_out = (70000, 384) if kind == "gather" else (384, 70000)
    extent = 270
    index_start = 32760 if index_mode == "wrapped_int16" else 0
    cast_index = index_mode in ("int16", "wrapped_int16")
    view_size = index_start + stride * (extent - 1) + 1

    @T.prim_func
    def main(A: T.Tensor((size_in,), "float16"), C: T.Tensor((size_out,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((size_in,), "float16")
            y = T.alloc_shared((size_out,), "float16")
            raw = x if kind == "gather" else y
            if address_form == "strided":
                view = tvm.tirx.decl_buffer((2, view_size), "float16", data=raw.data, strides=(origin, 1), scope="shared.dyn")
            elif address_form == "shifted":
                view = tvm.tirx.decl_buffer((view_size,), "float16", data=raw.data, elem_offset=origin, scope="shared.dyn")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    coord = stride * (extent - 1 - i) if index_mode == "reverse" else index_start + stride * i
                    if index_mode == "round_trip":
                        index = T.meta_var(stride * T.cast(T.cast(i, "uint16"), "int32"))
                    elif index_mode == "unsigned":
                        index = T.meta_var(T.uint32(stride) * T.cast(i, "uint32"))
                    else:
                        index = T.meta_var(T.cast(coord, "int16") if cast_index else coord)
                    flat_index = T.meta_var(T.cast(origin + index, "uint32") if index_mode == "cast_address" else origin + index)
                    if kind == "gather":
                        if address_form == "strided":
                            y[i] = view[1, index]
                        elif address_form == "shifted":
                            y[i] = view[index]
                        else:
                            y[i] = raw[flat_index]
                    else:
                        if address_form == "strided":
                            view[1, index] = x[i]
                        elif address_form == "shifted":
                            view[index] = x[i]
                        else:
                            raw[flat_index] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "address_form,index_mode",
    [("flat", None), ("strided", None), ("shifted", None), ("flat", "cast_address"), ("shifted", "round_trip"), ("flat", "unsigned")],
)
def test_b16_equivalent_affine_addresses_use_narrow_offsets(address_form, index_mode):
    kernel = affine_large_window_kernel("gather", address_form, index_mode)
    _, captured = lower_case(f"affine_large_window_{address_form}_{index_mode}", kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    (gather,) = calls(converted, "tl.vmi.vgather")
    assert str(gather.args[1].dtype) == "int32"
    assert str(gather.args[2].dtype) == "uint16x128"
    assert not calls(converted, "tl.vmi.vcvt")


@pytest.mark.pto
@pytest.mark.parametrize("address_form", ["flat", "strided", "shifted"])
def test_b16_descending_affine_addresses_use_narrow_offsets(address_form):
    kernel = affine_large_window_kernel("gather", address_form, "reverse")
    _, captured = lower_case(f"descending_large_window_{address_form}", kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    (gather,) = calls(converted, "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "uint16x128"
    assert not calls(converted, "tl.vmi.vcvt")


@pytest.mark.pto
def test_b16_relative_window_keeps_proven_nonnegative_signed_index():
    kernel = affine_large_window_kernel("gather", "shifted", "int16")
    _, captured = lower_case("signed_large_window", kernel)
    (gather,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert int(gather.args[1]) == 66000
    assert str(gather.args[2].dtype) == "int16x128"


@pytest.mark.pto
def test_b16_relative_window_rejects_signed_index_wrapping_on_cast():
    # A cast of 32768 to int16 yields -32768, which reaches owner element 0.
    # Reinterpreting the narrow signed carrier would reach 65536 instead.
    kernel = affine_large_window_kernel("gather", "shifted", "wrapped_int16", origin=32768)
    with pytest.raises(Exception, match="B16 indexed access cannot prove the input of a vector int32-to-int16 cast fits in int16"):
        # Exercise PTO admission without the scheduler's int16 interval analysis.
        lower_case("signed_large_window_wrap", kernel, auto_schedule=False)


def uniform_row_window(use_remainder):
    @T.prim_func
    def main(
        A: T.Tensor((76800,), "float16"),
        I: T.Tensor((32,), "uint8"),
        C: T.Tensor((128,), "float16"),
    ):
        with T.Kernel(1):
            raw = T.alloc_shared((76800,), "float16")
            idx = T.alloc_shared((32,), "uint8")
            y = T.alloc_shared((128,), "float16")
            view = tvm.tirx.decl_buffer((256, 300), "float16", data=raw.data, strides=(300, 1), scope="shared.dyn")
            T.copy(A, raw)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if use_remainder:
                        y[i] = view[T.truncmod(300, T.cast(idx[0], "int32")), 2 * i]
                    else:
                        y[i] = view[idx[0], 2 * i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("use_remainder", [False, True], ids=["load", "remainder"])
def test_b16_rebased_address_keeps_uniform_loaded_row_scalar(use_remainder):
    _, captured = lower_case("uniform_row_window", uniform_row_window(use_remainder))
    (gather,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    base, offsets = gather.args[1:3]
    assert str(base.dtype) == "int32"
    assert str(offsets.dtype) == "uint16x128"
    loads = []
    tvm.tirx.stmt_functor.post_order_visit(base, lambda node: loads.append(node) if isinstance(node, tvm.tirx.BufferLoad) else None)
    (row,) = loads
    assert str(row.dtype) == "uint8" and int(row.indices[0]) == 0


@T.prim_func
def saved_large_window(A: T.Tensor((256,), "float16"), I: T.Tensor((70000,), "uint16"), C: T.Tensor((128,), "float16")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float16")
        idx = T.alloc_shared((70000,), "uint16")
        y = T.alloc_shared((128,), "float16")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                slot = 66000 + 2 * i
                old = idx[slot]
                idx[slot] = 0
                y[i] = x[old]
        T.copy(y, C)


@pytest.mark.pto
def test_b16_rebased_address_uses_the_saved_index_value():
    _, captured = lower_case("saved_large_window", saved_large_window)
    converted = captured["VectorizeParallelToPTO_after"]
    gather_index, gather_value = calls(converted, "tl.vmi.vgather")
    assert int(gather_index.args[1]) == 66000
    assert str(gather_index.args[2].dtype) == str(gather_value.args[2].dtype) == "uint16x128"
    assert len(calls(converted, "tl.vmi.vscatter")) == 1


def saved_index_window_kernel(address_form, index_dtype, *, explicit_widen=False):
    @T.prim_func
    def main(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), index_dtype), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((70000,), "float16")
            idx = T.alloc_shared((128,), index_dtype)
            y = T.alloc_shared((128,), "float16")
            if address_form == "strided":
                view = tvm.tirx.decl_buffer((2, 4000), "float16", data=x.data, strides=(66000, 1), scope="shared.dyn")
            elif address_form == "shifted":
                view = tvm.tirx.decl_buffer((4000,), "float16", data=x.data, elem_offset=66000, scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    old = idx[i]
                    idx[i] = 0
                    if index_dtype == "int8":
                        offset = T.meta_var(T.cast(T.cast(old, "uint8"), "int16") + T.int16(1))
                    else:
                        offset = T.meta_var(T.cast(old, "int32") if explicit_widen else old)
                    if address_form == "strided":
                        y[i] = view[1, offset]
                    elif address_form == "shifted":
                        y[i] = view[offset]
                    else:
                        y[i] = x[66000 + offset]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("address_form", ["flat", "strided", "shifted"])
def test_b16_rebased_address_accepts_explicit_widening_of_saved_uint16(address_form):
    kernel = saved_index_window_kernel(address_form, "uint16", explicit_widen=True)
    _, captured = lower_case(f"widened_saved_window_{address_form}", kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    (gather,) = calls(converted, "tl.vmi.vgather")
    assert int(gather.args[1]) == 66000
    assert str(gather.args[2].dtype) == "uint16x128"
    assert not calls(converted, "tl.vmi.vcvt")


@pytest.mark.pto
def test_b16_rebased_address_accepts_arithmetic_on_widened_unsigned_byte():
    kernel = saved_index_window_kernel("shifted", "int8")
    _, captured = lower_case("widened_byte_arithmetic_window", kernel)
    (gather,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert int(gather.args[1]) == 66000
    assert str(gather.args[2].dtype) == "int16x128"


@pytest.mark.pto
@pytest.mark.parametrize("index_dtype,explicit_widen", [("int32", False), ("int16", False), ("int16", True)])
@pytest.mark.parametrize("address_form", ["flat", "shifted"])
def test_b16_rebased_address_rejects_saved_indices_without_a_safe_carrier(address_form, index_dtype, explicit_widen):
    kernel = saved_index_window_kernel(address_form, index_dtype, explicit_widen=explicit_widen)
    with pytest.raises(Exception, match="32-bit offsets require implicit narrowing to 16 bits"):
        lower_case(f"wide_saved_window_{address_form}", kernel)


@pytest.mark.pto
@pytest.mark.parametrize("owner_size,origin,source_dtype,index_dtype", [(512, 256, "uint8", "uint16"), (70000, 66000, "int16", "int16")])
def test_b16_view_accepts_widened_unsigned_byte_index(owner_size, origin, source_dtype, index_dtype):
    @T.prim_func
    def main(A: T.Tensor((owner_size,), "float16"), I: T.Tensor((128,), source_dtype), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            raw = T.alloc_shared((owner_size,), "float16")
            idx = T.alloc_shared((128,), source_dtype)
            y = T.alloc_shared((128,), "float16")
            view = tvm.tirx.decl_buffer((256,), "float16", data=raw.data, elem_offset=origin, scope="shared.dyn")
            T.copy(A, raw)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    y[i] = view[T.cast(T.cast(idx[i], "uint8"), index_dtype)]
            T.copy(y, C)

    _, captured = lower_case("widened_byte_window", main)
    (gather,) = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert int(gather.args[1]) == origin
    assert str(gather.args[2].dtype) == f"{index_dtype}x128"
