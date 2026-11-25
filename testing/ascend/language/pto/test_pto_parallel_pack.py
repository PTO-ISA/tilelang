"""Automatic PTO Parallel lowering for byte carriers and stride-two pairs."""

import pytest

import tilelang.ascend.language as T
from tilelang.ascend.target import normalize_pto_target

from test_pto_parallel_gather_scatter import ascend_transform, binding_values, calls, indexed_kernel, lower_case, tvm
from test_pto_parallel_indexed_control_flow import replace_unit_body


def scale_bits_kernel(extent):
    @T.prim_func
    def main(E: T.Tensor((256,), "uint8"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            e = T.alloc_shared((256,), "uint8")
            c = T.alloc_shared((256,), "float32")
            T.copy(E, e)
            if extent < 256:
                T.copy(C, c)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(extent):
                    c[i] = T.reinterpret(T.Cast("uint32", e[i]) << T.uint32(23), "float32")
            T.copy(c, C)

    return main


def reinterpret_kernel(src_dtype, dst_dtype, lanes, *, uniform=False):
    @T.prim_func
    def main(A: T.Tensor((lanes,), src_dtype), C: T.Tensor((lanes,), dst_dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((lanes,), src_dtype)
            c = T.alloc_shared((lanes,), dst_dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(lanes):
                    c[i] = T.reinterpret(a[0 if uniform else i], dst_dtype)
            T.copy(c, C)

    return main


def reinterpret_lut_kernel(saved_value, *, unary=False):
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), L: T.Tensor((16,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float32")
            lut = T.alloc_shared((16,), "float32")
            c = T.alloc_shared((64,), "float32")
            T.copy(A, a)
            T.copy(L, lut)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(64):
                    if saved_value:
                        value = T.sqrt(a[i]) if unary else a[i] + T.float32(1)
                        c[i] = lut[T.reinterpret(value, "uint32") & T.uint32(15)]
                    else:
                        c[i] = lut[T.reinterpret(T.sqrt(a[i]) if unary else a[i] + T.float32(1), "uint32") & T.uint32(15)]
            T.copy(c, C)

    return main


@T.prim_func
def uniform_bits(A: T.Tensor((64,), "int32"), C: T.Tensor((64,), "int32")):
    with T.Kernel(1):
        a = T.alloc_shared((64,), "int32")
        c = T.alloc_shared((64,), "int32")
        T.copy(A, a)
        with T.SimdVF(lanes=64):
            for i in T.Parallel(64):
                c[i] = ((a[0] >> (a[1] & 7)) & 15) | a[2]
        T.copy(c, C)


def right_shift_kernel(bits):
    lanes = 2048 // bits
    signed = f"int{bits}"
    unsigned = f"uint{bits}"

    @T.prim_func
    def main(A: T.Tensor((lanes,), signed), C: T.Tensor((2, lanes), signed)):
        with T.Kernel(1):
            a = T.alloc_shared((lanes,), signed)
            c = T.alloc_shared((2, lanes), signed)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(lanes):
                    raw = a[i]
                    logical = T.reinterpret(raw, unsigned) >> T.Cast(unsigned, 3)
                    c[0, i] = raw >> T.Cast(signed, 3)
                    c[1, i] = T.reinterpret(logical, signed)
            T.copy(c, C)

    return main


def uint32_bitwise_kernel(uniform):
    @T.prim_func
    def main(A: T.Tensor((128,), "uint32"), C: T.Tensor((2, 128), "uint32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "uint32")
            c = T.alloc_shared((2, 128), "uint32")
            T.copy(A, a)
            T.copy(C, c)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(65):
                    x = a[0 if uniform else i]
                    c[0, i] = (x >> 23) ^ x
                    c[1, i] = T.bitwise_not(x)
            T.copy(c, C)

    return main


def scale_pack_kernel(accumulate_invalid):
    output_dtype = "uint32" if accumulate_invalid else "uint16"

    @T.prim_func
    def main(A: T.Tensor((2, 128), "uint32"), C: T.Tensor((128,), output_dtype)):
        with T.Kernel(1):
            words = T.alloc_shared((2, 128), "uint32")
            out = T.alloc_shared((128,), output_dtype)
            T.copy(A, words)
            T.copy(C, out)
            with T.SimdVF(lanes=64):
                for j in T.serial(2):
                    for i in T.Parallel(65):
                        sf_bits = words[j, i]
                        if accumulate_invalid:
                            out[i] = out[i] | (sf_bits & T.uint32(0x807FFFFF))
                        else:
                            sf_exp = T.cast(sf_bits >> 23, T.uint16, sat=False)
                            out[i] = T.Cast("uint16", out[i] | (sf_exp << (j * 8)))
            T.copy(out, C)

    return main


def varying_shift_kernel(dtype, *, right):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((128,), dtype)
            b = T.alloc_shared((128,), dtype)
            out = T.alloc_shared((128,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            T.copy(C, out)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(65):
                    if right:
                        out[i] = a[i] >> b[i]
                    else:
                        out[i] = a[i] << b[i]
            T.copy(out, C)

    return main


def byte_lut_kernel(kind, extent):
    lanes = 64 if kind == "b32" else 128
    byte_dtype = "uint8" if kind == "b16" else "int8"
    lut_dtype = "float32" if kind == "b32" else "float16"
    index_dtype = "uint16" if kind == "b16" else "int32"
    lut_size = 32 if kind == "b16_wide" else 16

    @T.prim_func
    def main(B: T.Tensor((256,), byte_dtype), L: T.Tensor((lut_size,), lut_dtype), C: T.Tensor((512,), lut_dtype)):
        with T.Kernel(1):
            b = T.alloc_shared((256,), byte_dtype)
            lut = T.alloc_shared((lut_size,), lut_dtype)
            c = T.alloc_shared((512,), lut_dtype)
            T.copy(B, b)
            T.copy(L, lut)
            if extent < 256:
                T.copy(C, c)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(extent):
                    x = T.Cast(index_dtype, b[i])
                    if kind == "b16":
                        c[2 * i] = lut[x & T.uint16(15)]
                        c[2 * i + 1] = lut[x >> T.uint16(4)]
                    elif kind == "b16_wide":
                        c[2 * i] = lut[1 + (x & 15)]
                        c[2 * i + 1] = lut[1 + ((x >> 4) & 15)]
                    else:
                        c[2 * i] = lut[x & 15]
                        c[2 * i + 1] = lut[(x >> 4) & 15]
            T.copy(c, C)

    return main


def alternating_kernel(extent, width):
    @T.prim_func
    def main(A: T.Tensor((width,), "float16"), C: T.Tensor((2, 128), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((width,), "float16")
            c = T.alloc_shared((2, 128), "float16")
            T.copy(A, a)
            if extent < 128:
                T.copy(C, c)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    c[0, i] = a[2 * i]
                    c[1, i] = a[2 * i + 1]
            T.copy(c, C)

    return main


@T.prim_func
def bounded_byte_index(L: T.Tensor((256,), "float16"), I: T.Tensor((128,), "uint8"), C: T.Tensor((128,), "float16")):
    with T.Kernel(1):
        lut = T.alloc_shared((256,), "float16")
        idx = T.alloc_shared((128,), "uint8")
        c = T.alloc_shared((128,), "float16")
        T.copy(L, lut)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                c[i] = lut[T.Cast("int32", idx[i])]
        T.copy(c, C)


def signed_b16_index_kernel(kind, index_mode):
    index_dtype = "int16" if index_mode == "native" else "int32"

    @T.prim_func
    def main(I: T.Tensor((128,), index_dtype), A: T.Tensor((128,), "float16"), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            idx = T.alloc_shared((128,), index_dtype)
            a = T.alloc_shared((128,), "float16")
            c = T.alloc_shared((128,), "float16")
            T.copy(I, idx)
            T.copy(A, a)
            T.copy(C, c)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if index_mode == "native":
                        offset = idx[i]
                    elif index_mode == "bounded":
                        offset = T.meta_var(T.cast(idx[i] & 127, T.int16, sat=False))
                    elif index_mode == "bound_cast":
                        narrowed = T.bind(T.cast(idx[i], T.int16, sat=False))
                        offset = T.meta_var(narrowed & T.int16(127))
                    elif index_mode == "scalar_cast":
                        base = T.meta_var(T.cast(idx[0], T.int16, sat=False) & T.int16(1))
                        offset = T.meta_var(base | T.cast(idx[i] & 126, T.int16, sat=False))
                    else:
                        offset = T.meta_var(T.cast(idx[i], T.int16, sat=False) & T.int16(127))
                    if kind == "gather":
                        c[i] = a[offset]
                    else:
                        c[offset] = a[i]
            T.copy(c, C)

    return main


def row_scale_kernel(row_record):
    row_width = 132 if row_record else 128

    @T.prim_func
    def main(
        A: T.Tensor((65, row_width), "float32"),
        S: T.Tensor((64, 4), "float32"),
        I: T.Tensor((32,), "int32"),
        C: T.Tensor((32, 128), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((65, row_width), "float32")
            sf = T.alloc_shared((64, 4), "float32")
            idx = T.alloc_shared((32,), "int32")
            c = T.alloc_shared((32, 128), "float32")
            T.copy(A, a)
            if not row_record:
                T.copy(S, sf)
            T.copy(I, idx)
            with T.SimdVF(lanes=64):
                for q in T.serial(32):
                    slot = idx[q]
                    for g in T.serial(4):
                        for j in T.Parallel(32):
                            if row_record:
                                c[q, g * 32 + j] = a[slot, g * 32 + j] * a[slot, 128 + g]
                            else:
                                c[q, g * 32 + j] = a[slot, g * 32 + j] * sf[slot, g]
            T.copy(c, C)

    return main


def vectorized(func):
    _, captured = lower_case("parallel_pack", func)
    return captured["VectorizeParallelToPTO_after"]


@pytest.mark.pto
@pytest.mark.parametrize("kind,index_mode", [("gather", "inline"), ("gather", "bound_cast"), ("scatter", "inline")])
def test_b16_native_index_rejects_potentially_wrapping_vector_cast(kind, index_mode):
    with pytest.raises(Exception, match="B16 indexed access cannot prove the input of a vector int32-to-int16 cast fits in int16"):
        vectorized(signed_b16_index_kernel(kind, index_mode))


@pytest.mark.pto
@pytest.mark.parametrize("index_mode", ["native", "bounded"])
def test_b16_native_index_accepts_safe_signed_carrier(index_mode):
    mod = vectorized(signed_b16_index_kernel("gather", index_mode))
    (gather,) = calls(mod, "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "int16x128"


@pytest.mark.pto
def test_b16_index_allows_scalar_cast_before_masking():
    mod = vectorized(signed_b16_index_kernel("gather", "scalar_cast"))
    (gather,) = calls(mod, "tl.vmi.vgather")
    assert str(gather.args[2].dtype) == "int16x128"


@pytest.mark.pto
def test_unsigned_exponent_uses_value_preserving_widen_shift_and_reinterpret():
    mod = vectorized(scale_bits_kernel(256))
    definitions = binding_values(mod)
    (widen,) = calls(mod, "tl.vmi.vcvt")
    (shift,) = calls(mod, "tl.vmi.vshls")
    (reinterpret,) = calls(mod, "tl.vmi.vinterpret_cast")
    (store,) = calls(mod, "tl.vmi.vstore")
    assert str(widen.args[0].dtype) == "uint8x64"
    assert str(widen.dtype) == "uint32x64"
    assert definitions[shift.args[0]].same_as(widen)
    assert (int(shift.args[1]), str(shift.args[1].dtype)) == (23, "int16")
    assert definitions[reinterpret.args[0]].same_as(shift)
    assert str(reinterpret.dtype) == "float32x64"
    assert definitions[store.args[0]].same_as(reinterpret)


@pytest.mark.pto
@pytest.mark.parametrize(
    "src_dtype,dst_dtype,lanes",
    [("int8", "uint8", 256), ("uint16", "int16", 128), ("int32", "uint32", 64), ("float32", "uint32", 64)],
)
def test_reinterpret_preserves_element_width_and_lanes(src_dtype, dst_dtype, lanes):
    mod = vectorized(reinterpret_kernel(src_dtype, dst_dtype, lanes))
    (reinterpret,) = calls(mod, "tl.vmi.vinterpret_cast")
    assert str(reinterpret.args[0].dtype) == f"{src_dtype}x{lanes}"
    assert str(reinterpret.dtype) == f"{dst_dtype}x{lanes}"
    assert not calls(mod, "tl.vmi.vcvt")


@pytest.mark.pto
@pytest.mark.parametrize("saved_value", [False, True], ids=["inline", "bind"])
@pytest.mark.parametrize("unary", [False, True], ids=["add", "sqrt"])
def test_lut_index_reinterprets_fp32_arithmetic(saved_value, unary):
    mod = vectorized(reinterpret_lut_kernel(saved_value, unary=unary))
    definitions = binding_values(mod)
    (arithmetic,) = calls(mod, "tl.vmi.vsqrt" if unary else "tl.vmi.vadd")
    (reinterpret,) = calls(mod, "tl.vmi.vinterpret_cast")
    (mask_bits,) = calls(mod, "tl.vmi.vand")
    (gather,) = calls(mod, "tl.vmi.vgather")
    value = definitions[reinterpret.args[0]]
    if saved_value:
        value = definitions[value]
    assert value.same_as(arithmetic)
    assert definitions[mask_bits.args[0]].same_as(reinterpret)
    assert definitions[gather.args[2]].same_as(mask_bits)
    assert str(gather.args[2].dtype) == "uint32x64"


@pytest.mark.pto
def test_uniform_integer_bit_ops_can_compute_before_broadcast():
    mod = vectorized(uniform_bits)
    (shift,) = calls(mod, "tirx.shift_right")
    (or_,) = calls(mod, "tirx.bitwise_or")
    assert str(shift.dtype) == str(or_.dtype) == "int32"
    assert shift.args[1].op.name == "tirx.bitwise_and"
    assert len(calls(mod, "tl.vmi.vbrc")) == 1
    assert not calls(mod, "tl.vmi.vshrs")
    assert not calls(mod, "tl.vmi.vand")
    assert not calls(mod, "tl.vmi.vor")


@pytest.mark.pto
def test_uint32_bitwise_xor_and_not_keep_dtype_and_tail_mask():
    mod = vectorized(uint32_bitwise_kernel(uniform=False))
    (xor,) = calls(mod, "tl.vmi.vxor")
    (complement,) = calls(mod, "tl.vmi.vnot")
    xor_store, not_store = calls(mod, "tl.vmi.vstore")
    assert str(xor.dtype) == str(xor.args[0].dtype) == str(xor.args[1].dtype) == "uint32x64"
    assert str(complement.dtype) == str(complement.args[0].dtype) == "uint32x64"
    assert xor.args[2].same_as(xor_store.args[3])
    assert complement.args[1].same_as(not_store.args[3])


@pytest.mark.pto
@pytest.mark.parametrize("dtype,right", [("uint8", False), ("int8", True), ("uint16", True), ("int32", False)])
def test_varying_shift_keeps_dtype_and_tail_mask(dtype, right):
    mod = vectorized(varying_shift_kernel(dtype, right=right))
    (shift,) = calls(mod, "tl.vmi.vshr" if right else "tl.vmi.vshl")
    (store,) = calls(mod, "tl.vmi.vstore")
    assert str(shift.dtype) == str(shift.args[0].dtype) == str(shift.args[1].dtype) == f"{dtype}x64"
    assert shift.args[2].same_as(store.args[3])


@pytest.fixture(scope="module")
def verify_input():
    _, captured = lower_case("pack_verify_base", indexed_kernel())
    return captured["VerifyParallelToPTO_before"]


@pytest.mark.pto
@pytest.mark.parametrize("stride", [0, 1], ids=["uniform", "continuous"])
def test_inline_address_rejects_uniform_float_arithmetic(verify_input, stride):
    def make_body(i, payload, index, output):
        value = payload[0] + tvm.tirx.FloatImm("float32", 1)
        base = tvm.tirx.reinterpret("uint32", value) & tvm.tirx.IntImm("uint32", 15)
        offset = tvm.tirx.Cast("int32", base) + stride * i
        return tvm.tirx.BufferStore(output, payload[offset], [i])

    mod = replace_unit_body(verify_input, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="runtime lane-uniform float"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
@pytest.mark.parametrize("count", [-1, 32])
def test_invalid_shift_counts_fail_verification(verify_input, count):
    def make_body(i, payload, index, output):
        amount = tvm.tirx.IntImm("int32", count)
        shift = tvm.tirx.call_intrin("int32", tvm.ir.Op.get("tirx.shift_left"), index[i], amount)
        return tvm.tirx.BufferStore(output, tvm.tirx.Cast("float32", shift), [i])

    mod = replace_unit_body(verify_input, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="(?is)shift.*(count|amount|constant|range)"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["signed_to_float", "cross_width"])
def test_reinterpret_rejects_unsupported_types(verify_input, kind):
    def make_body(i, payload, index, output):
        op = tvm.ir.Op.get("tirx.reinterpret")
        if kind == "signed_to_float":
            value = tvm.tirx.call_intrin("float32", op, index[i])
        else:
            value = tvm.tirx.Cast("float32", tvm.tirx.call_intrin("uint16", op, payload[i]))
        return tvm.tirx.BufferStore(output, value, [i])

    mod = replace_unit_body(verify_input, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="(?is)reinterpret"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
@pytest.mark.parametrize("kind,extent", [("b32", 256), ("b16", 256), ("b16", 75), ("b16", 62), ("b16_wide", 128)])
def test_nibble_lut_gathers_then_scatters_each_nibble(kind, extent):
    mod = vectorized(byte_lut_kernel(kind, extent))
    (shift,) = calls(mod, "tl.vmi.vshrs")
    gathered = calls(mod, "tl.vmi.vgather")
    scattered = calls(mod, "tl.vmi.vscatter")
    assert len(calls(mod, "tl.vmi.vload")) == 1
    assert len(calls(mod, "tl.vmi.vand")) == (1 if kind == "b16" else 2)
    assert str(shift.dtype) == {"b32": "int32x64", "b16": "uint16x128", "b16_wide": "int32x128"}[kind]
    assert (int(shift.args[1]), str(shift.args[1].dtype)) == (4, "int16")
    assert len(gathered) == len(scattered) == 2
    expected_offset = {"int32x64", "uint32x64"} if kind == "b32" else {"uint16x128"}
    assert {str(gather.args[2].dtype) for gather in gathered} <= expected_offset
    assert all(gather.args[3].same_as(scatter.args[4]) for gather, scatter in zip(gathered, scattered))


@pytest.mark.pto
def test_stride_two_reads_use_masked_gathers():
    mod = vectorized(alternating_kernel(75, 160))
    assert len(calls(mod, "tl.vmi.vgather")) == 2
    assert not calls(mod, "tl.vmi.vload")


@pytest.mark.pto
@pytest.mark.parametrize("row_record", [True, False])
def test_dynamic_row_scale_uses_contiguous_values_and_broadcast(row_record):
    mod = vectorized(row_scale_kernel(row_record))
    for operation in ("vload", "vbrc", "vmul", "vstore"):
        assert len(calls(mod, "tl.vmi." + operation)) == 1
    assert not calls(mod, "tl.vmi.vgather")
