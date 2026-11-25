"""NPU regressions for automatic PTO Parallel lowering of packed bytes."""

import pytest
import torch

import tilelang.ascend.language as T

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_pack import (
    alternating_kernel,
    bounded_byte_index,
    byte_lut_kernel,
    reinterpret_kernel,
    reinterpret_lut_kernel,
    right_shift_kernel,
    row_scale_kernel,
    scale_bits_kernel,
    scale_pack_kernel,
    signed_b16_index_kernel,
    uint32_bitwise_kernel,
    uniform_bits,
    varying_shift_kernel,
)

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_index_bounds_wide_values_before_narrowing(kind, npu):
    source = torch.arange(128, dtype=torch.float16) - 50
    permutation = (torch.arange(128, dtype=torch.int32) * 37) % 128
    indices = torch.where(permutation < 64, permutation - 65536, permutation + 65536)
    expected = source[permutation.long()]
    if kind == "scatter":
        expected = torch.empty_like(source)
        expected[permutation.long()] = source
    output = torch.full_like(source, -123)
    _run(f"pack_b16_bounded_{kind}", signed_b16_index_kernel(kind, "bounded"), [indices, source, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("extent", [256, 150])
def test_unsigned_exponent_decodes_each_fp32_bit_pattern(extent, npu):
    encodings = torch.arange(256, dtype=torch.int32)
    output = torch.full((256,), -23.0, dtype=torch.float32)
    expected = output.clone()
    expected.view(torch.int32)[:extent] = encodings[:extent] << 23
    _run(f"pack_exponent_{extent}", scale_bits_kernel(extent), [encodings.to(torch.uint8), output], expected, npu, compare_bits=True)


@pytest.mark.pto
@pytest.mark.parametrize("bits", [16, 32])
def test_signed_and_unsigned_right_shift_high_bit(bits, npu):
    lanes = 2048 // bits
    values = torch.arange(lanes, dtype=torch.int64) - lanes // 2
    dtype = torch.int16 if bits == 16 else torch.int32
    output = torch.empty((2, lanes), dtype=dtype)
    expected = torch.stack((values >> 3, (values % (1 << bits)) >> 3)).to(dtype)
    _run(f"pack_right_shift_{bits}", right_shift_kernel(bits), [values.to(dtype), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("uniform", [False, True], ids=["varying", "uniform"])
def test_uint32_bitwise_xor_and_not_preserve_bits_and_tail(uniform, npu):
    values = (torch.arange(128, dtype=torch.int64) * 0x9E3779B9) & 0xFFFFFFFF
    values[:4] = torch.tensor([0x80000000, 0, 0xFFFFFFFF, 0xAAAAAAAA])
    output = torch.full((2, 128), 43, dtype=torch.uint32)
    expected = output.to(torch.int64)
    x = values[:1] if uniform else values[:65]
    expected[0, :65] = (x >> 23) ^ x
    expected[1, :65] = x ^ 0xFFFFFFFF
    _run(f"uint32_bitwise_{uniform}", uint32_bitwise_kernel(uniform), [values.to(output.dtype), output], expected.to(output.dtype), npu)


@pytest.mark.pto
@pytest.mark.parametrize("accumulate_invalid", [True, False], ids=["invalid_bits", "pack_exponents"])
def test_scale_pack_accumulates_two_rows_and_preserves_tail(accumulate_invalid, npu):
    col = torch.arange(128, dtype=torch.int64)
    exponents = torch.stack(((37 + col) % 256, (189 + col * 3) % 256))
    words = exponents << 23
    words[0, 1::2] |= 0x80000000
    words[1] |= col % 17
    output = torch.full((128,), 0x58, dtype=torch.uint32 if accumulate_invalid else torch.uint16)
    expected = output.to(torch.int64)
    if accumulate_invalid:
        expected[:65] |= (words[0, :65] | words[1, :65]) & 0x807FFFFF
    else:
        expected[:65] |= (words[0, :65] >> 23) | ((words[1, :65] >> 23) << 8)
    args = [words.to(torch.uint32), output]
    _run(f"scale_pack_{accumulate_invalid}", scale_pack_kernel(accumulate_invalid), args, expected.to(output.dtype), npu)


@pytest.mark.pto
def test_bitwise_consumer_of_unsigned_max(npu):
    @T.prim_func
    def main(A: T.Tensor((64,), "uint32"), C: T.Tensor((64,), "uint32")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "uint32")
            maximum = T.alloc_shared((1,), "uint32")
            c = T.alloc_shared((64,), "uint32")
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                T.reduce_max(a, maximum, dim=0)
                for i in T.Parallel(64):
                    saved = T.bind(maximum[0])
                    c[i] = a[i] | (saved & T.uint32(0x807FFFFF))
            T.copy(c, C)

    values = 0x3F800000 | torch.arange(64, dtype=torch.int64)
    values[3] = 0xC0800001
    output = torch.empty(64, dtype=torch.uint32)
    expected = values | (int(values.max()) & 0x807FFFFF)
    _run("pack_bitwise_max", main, [values.to(output.dtype), output], expected.to(output.dtype), npu)


@pytest.mark.pto
@pytest.mark.parametrize("dtype,right", [(torch.uint8, False), (torch.int8, True), (torch.uint16, True), (torch.int32, False)])
def test_varying_shift_preserves_tail(dtype, right, npu):
    values = torch.arange(128, dtype=torch.int64) | 0x80
    values = values.to(dtype)
    if dtype == torch.uint16:
        values = (values.to(torch.int64) | 0x8000).to(dtype)
    counts = torch.arange(128, dtype=torch.int64) % 7
    output = torch.full((128,), 43, dtype=dtype)
    expected = output.clone()
    wide = values[:65].to(torch.int64)
    expected[:65] = (wide >> counts[:65] if right else wide << counts[:65]).to(dtype)
    args = [values, counts.to(dtype), output]
    kernel = varying_shift_kernel(str(dtype).removeprefix("torch."), right=right)
    _run(f"pack_varying_shift_{dtype}_{right}", kernel, args, expected, npu)


@pytest.mark.pto
def test_uniform_runtime_shift(npu):
    values = torch.zeros(64, dtype=torch.int32)
    values[:3] = torch.tensor([-40, 3, 0x100], dtype=values.dtype)
    output = torch.empty_like(values)
    expected = torch.full_like(values, ((-40 >> 3) & 15) | 0x100)
    _run("pack_uniform_runtime_shift", uniform_bits, [values, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("src_dtype,dst_dtype,bits", [("uint32", "float32", -0x80000000), ("float32", "uint32", 0x7FC12345)])
def test_uniform_reinterpret_preserves_runtime_bits(src_dtype, dst_dtype, bits, npu):
    lanes = 64
    source = torch.zeros(lanes, dtype=torch.int32)
    source[0] = bits
    source = source.view(getattr(torch, src_dtype))
    output = torch.empty(lanes, dtype=getattr(torch, dst_dtype))
    expected = torch.full((lanes,), bits, dtype=torch.int32).view(output.dtype)
    kernel = reinterpret_kernel(src_dtype, dst_dtype, lanes, uniform=True)
    _run(f"pack_uniform_bitcast_{src_dtype}", kernel, [source, output], expected, npu, compare_bits=dst_dtype == "float32")


@pytest.mark.pto
@pytest.mark.parametrize("saved_value", [False, True], ids=["inline", "bind"])
def test_lut_index_reinterprets_fp32_arithmetic_on_npu(saved_value, npu):
    source = (0x3F800000 + 3 * torch.arange(64, dtype=torch.int32)).view(torch.float32)
    lut = 11 - 3 * torch.arange(16, dtype=torch.float32)
    output = torch.empty_like(source)
    offsets = (source + 1).view(torch.int32) & 15
    expected = lut[offsets.to(torch.int64)]
    _run(f"pack_reinterpret_lut_{saved_value}", reinterpret_lut_kernel(saved_value), [source, lut, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind,extent", [("b32", 256), ("b16", 256), ("b16", 75), ("b16", 62), ("b16_wide", 128)])
def test_nibble_lut_keeps_even_odd_order_and_bit_patterns(kind, extent, npu):
    byte_dtype = torch.uint8 if kind == "b16" else torch.int8
    result_dtype = torch.float32 if kind == "b32" else torch.float16
    lut_start = int(kind == "b16_wide")
    lut_size = 32 if kind == "b16_wide" else 16
    table = (torch.arange(lut_size, dtype=torch.float32) * -3 + 11).to(result_dtype)
    table[lut_start] = -0.0
    packed = (torch.arange(256) * 37 + 201) % 256
    output = torch.full((512,), 703, dtype=result_dtype)
    expected = output.clone()
    expected[: 2 * extent : 2] = table[lut_start + packed[:extent] % 16]
    expected[1 : 2 * extent : 2] = table[lut_start + packed[:extent] // 16]
    _run(
        f"pack_lut_{kind}_{extent}", byte_lut_kernel(kind, extent), [packed.to(byte_dtype), table, output], expected, npu, compare_bits=True
    )


@pytest.mark.pto
@pytest.mark.parametrize("extent,width", [(128, 256), (75, 160)])
def test_stride_two_reads_preserve_lane_order_and_inactive_tail(extent, width, npu):
    source = torch.arange(width, dtype=torch.float16) - 37
    output = torch.full((2, 128), -511, dtype=torch.float16)
    expected = output.clone()
    expected[0, :extent] = source[: 2 * extent : 2]
    expected[1, :extent] = source[1 : 2 * extent : 2]
    _run(f"pack_stride_two_{extent}_{width}", alternating_kernel(extent, width), [source, output], expected, npu)


@pytest.mark.pto
def test_b16_gather_narrows_provably_bounded_int32_indices(npu):
    table = torch.arange(256, dtype=torch.float16) - 37
    indices = (torch.arange(128) * 13 + 255) % 256
    output = torch.empty(128, dtype=torch.float16)
    _run("pack_bounded_byte_index", bounded_byte_index, [table, indices.to(torch.uint8), output], table[indices], npu)


@pytest.mark.pto
@pytest.mark.parametrize("row_record", [True, False])
def test_padded_row_scale_uses_selected_row_and_group(row_record, npu):
    values = (torch.arange(65 * 128, dtype=torch.float32) % 127 - 63).reshape(65, 128)
    scales = (torch.arange(64 * 4, dtype=torch.float32) % 17 - 8).reshape(64, 4)
    index = (63 - 13 * torch.arange(32)) % 64
    packed = torch.cat((values, torch.cat((scales, torch.zeros((1, 4))), dim=0)), dim=1) if row_record else values
    output = torch.empty((32, 128), dtype=torch.float32)
    expected = values[index] * scales[index].repeat_interleave(32, dim=1)
    _run(f"pack_row_scale_{row_record}", row_scale_kernel(row_record), [packed, scales, index.to(torch.int32), output], expected, npu)
