"""NPU verification of arithmetic inside full indexed offsets."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_offset_expr import (
    chunk_division_offset,
    floor_divmod_kernel,
    nd_to_nz_kernel,
    nested_b16_rebased_gather_kernel,
    scalar_floor_coordinate_kernel,
    scalar_select_coordinate_cast_kernel,
    varying_floor_offset_kernel,
)

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("access", ["gather", "scatter"])
def test_nd_to_nz_numerics(access, npu):
    source = torch.arange(512, dtype=torch.float16).reshape(16, 32)
    expected = source.reshape(16, 2, 16).permute(1, 0, 2).contiguous()
    output = torch.full_like(expected, -1)
    _run(f"simd_parallel_nd_to_nz_{access}", nd_to_nz_kernel(access), [source, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize(
    "dtype,divisor,lanes,extent",
    [(dtype, None, 64, 128) for dtype in ("int8", "int16", "int32", "uint8", "uint16", "uint32")]
    + [(dtype, None, 256, 512) for dtype in ("uint8", "uint16", "uint32")]
    + [("int8", None, 256, 480), ("int32", 16, 64, 128), ("int32", 3, 64, 128)],
)
def test_floor_divmod_numerics(dtype, divisor, lanes, extent, npu):
    torch_dtype = getattr(torch, dtype)
    limits = torch.iinfo(torch_dtype)
    if limits.min < 0:
        values = [limits.min, limits.min + 1, -17, -16, -15, -1, 0, 1, 15, 16, 17, limits.max]
        divisors = [1, -1, 3, -3, 16, -16, limits.min, limits.max]
    else:
        high_bit = (limits.max + 1) // 2
        values = [0, 1, 3, high_bit - 1, high_bit, high_bit + 1, limits.max]
        divisors = [1, 2, 3, 17, high_bit - 1, high_bit, high_bit + 1, limits.max]
    if limits.bits == 32:
        values += [-(2**24) - 1, 2**24 + 1] if limits.min < 0 else [2**24 + 1]
    pairs = [(a, b) for a in values for b in divisors if (a, b) != (limits.min, -1)]
    source, rhs = torch.tensor(pairs, dtype=torch.int64).repeat((extent + len(pairs) - 1) // len(pairs), 1)[:extent].T
    if divisor is not None:
        rhs = torch.full_like(rhs, divisor)
    expected = torch.stack((torch.div(source, rhs, rounding_mode="floor"), torch.remainder(source, rhs))).to(torch_dtype)
    output = torch.empty_like(expected)
    args = [source.to(torch_dtype).contiguous(), rhs.to(torch_dtype).contiguous(), output]
    _run(
        f"simd_parallel_floor_divmod_{dtype}_{divisor}_{lanes}",
        floor_divmod_kernel(dtype, divisor, lanes, extent=extent),
        args,
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize(
    "dtype,dividends,divisors,remainders",
    [
        ("int32", [-8, 0, 8, -8, 8, -8, 8, -9], [0, 0, 0, 3, -3, -3, 3, 3], [-1, -1, -1, 1, -1, -2, 2, 0]),
        ("uint32", [0, 8, 8, 9], [0, 0, 3, 3], [2**32 - 1, 2**32 - 1, 2, 0]),
    ],
)
def test_floor_mod_zero_divisor_numerics(dtype, dividends, divisors, remainders, npu):
    # The asc-devkit SIMD FloorMod example returns the all-ones value for a zero divisor.
    torch_dtype = getattr(torch, dtype)
    repeats = 128 // len(dividends)
    source = torch.tensor(dividends, dtype=torch_dtype).repeat(repeats)
    rhs = torch.tensor(divisors, dtype=torch_dtype).repeat(repeats)
    expected = torch.tensor([remainders], dtype=torch_dtype).repeat(1, repeats)
    kernel = floor_divmod_kernel(dtype, None, 64, "remainder_only")
    _run(f"simd_parallel_floor_mod_zero_divisor_{dtype}", kernel, [source, rhs, torch.empty_like(expected)], expected, npu)


@pytest.mark.pto
def test_floor_divmod_reloads_an_operand_written_through_a_view(npu):
    lanes = 64
    source = torch.arange(-lanes, lanes, dtype=torch.int32)
    rhs = torch.where(source % 2 == 0, 3, -5).to(torch.int32)
    expected = torch.stack((source // rhs, (source + 1) % rhs))
    _run(
        "simd_parallel_floor_divmod_overwritten",
        floor_divmod_kernel("int32", None, lanes, "overwritten"),
        [source, rhs, torch.empty_like(expected)],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather_floordiv", "gather_floormod", "scatter_floordiv"])
def test_floor_index_with_non_power_of_two_divisor_numerics(kind, npu):
    source = torch.arange(256, dtype=torch.float32)
    lanes = torch.arange(256)
    permutation = (lanes * 37 + 11) % 256
    output = torch.full_like(source, -1)
    if kind == "gather_floordiv":
        index = torch.zeros_like(lanes)
        expected = source[lanes // 3]
    elif kind == "gather_floormod":
        index = permutation % 254
        expected = source[index + lanes % 3]
    else:
        index = permutation * 3 + lanes % 3
        expected = torch.empty_like(source)
        expected[permutation] = source
    _run(f"simd_parallel_floor_index_{kind}", varying_floor_offset_kernel(kind, 3), [source, index.to(torch.int32), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("extent", [256, 150])
def test_scalar_select_cast_keeps_indexed_address_semantics(extent, npu):
    source = torch.arange(512, dtype=torch.float32) - 123
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 128
    index[0] = 0
    output = torch.full((256,), -777.0)
    expected = output.clone()
    offset = index[:extent] + 1
    expected[:extent] = source[offset]
    index[extent:] = -1
    _run(
        f"scalar_select_coordinate_cast_{extent}",
        scalar_select_coordinate_cast_kernel(extent),
        [source, index.to(torch.int32), output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_nested_b16_rebased_gather_numerics(npu):
    source = torch.arange(70000, dtype=torch.int64) % 256
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 256
    expected = index.clone()
    for _ in range(14):
        expected = source[66000 + expected]
    output = torch.full((128,), 211, dtype=torch.uint16)
    _run(
        "nested_b16_rebased_gather",
        nested_b16_rebased_gather_kernel,
        [source.to(torch.uint16), index.to(torch.uint16), output],
        expected.to(torch.uint16),
        npu,
    )


@pytest.mark.pto
def test_scalar_floor_coordinate_gather_numerics(npu):
    # `bx % 2` stays one scalar coordinate per launch block and adds to the
    # loaded offsets; every block must gather with its own broadcast value.
    source = torch.arange(256, dtype=torch.float32) - 123
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 200
    output = torch.full((4, 256), -777.0)
    expected = output.clone()
    for bx in range(4):
        expected[bx, :128] = source[index + bx % 2]
    _run(
        "scalar_floor_coordinate_gather",
        scalar_floor_coordinate_kernel("gather"),
        [source, index.to(torch.int32), output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_chunk_division_offset_numerics(npu):
    source = torch.arange(512, dtype=torch.int32) - 123
    columns = torch.arange(256)
    index = (columns * 37 + 11) % 256
    output = torch.full((4, 256), -777, dtype=torch.int32)
    expected = source[index + columns // 128].expand(4, 256).contiguous()
    _run("chunk_division_offset", chunk_division_offset, [source, index.to(torch.int32), output], expected, npu)
