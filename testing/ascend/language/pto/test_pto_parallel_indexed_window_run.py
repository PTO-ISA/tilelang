"""NPU verification of block-dependent windows in a large B16 owner."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as npu
from test_pto_parallel_indexed_window import (
    affine_large_window_kernel,
    alternate_axis_window_kernel,
    outer_range_window_kernel,
    saved_index_window_kernel,
    saved_large_window,
)


@pytest.mark.pto
def test_b16_window_with_block_dependent_base_numerics(npu):
    source = (torch.arange(70000, dtype=torch.int64) % 113 - 56).to(torch.float16)
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 256
    output = torch.full((2, 128), 211, dtype=torch.float16)
    expected = torch.stack([source[index], source[35000 + index]])
    _run("outer_range_window", outer_range_window_kernel(flat=True), [source, index.to(torch.uint16), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_alternate_axis_with_scalar_row_base_numerics(kind, npu):
    if kind == "gather":
        source = (torch.arange(128 * 600, dtype=torch.int64) % 113 - 56).reshape(128, 600).to(torch.float16)
        output = torch.full((2, 128), 211, dtype=torch.float16)
        expected = source[:, :4:2].T.contiguous()
    else:
        source = (torch.arange(256, dtype=torch.int64) % 113 - 56).reshape(2, 128).to(torch.float16)
        output = torch.full((128, 600), 211, dtype=torch.float16)
        expected = output.clone()
        expected[:, :4:2] = source.T
    _run(f"alternate_axis_{kind}", alternate_axis_window_kernel(kind), [source, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize(
    "kind,address_form,index_mode,origin,stride",
    [
        ("gather", "flat", None, 66000, 2),
        ("scatter", "flat", None, 66000, 2),
        ("gather", "flat", "cast_address", 66000, 2),
        ("scatter", "shifted", "round_trip", 66000, 2),
        ("gather", "shifted", "reverse", 66000, 2),
        ("scatter", "flat", "reverse", 0, 256),
        ("gather", "shifted", "int16", 66000, 2),
    ],
)
def test_b16_affine_window_numerics(kind, address_form, index_mode, origin, stride, npu):
    positions = origin + stride * torch.arange(270)
    if index_mode == "reverse":
        positions = positions.flip((0,))
    size_in, size_out = (70000, 384) if kind == "gather" else (384, 70000)
    source = (torch.arange(size_in) % 113 - 56).to(torch.float16)
    output = torch.full((size_out,), 211, dtype=torch.float16)
    expected = output.clone()
    if kind == "gather":
        expected[:270] = source[positions]
    else:
        expected[positions] = source[:270]
    kernel = affine_large_window_kernel(kind, address_form, index_mode, origin=origin, stride=stride)
    _run(f"affine_large_window_{kind}_{address_form}_{index_mode}", kernel, [source, output], expected, npu)


@pytest.mark.pto
def test_b16_rebased_saved_index_numerics(npu):
    source = torch.arange(256, dtype=torch.float16)
    slots = 66000 + 2 * torch.arange(128)
    index = torch.zeros(70000, dtype=torch.int32)
    index[slots] = ((31 * torch.arange(128) + 1) % 256).to(torch.int32)
    output = torch.full((128,), 211, dtype=torch.float16)
    expected = source[index[slots].to(torch.int64)]
    _run("saved_large_window", saved_large_window, [source, index.to(torch.uint16), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("index_dtype", ["uint16", "int8"])
def test_b16_rebased_address_preserves_saved_index_arithmetic_numerics(index_dtype, npu):
    source = torch.zeros(70000, dtype=torch.float16)
    source[66000:70000] = torch.arange(4000, dtype=torch.float16)
    index = torch.arange(128, dtype=torch.int64)
    if index_dtype == "uint16":
        index = (37 * index + 17) % 2000
        offsets = index
    else:
        index = 2 * index - 1
        index = index.to(torch.int8)
        offsets = (index.to(torch.int64) % 256) + 1
    output = torch.full((128,), 211, dtype=torch.float16)
    expected = source[66000 + offsets]
    kernel = saved_index_window_kernel("shifted", index_dtype, explicit_widen=index_dtype == "uint16")
    _run(f"saved_index_{index_dtype}_window", kernel, [source, index.to(getattr(torch, index_dtype)), output], expected, npu)
