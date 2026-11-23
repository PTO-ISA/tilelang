"""NPU checks for byte gather/scatter semantics and byte offset signedness."""

import pytest
import torch

from test_pto_parallel_gather_scatter import indexed_kernel
from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_bytes import (
    byte_gather_kernel,
    byte_gather_scatter_chain_kernel,
    byte_offset_view_kernel,
    byte_scatter_kernel,
    cast_bind_kernel,
    explicitly_widened_byte_offset_kernel,
    version_slot_view_kernel,
)

npu = _npu

pytestmark = pytest.mark.pto


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_cast_bind_numerics(dtype, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    output = torch.zeros(256, dtype=torch.int32)
    _run(f"cast_bind_{dtype}", cast_bind_kernel(dtype), [source, output], source.to(torch.int32) + 1, npu)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("widen", [False, True])
def test_byte_gather_source_sign_and_tail(dtype, widen, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    # Includes -128, -1, 0, 1, 127 in the gathered source for signed payloads.
    offsets = (torch.arange(256, dtype=torch.int32) * 37 + 11) % 256
    offsets[150:] = 65535
    output = torch.full((256,), 73, dtype=torch.int32 if widen else source.dtype)
    expected = output.clone()
    values = source[offsets[:150].to(torch.int64)]
    expected[:150] = values.to(torch.int32) + 1 if widen else values
    _run(f"byte_gather_{dtype}_{widen}", byte_gather_kernel(dtype, widen=widen), [source, offsets.to(torch.uint16), output], expected, npu)


@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
def test_explicit_byte_offset_widening_numerics(index_dtype, npu):
    owner_size = 65536 if index_dtype == "int8" else 256
    source = (torch.arange(owner_size, dtype=torch.int64) % 113).to(torch.float16)
    offsets = ((torch.arange(128, dtype=torch.int32) * 37 + 131) % 256).to(getattr(torch, index_dtype))
    output = torch.empty(128, dtype=torch.float16)
    # An explicit uint16 cast sign-extends signed bytes before reinterpreting
    # them. Negative inputs address the end of the 65536-element owner.
    positions = offsets.to(torch.int64) % 65536
    _run(
        f"widen_byte_offset_{index_dtype}",
        explicitly_widened_byte_offset_kernel(index_dtype),
        [source, offsets, output],
        source[positions],
        npu,
    )


@pytest.mark.parametrize("payload", ["float16", "float32"])
@pytest.mark.parametrize("index_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_byte_offsets_reinterpret_before_widening(payload, index_dtype, kind, npu):
    source = torch.arange(256, dtype=torch.float32).to(getattr(torch, payload))
    # The signed representation is negative for positions 128..255. Offsets
    # are same-width unsigned positions before extension, as in PTODSL.
    positions = (torch.arange(256, dtype=torch.int32) * 37 + 131) % 256
    offsets = positions.to(getattr(torch, index_dtype))
    output = torch.full((256,), -19, dtype=source.dtype)
    expected = output.clone()
    if kind == "gather":
        expected[:150] = source[positions[:150].to(torch.int64)]
    else:
        expected[positions[:150].to(torch.int64)] = source[:150]
    _run(
        f"byte_offsets_{payload}_{index_dtype}_{kind}",
        indexed_kernel(kind, payload, 128, 150, index_dtype=index_dtype),
        [source, offsets, output],
        expected,
        npu,
    )


@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize(
    "payload,index_dtype,origin,stride,index_cast",
    [
        ("float32", "int8", 256, 2, None),
        ("float32", "int8", 256, 2, "uint8"),
        ("float32", "int8", 256, 2, "int32"),
        ("float16", "int8", 32, 1, "uint8"),
        ("float16", "uint8", 32, 1, None),
    ],
)
def test_byte_offset_view_numerics(kind, payload, index_dtype, origin, stride, index_cast, npu):
    values = torch.arange(256 * (stride + 1), dtype=getattr(torch, payload))
    positions = (37 * torch.arange(256) + 131) % 256
    offsets = positions.to(getattr(torch, index_dtype))
    if index_cast != "uint8":
        positions = offsets.to(torch.int64)
    targets = origin + stride * positions[:150]
    output = torch.full_like(values, -211)
    expected = output.clone()
    if kind == "gather":
        expected[:150] = values[targets]
    else:
        expected[targets] = values[:150]
    kernel = byte_offset_view_kernel(kind, index_cast, payload=payload, index_dtype=index_dtype, origin=origin, stride=stride)
    _run(f"byte_view_{kind}_{payload}_{index_dtype}_{index_cast}", kernel, [values, offsets, output], expected, npu)


@pytest.mark.parametrize(
    "payload,index_dtype,origin,index_cast,target",
    [("float32", "int8", 128, None, 127), ("float16", "int8", 32, "uint8", 287), ("float16", "uint8", 32, None, 287)],
)
def test_uniform_byte_scatter_with_view_origin(payload, index_dtype, origin, index_cast, target, npu):
    values = torch.arange(512, dtype=getattr(torch, payload))
    offsets = torch.full((256,), -1 if index_dtype == "int8" else 255, dtype=getattr(torch, index_dtype))
    output = torch.full_like(values, -211)
    expected = output.clone()
    expected[target] = values[0]
    kernel = byte_offset_view_kernel("scatter", index_cast, payload=payload, index_dtype=index_dtype, origin=origin, uniform_index=True)
    _run("uniform_byte_scatter", kernel, [values, offsets, output], expected, npu)


@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("scheduled,widen,origin", [(True, True, 0), (False, False, 32)])
def test_aligned_version_view_numerics(kind, scheduled, widen, origin, npu):
    values = torch.arange(4 * 256, dtype=torch.float16).reshape(4, 256)
    positions = (37 * torch.arange(128) + 17 * torch.arange(4)[:, None]) % 128
    output = torch.full_like(values, -211)
    expected = output.clone()
    if kind == "gather":
        expected[:, :128] = values.gather(1, origin + positions)
    else:
        expected.scatter_(1, origin + positions, values[:, :128])
    kernel = version_slot_view_kernel(kind, scheduled, widen, origin)
    _run(f"aligned_version_view_{kind}_{scheduled}", kernel, [values, positions.to(torch.uint8), output], expected, npu)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_scatter_source_sign_and_tail(dtype, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    # A permutation keeps every scattered target unique, so the result does
    # not depend on duplicate-offset write behavior.
    targets = (torch.arange(256, dtype=torch.int32) * 37 + 11) % 256
    # Keep the last inactive target in bounds to check its canary as well.
    targets[150:-1] = 65535
    output = torch.full((256,), 73, dtype=source.dtype)
    expected = output.clone()
    expected[targets[:150].to(torch.int64)] = source[:150]
    _run(f"byte_scatter_{dtype}", byte_scatter_kernel(dtype), [source, targets.to(torch.uint16), output], expected, npu)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_byte_gather_scatter_chain_numerics(dtype, npu):
    source = (torch.arange(256, dtype=torch.int32) - 128).to(getattr(torch, dtype))
    gather_offsets = ((torch.arange(256, dtype=torch.int32) * 37 + 11) % 256).to(torch.uint16)
    scatter_targets = ((torch.arange(256, dtype=torch.int32) * 53 + 7) % 256).to(torch.uint16)
    output = torch.full((256,), 73, dtype=source.dtype)
    expected = output.clone()
    expected[scatter_targets[:150].to(torch.int64)] = source[gather_offsets[:150].to(torch.int64)]
    _run(
        f"byte_chain_{dtype}",
        byte_gather_scatter_chain_kernel(dtype),
        [source, gather_offsets, scatter_targets, output],
        expected,
        npu,
    )
