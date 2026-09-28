"""On-device coverage for VMI packed-FP4 conversion in both directions."""

from __future__ import annotations

import shutil

import pytest
import torch

import tilelang
import tilelang.ascend.language as T


LANES = 256
ACTIVE_PHYSICAL_LANES = 37
FP4_CODES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _vmi_bf16_to_fp4():
    @T.prim_func
    def main(A: T.Buffer((LANES,), "bfloat16"), B: T.Buffer((LANES,), "float4_e2m1fn")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((LANES,), "bfloat16")
            b_ub = T.alloc_shared((LANES,), "float4_e2m1fn")
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(LANES // 2, size=LANES // 2)
                bf16 = T.vmi.vload(a_ub[0], size=LANES)
                fp4 = T.vmi.vcvt(bf16, "float4_e2m1fn")
                T.vmi.vstore(fp4, b_ub[0], mask)
            T.copy(b_ub, B)

    return main


def _vmi_bf16_to_fp4_partial():
    @T.prim_func
    def main(
        A: T.Buffer((LANES,), "bfloat16"),
        B: T.Buffer((LANES,), "float4_e2m1fn"),
        C: T.Buffer((LANES,), "float4_e2m1fn"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((LANES,), "bfloat16")
            c_ub = T.alloc_shared((LANES,), "float4_e2m1fn")
            T.copy(A, a_ub)
            T.copy(B, c_ub)
            with T.SimdVF():
                # FP4 stores consume physical f4e2m1x2 predicate lanes.
                mask = T.vmi.create_mask(ACTIVE_PHYSICAL_LANES, size=LANES // 2)
                bf16 = T.vmi.vload(a_ub[0], size=LANES)
                fp4 = T.vmi.vcvt(bf16, "float4_e2m1fn")
                T.vmi.vstore(fp4, c_ub[0], mask)
            T.copy(c_ub, C)

    return main


def _vmi_fp4_to_bf16(active_physical_lanes=LANES // 2):
    @T.prim_func
    def main(
        A: T.Buffer((LANES,), "float4_e2m1fn"),
        B: T.Buffer((LANES,), "bfloat16"),
        C: T.Buffer((LANES,), "bfloat16"),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((LANES,), "float4_e2m1fn")
            c_ub = T.alloc_shared((LANES,), "bfloat16")
            T.copy(A, a_ub)
            T.copy(B, c_ub)
            with T.SimdVF():
                # The source is loaded in logical FP4 scalar lanes. Each
                # physical predicate lane covers one packed pair.
                fp4 = T.vmi.vload(a_ub[0], size=LANES)
                bf16 = T.vmi.vcvt(fp4, "bfloat16")
                mask = T.vmi.create_mask(active_physical_lanes * 2, size=LANES)
                T.vmi.vstore(bf16, c_ub[0], mask)
            T.copy(c_ub, C)

    return main


def _fp4_test_data():
    repeats = (LANES + FP4_CODES.numel() - 1) // FP4_CODES.numel()
    codes = FP4_CODES.repeat(repeats)[:LANES]
    values = FP4_VALUES.repeat(repeats)[:LANES].to(torch.bfloat16)
    return values, codes[0::2] | (codes[1::2] << 4)


def _fp4_reverse_test_data():
    # Include both signed zeros and every finite E2M1 code. PyTorch's E2M1
    # storage also accepts all sixteen raw nibble encodings.
    codes = torch.arange(16, dtype=torch.uint8).repeat((LANES + 15) // 16)[:LANES]
    magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    reference = magnitude[(codes & 7).long()]
    reference = torch.where((codes & 8) != 0, -reference, reference).to(torch.bfloat16)
    return (codes[0::2] | (codes[1::2] << 4)).view(torch.float4_e2m1fn_x2), reference


@pytest.mark.pto
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "npu") or not torch.npu.is_available() or shutil.which("ptoas") is None,
    reason="NPU FP4 storage is unavailable",
)
def test_vmi_bf16_to_fp4_e2e():
    values, expected_bytes = _fp4_test_data()
    kernel = tilelang.compile(_vmi_bf16_to_fp4(), target="pto", out_idx=-1)

    result = kernel(values.npu())
    torch.npu.synchronize()

    assert result.dtype == torch.float4_e2m1fn_x2
    assert result.shape == (LANES // 2,)
    torch.testing.assert_close(result.view(torch.uint8).cpu(), expected_bytes)


@pytest.mark.pto
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "npu") or not torch.npu.is_available() or shutil.which("ptoas") is None,
    reason="NPU FP4 storage is unavailable",
)
def test_vmi_bf16_to_fp4_physical_partial_mask_e2e():
    values, packed_values = _fp4_test_data()
    initial_bytes = torch.full((LANES // 2,), 0xFF, dtype=torch.uint8, device="npu")
    initial = initial_bytes.view(torch.float4_e2m1fn_x2)
    kernel = tilelang.compile(_vmi_bf16_to_fp4_partial(), target="pto", out_idx=-1)

    result = kernel(values.npu(), initial)
    torch.npu.synchronize()

    expected_bytes = initial_bytes.cpu()
    expected_bytes[:ACTIVE_PHYSICAL_LANES] = packed_values[:ACTIVE_PHYSICAL_LANES]
    torch.testing.assert_close(result.view(torch.uint8).cpu(), expected_bytes)


@pytest.mark.pto
@pytest.mark.parametrize("active_physical_lanes", [LANES // 2, ACTIVE_PHYSICAL_LANES])
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "npu") or not torch.npu.is_available() or shutil.which("ptoas") is None,
    reason="NPU FP4 storage is unavailable",
)
def test_vmi_fp4_to_bf16_e2e(active_physical_lanes):
    packed, expected = _fp4_reverse_test_data()
    initial = torch.full((LANES,), 123, dtype=torch.bfloat16)
    kernel = tilelang.compile(_vmi_fp4_to_bf16(active_physical_lanes), target="pto", out_idx=-1)

    result = kernel(packed.npu(), initial.npu())
    torch.npu.synchronize()

    active_logical_lanes = active_physical_lanes * 2
    reference = initial
    reference[:active_logical_lanes] = expected[:active_logical_lanes]
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
