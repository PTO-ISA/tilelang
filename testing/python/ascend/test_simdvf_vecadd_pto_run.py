"""pytest test for SimdVF vecadd PTO runnable path."""

import os
import sys

import torch
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang

from examples.ascend.example_simdvf_vecadd_lower import ref_program, vector_add


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")

    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


def test_pto_run_kernel():
    _require_npu_runtime()

    n = 8192 * 64 * 2

    # Auto-schedule must stay enabled: InsertSync (MTE sync) only runs in the
    # auto-schedule branch of the pipeline, so disabling it yields garbage
    # numerics on every pto codegen path (pre-existing pipeline behavior,
    # recorded in docs/vectorize_parallel_to_pto_dev_log.md 2026-09-14).
    kernel = tilelang.compile(
        vector_add(n),
        target="pto",
        out_idx=-1,
    )

    device = torch.device("npu")
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = kernel(a, b)
    torch.npu.synchronize()

    assert torch.equal(c, ref_program(a, b))


if __name__ == "__main__":
    test_pto_run_kernel()
