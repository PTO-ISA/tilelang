"""In-tree regression for the GPU-shaped SimdVF Parallel+reduce top-k frontend.

Ported from PR #262 head (`pr262-tree-verify-9008cc1b`,
commit 9008cc1b285d80fa4275b18c623b9e15a4575813,
``examples/ascend/test_simdvf_parallel_topk_frontend.py``,
sha256 b780971fef2f208a...).  Adaptations for the 5B port:

* compile+contract runs under ``Verify -> Legalize -> Vectorize`` (the
  contract helper captures the three passes);
* the whole index chain is int32 (``IdxOut`` included), so the numeric
  comparison is int32 vs int32;
* numeric coverage adds the tie-break rule and the all--inf valid-score
  case (tail lanes must never win through the ``-inf`` padding). PR262's
  tournament repeats index 0 on all--inf inputs; this limitation is
  preserved, rather than claiming distinct TopK indices for that input.
"""

from __future__ import annotations

import os

import pytest
import torch

import tilelang.testing

os.environ.setdefault("TILELANG_DISABLE_DATA_RACE_CHECK", "1")

from example_simdvf_parallel_topk_frontend import (
    E,
    K_DEFAULT,
    L,
    _stage_capture,
    align,
    assert_lowered_contract,
    compile_frontend,
    ref_topk_indices,
)


@pytest.mark.parametrize(
    ("num_experts", "num_topk", "lanes"),
    [
        (256, 1, 128),
        (256, 8, 128),
        (256, 8, 64),
        (161, 8, 64),
    ],
)
def test_parallel_topk_frontend_pto_compile_contract(num_experts, num_topk, lanes):
    assert align(num_experts, lanes) >= num_experts
    with _stage_capture() as captured:
        compiled = compile_frontend(num_experts, num_topk, lanes)
    assert_lowered_contract(compiled, captured)


def _npu_available() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
        return True
    except Exception:  # noqa: BLE001
        return False


def _run_scores(scores: torch.Tensor, num_topk: int = K_DEFAULT, lanes: int = L):
    compiled = compile_frontend(scores.numel(), num_topk, lanes)
    out = compiled(scores.to(torch.device("npu")))
    torch.npu.synchronize()
    assert out.dtype == torch.int32, out.dtype
    return out.cpu()


@pytest.mark.parametrize(
    ("num_experts", "lanes"),
    [
        (256, 128),
        (256, 64),
        (161, 64),
    ],
)
def test_parallel_topk_frontend_pto_numeric(num_experts, lanes):
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    torch.manual_seed(0)
    scores = torch.randn(num_experts, dtype=torch.float32)
    got = _run_scores(scores, K_DEFAULT, lanes)
    exp = ref_topk_indices(scores, K_DEFAULT)
    assert torch.equal(got, exp), f"E={num_experts} L={lanes} got={got.tolist()} exp={exp.tolist()}"


def test_parallel_topk_frontend_tie_break_smaller_index_wins():
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    torch.manual_seed(1)
    scores = torch.full((E,), -5.0, dtype=torch.float32)
    # Several exact ties at the same peak value: 144, 33, 200, 7.
    for idx in (144, 33, 200, 7):
        scores[idx] = 3.0
    got = _run_scores(scores)
    exp = torch.tensor([7, 33, 144, 200, 0, 1, 2, 3], dtype=torch.int32)
    assert torch.equal(got, exp), f"got={got.tolist()} exp={exp.tolist()}"


def test_parallel_topk_frontend_valid_scores_all_neg_inf():
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    # PR262 replaces each selected score with -inf. On all--inf inputs the
    # smallest valid index therefore wins again; distinct indices are not
    # promised for this boundary. Check the literal source result.
    scores = torch.full((E,), float("-inf"), dtype=torch.float32)
    got = _run_scores(scores)
    exp = torch.zeros(K_DEFAULT, dtype=torch.int32)
    assert torch.equal(got, exp), f"got={got.tolist()} exp={exp.tolist()}"
    assert bool((got >= 0).all() and (got < E).all()), got


def test_parallel_topk_frontend_nondivisible_valid_scores_all_neg_inf():
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    # E=161 with L=64: the padded scores[161:192] are -inf as well; with all
    # valid scores -inf the i < E guard is what keeps the padding out.
    scores = torch.full((161,), float("-inf"), dtype=torch.float32)
    got = _run_scores(scores, K_DEFAULT, 64)
    exp = torch.zeros(K_DEFAULT, dtype=torch.int32)
    assert torch.equal(got, exp), f"got={got.tolist()} exp={exp.tolist()}"


if __name__ == "__main__":
    tilelang.testing.main()
