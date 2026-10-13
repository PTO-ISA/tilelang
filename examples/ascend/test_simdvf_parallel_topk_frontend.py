"""In-tree regression for the GPU-shaped SimdVF Parallel+reduce top-k frontend.

Ported from PR #262 head (`pr262-tree-verify-9008cc1b`,
commit 9008cc1b285d80fa4275b18c623b9e15a4575813,
``examples/ascend/test_simdvf_parallel_topk_frontend.py``,
sha256 b780971fef2f208a...).  Adaptations for the 5B port:

* compile+contract runs under ``Verify -> Legalize -> Vectorize`` (the
  contract helper captures the three passes);
* the GM ``IdxOut`` ABI is int64 (the PR262 ABI): the int32 winner chain
  (``idx_cand``/``best``/``idx_ub``) is widened in UB and published by a
  same-dtype int64 copy, so numeric results are compared as full int64
  values with a zero high half -- never by casting the device output back
  to int32;
* numeric coverage spans (E, L) x K x input class, adding the tie-break
  rule and the all--inf valid-score case (tail lanes must never win through
  the ``-inf`` padding). PR262's tournament repeats index 0 on all--inf
  inputs; this limitation is preserved, rather than claiming distinct TopK
  indices for that input.
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
    assert out.dtype == torch.int64, out.dtype
    return out.cpu()


def _assert_index_values(got, exp, num_experts, context):
    """Full int64 comparison: dtype first, then every value, then the ABI
    invariants (valid expert id, high 32 bits zero).  No int32 narrowing."""
    assert got.dtype == torch.int64, f"{context}: result dtype {got.dtype}, expected torch.int64"
    assert exp.dtype == torch.int64, f"{context}: reference dtype {exp.dtype}, expected torch.int64"
    assert got.shape == exp.shape, f"{context}: shape {tuple(got.shape)} vs {tuple(exp.shape)}"
    assert torch.equal(got, exp), f"{context}: got={got.tolist()} exp={exp.tolist()}"
    assert bool((got >= 0).all()) and bool((got < num_experts).all()), f"{context}: invalid index {got.tolist()}"
    assert bool(((got >> 32) == 0).all()), f"{context}: nonzero high half {got.tolist()}"


# (E, L) x K grid required by the int64 ABI acceptance matrix.
MATRIX_E_L = [(256, 128), (256, 64), (161, 64), (70, 128)]
MATRIX_K = [1, 6, 8]
_TIE_POSITIONS = (7, 33, 60, 69)


@pytest.mark.parametrize("num_topk", MATRIX_K, ids=[f"k{k}" for k in MATRIX_K])
@pytest.mark.parametrize(("num_experts", "lanes"), MATRIX_E_L, ids=[f"e{e}-l{l}" for e, l in MATRIX_E_L])
def test_parallel_topk_frontend_pto_numeric(num_experts, lanes, num_topk):
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    torch.manual_seed(0)
    scores = torch.randn(num_experts, dtype=torch.float32)
    got = _run_scores(scores, num_topk, lanes)
    exp = ref_topk_indices(scores, num_topk)
    _assert_index_values(got, exp, num_experts, f"random E={num_experts} L={lanes} K={num_topk}")


@pytest.mark.parametrize("num_topk", MATRIX_K, ids=[f"k{k}" for k in MATRIX_K])
@pytest.mark.parametrize(("num_experts", "lanes"), MATRIX_E_L, ids=[f"e{e}-l{l}" for e, l in MATRIX_E_L])
def test_parallel_topk_frontend_tie_break_smaller_index_wins(num_experts, lanes, num_topk):
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    scores = torch.full((num_experts,), -5.0, dtype=torch.float32)
    for idx in _TIE_POSITIONS:
        assert idx < num_experts, f"tie position {idx} outside E={num_experts}"
        scores[idx] = 3.0
    got = _run_scores(scores, num_topk, lanes)
    # Four-way tie at the peak: smallest index wins first, then the -5.0
    # tail is consumed in index order.
    exp = torch.tensor([*_TIE_POSITIONS, 0, 1, 2, 3], dtype=torch.int64)[:num_topk]
    _assert_index_values(got, exp, num_experts, f"tie E={num_experts} L={lanes} K={num_topk}")


@pytest.mark.parametrize("num_topk", MATRIX_K, ids=[f"k{k}" for k in MATRIX_K])
@pytest.mark.parametrize(("num_experts", "lanes"), MATRIX_E_L, ids=[f"e{e}-l{l}" for e, l in MATRIX_E_L])
def test_parallel_topk_frontend_valid_scores_all_neg_inf(num_experts, lanes, num_topk):
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    # PR262 replaces each selected score with -inf. On all--inf inputs the
    # smallest valid index therefore wins again; distinct indices are not
    # promised for this boundary. Check the literal source result.
    scores = torch.full((num_experts,), float("-inf"), dtype=torch.float32)
    got = _run_scores(scores, num_topk, lanes)
    exp = torch.zeros(num_topk, dtype=torch.int64)
    _assert_index_values(got, exp, num_experts, f"all--inf E={num_experts} L={lanes} K={num_topk}")


def test_parallel_topk_frontend_nondivisible_valid_scores_all_neg_inf():
    if not _npu_available():
        pytest.skip("NPU runtime unavailable")

    # Historical standalone case (E=161 with L=64 is also a matrix cell): the
    # padded scores[161:192] are -inf as well; with all valid scores -inf the
    # i < E guard is what keeps the padding out. Kept under its own nodeid.
    scores = torch.full((161,), float("-inf"), dtype=torch.float32)
    got = _run_scores(scores, K_DEFAULT, 64)
    exp = torch.zeros(K_DEFAULT, dtype=torch.int64)
    _assert_index_values(got, exp, 161, "nondivisible all--inf E=161 L=64")


if __name__ == "__main__":
    tilelang.testing.main()
