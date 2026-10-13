"""GPU-shaped SimdVF Parallel + reduce top-k frontend (PTO regression).

Ported from PR #262 head (`pr262-tree-verify-9008cc1b`,
commit 9008cc1b285d80fa4275b18c623b9e15a4575813,
``examples/ascend/example_simdvf_parallel_topk_frontend.py``,
sha256 a426c70e2a2cda43...): a re-export shim over the tournament body in
``example_simdvf_topk_gate_frontend`` so both names stay in sync.

The normative contract now lowers through ``Verify -> Legalize -> Vectorize``
(assert_lowered_contract captures the three passes); the tournament index
chain stays int32 while the GM ``IdxOut`` ABI is int64, produced by the
``tl.ExpandDtypeMismatchedCopy`` UB widening. See
example_simdvf_topk_gate_frontend for the full adaptation list.
"""

from __future__ import annotations

# Re-export the tournament body (same PrimFunc / compile / contract helpers).
from example_simdvf_topk_gate_frontend import (  # noqa: F401
    E,
    INT_MAX,
    K_DEFAULT,
    L,
    _stage_capture,
    align,
    assert_lowered_contract,
    compile_frontend,
    ref_topk_indices,
    source_markers,
    topk_gate_frontend as parallel_topk_frontend,
)

__all__ = [
    "E",
    "INT_MAX",
    "K_DEFAULT",
    "L",
    "align",
    "assert_lowered_contract",
    "compile_frontend",
    "parallel_topk_frontend",
    "ref_topk_indices",
    "source_markers",
]


if __name__ == "__main__":
    for e, k, lanes in ((E, K_DEFAULT, L), (256, 8, 64), (161, 8, 64)):
        with _stage_capture() as captured:
            compiled = compile_frontend(e, k, lanes)
        markers = assert_lowered_contract(compiled, captured)
        print("COMPILE_OK", f"E={e} K={k} lanes={lanes} aligned={align(e, lanes)}", markers)
    print("source_contract_ok")
