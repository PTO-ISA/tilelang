#!/usr/bin/env python3
"""ST-SV9V — Topk e2e keep / remat_scores / remat_idx (SimdVF / VMI twin of SV9).

**Twin fidelity rule:** MUST mirror Simt SV9 control flow. SV9 has **no**
token-loop unroll → SV9V must **not** invent AABBCC / ABCABC / ``token_tile``
/ ``vf_fuse``. Those knobs live in separate CF STs (CF4/CF5) that require a
Simt (or shared) ST that actually has them — see ``ST_CF_TOPK_GATE_DESIGN.md``
and ``ST_VMI_TWINS.md``.

Same math/IO as Simt SV9 (stable min-index-on-ties top-K):
  for k in serial(K):
    amax = reduce_max(scores)
    idx_cand[i] = idxs[i] if scores[i]==amax else INT_MAX
    best = reduce_min(idx_cand)
    emit IdxOut[k] = best
    kill winner (NEG) per arm

Arms (separate ``@T.prim_func`` bodies) — same three as SV9:
- ``keep``         — scores+idxs KEEP (mode-3 stay-alive); kill NEG in working set
- ``remat_scores`` — reload scores from shared each K; kill in shared home
- ``remat_idx``    — KEEP scores; no idxs buffer; remat ``i`` at use

Tags: ``sv9v_e{E}_k{K}_t{L}_{keep,remat_scores,remat_idx}``.
Primary: E=256, K∈{1,8}, lanes=64.

VMI ABI only: ``SimdVF(lanes=)`` + prefer ``alloc_shared`` for VF working set;
1-elem reduce scratch may stay fragment (``T.reduce_max`` API). Overlay path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

NEG = np.float32(-3.402823e38).item()
INT_MAX = np.iinfo(np.int32).max


def build_sv9v(num_experts: int, num_topk: int, lanes: int, arm: str):
    import tilelang.ascend.language as T

    if arm == "remat_scores":

        @T.prim_func
        def main(
            A: T.Tensor((num_experts,), "float32"),
            IdxOut: T.Tensor((num_topk,), "int32"),
        ):
            with T.Kernel(1):
                s = T.alloc_shared((num_experts,), "float32")
                idx_ub = T.alloc_shared((num_topk,), "int32")
                T.copy(A, s)
                with T.SimdVF(lanes=lanes):
                    scores = T.alloc_shared((num_experts,), "float32")
                    idxs = T.alloc_shared((num_experts,), "int32")
                    idx_cand = T.alloc_shared((num_experts,), "int32")
                    # 1-elem reduce scratch: fragment (reduce API)
                    amax = T.alloc_fragment((1,), "float32")
                    best = T.alloc_fragment((1,), "int32")
                    for i in T.Parallel(num_experts):
                        idxs[i] = i
                    for k in T.serial(num_topk):
                        for i in T.Parallel(num_experts):
                            scores[i] = s[i]
                        T.reduce_max(scores, amax)
                        for i in T.Parallel(num_experts):
                            if scores[i] == amax[0]:
                                idx_cand[i] = idxs[i]
                            else:
                                idx_cand[i] = INT_MAX
                        T.reduce_min(idx_cand, best)
                        idx_ub[k] = best[0]
                        for i in T.Parallel(num_experts):
                            if idxs[i] == best[0]:
                                s[i] = NEG
                T.copy(idx_ub, IdxOut)

        return main

    if arm == "remat_idx":

        @T.prim_func
        def main(
            A: T.Tensor((num_experts,), "float32"),
            IdxOut: T.Tensor((num_topk,), "int32"),
        ):
            with T.Kernel(1):
                s = T.alloc_shared((num_experts,), "float32")
                idx_ub = T.alloc_shared((num_topk,), "int32")
                T.copy(A, s)
                with T.SimdVF(lanes=lanes):
                    scores = T.alloc_shared((num_experts,), "float32")
                    # NO idxs KeepLive buffer — remat index at use
                    idx_cand = T.alloc_shared((num_experts,), "int32")
                    amax = T.alloc_fragment((1,), "float32")
                    best = T.alloc_fragment((1,), "int32")
                    for i in T.Parallel(num_experts):
                        scores[i] = s[i]
                    for k in T.serial(num_topk):
                        T.reduce_max(scores, amax)
                        for i in T.Parallel(num_experts):
                            idx_i = i  # remat index (vci outer analogue)
                            if scores[i] == amax[0]:
                                idx_cand[i] = idx_i
                            else:
                                idx_cand[i] = INT_MAX
                        T.reduce_min(idx_cand, best)
                        idx_ub[k] = best[0]
                        for i in T.Parallel(num_experts):
                            if i == best[0]:
                                scores[i] = NEG
                T.copy(idx_ub, IdxOut)

        return main

    # arm == "keep"
    @T.prim_func
    def main(
        A: T.Tensor((num_experts,), "float32"),
        IdxOut: T.Tensor((num_topk,), "int32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((num_experts,), "float32")
            idx_ub = T.alloc_shared((num_topk,), "int32")
            T.copy(A, s)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_shared((num_experts,), "float32")
                idxs = T.alloc_shared((num_experts,), "int32")
                idx_cand = T.alloc_shared((num_experts,), "int32")
                amax = T.alloc_fragment((1,), "float32")
                best = T.alloc_fragment((1,), "int32")
                for i in T.Parallel(num_experts):
                    scores[i] = s[i]
                    idxs[i] = i
                for k in T.serial(num_topk):
                    T.reduce_max(scores, amax)
                    for i in T.Parallel(num_experts):
                        if scores[i] == amax[0]:
                            idx_cand[i] = idxs[i]
                        else:
                            idx_cand[i] = INT_MAX
                    T.reduce_min(idx_cand, best)
                    idx_ub[k] = best[0]
                    for i in T.Parallel(num_experts):
                        if idxs[i] == best[0]:
                            scores[i] = NEG
            T.copy(idx_ub, IdxOut)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lanes = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    arm = sys.argv[4] if len(sys.argv) > 4 else "keep"
    if arm not in ("keep", "remat_scores", "remat_idx"):
        print(f"unknown arm {arm!r}; use keep|remat_scores|remat_idx", flush=True)
        return 2
    tag = f"sv9v_e{E}_k{K}_t{lanes}_{arm}"
    so = compile_prim(build_sv9v(E, K, lanes, arm), tag, target="pto")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
