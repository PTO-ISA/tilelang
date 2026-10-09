#!/usr/bin/env python3
"""ST-SV9 — Topk e2e (KEEP / remat_scores / remat_idx).

Extracted from ``kernels/_removed_legacy/sv2_topk_keep.py`` (+ SV3/SV4 remat forks).
Replaces the three legacy topk micros with one e2e ST and three arms.

Math (stable min-index-on-ties top-K):
  for k in serial(K):
    amax = reduce_max(scores)
    idx_cand[i] = idxs[i] if scores[i]==amax else INT_MAX
    best = reduce_min(idx_cand)
    emit IdxOut[k] = best
    kill winner (NEG) per arm semantics

Arms (separate ``@T.prim_func`` bodies):
- ``keep``         — scores+idxs KEEP in fragment RF; kill in RF
- ``remat_scores`` — reload scores from shared each K; kill in shared
- ``remat_idx``    — KEEP scores; no idxs fragment; remat ``i`` at use

Tags: ``sv9_e{E}_k{K}_t{T}_{keep,remat_scores,remat_idx}``.
Primary matrix: E=256, K∈{1,8}, T=32 (+ optional T=128 keep).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import NEG, INT_MAX, boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402


def build_sv9(num_experts: int, num_topk: int, threads: int, arm: str):
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
                with vf_region(threads):
                    scores = T.alloc_fragment((num_experts,), "float32")
                    idxs = T.alloc_fragment((num_experts,), "int32")
                    idx_cand = T.alloc_fragment((num_experts,), "int32")
                    amax = T.alloc_fragment((1,), "float32")
                    best = T.alloc_fragment((1,), "int32")
                    for i in T.Parallel(num_experts):
                        idxs[i] = i
                    for k in T.serial(num_topk):
                        # remat via reload: materialize working set from shared each K
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
                        # kill in shared home (not KEEP in fragment across K)
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
                with vf_region(threads):
                    scores = T.alloc_fragment((num_experts,), "float32")
                    # NO idxs KeepLive fragment — remat index at use
                    idx_cand = T.alloc_fragment((num_experts,), "int32")
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

    # arm == "keep" (default): scores+idxs KEEP; kill in RF
    @T.prim_func
    def main(
        A: T.Tensor((num_experts,), "float32"),
        IdxOut: T.Tensor((num_topk,), "int32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((num_experts,), "float32")
            idx_ub = T.alloc_shared((num_topk,), "int32")
            T.copy(A, s)
            with vf_region(threads):
                scores = T.alloc_fragment((num_experts,), "float32")
                idxs = T.alloc_fragment((num_experts,), "int32")
                idx_cand = T.alloc_fragment((num_experts,), "int32")
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
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    arm = sys.argv[4] if len(sys.argv) > 4 else "keep"
    if arm not in ("keep", "remat_scores", "remat_idx"):
        print(f"unknown arm {arm!r}; use keep|remat_scores|remat_idx", flush=True)
        return 2
    tag = f"sv9_e{E}_k{K}_t{threads}_{arm}"
    so = compile_prim(build_sv9(E, K, threads, arm), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
