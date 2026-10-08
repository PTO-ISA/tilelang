#!/usr/bin/env python3
"""ST-SV2 — Topk loop-carried KEEP (EXTRACT from run_simd_vs_simt_perf.py SimtVF arm).

Matrix: E=256 × K∈{1,8} × threads∈{32,128}
target=ascend + cython; deps-native lib.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import NEG, INT_MAX, boot, compile_prim, set_out  # noqa: E402


def build_sv2(num_experts: int, num_topk: int, threads: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((num_experts,), "float32"),
        IdxOut: T.Tensor((num_topk,), "int32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((num_experts,), "float32")
            idx_ub = T.alloc_shared((num_topk,), "int32")
            T.copy(A, s)
            with T.SimtVF(threads=threads):
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

    out = os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel")
    set_out(out)
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    tag = f"sv2_e{E}_k{K}_t{threads}"
    prim = build_sv2(E, K, threads)
    so = compile_prim(prim, tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
