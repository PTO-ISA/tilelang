#!/usr/bin/env python3
"""ST-CF1 — thresh kill KEEP (one Parallel if vs scalar thresh).

KEEP scores in fragment RF; each k: thr=thresh[k]; if scores[i] > thr: scores[i]=NEG.
Tags: cf1_e{E}_k{K}_t{T}_keep. Primary: E=256, K∈{1,8}, T=32.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import NEG, boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402


def build_cf1(E: int, K: int, threads: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        Thresh: T.Tensor((K,), "float32"),
        Out: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, s)
            T.copy(Thresh, thr_ub)
            with vf_region(threads):
                scores = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s[i]
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        if scores[i] > thr:
                            scores[i] = NEG
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    tag = f"cf1_e{E}_k{K}_t{threads}_keep"
    so = compile_prim(build_cf1(E, K, threads), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
