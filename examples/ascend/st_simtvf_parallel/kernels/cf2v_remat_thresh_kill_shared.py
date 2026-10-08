#!/usr/bin/env python3
"""ST-CF2V — remat + kill-in-shared (SimdVF / VMI twin of CF2).

Same if semantics as CF1; remat scores from shared each k; kill writes s[i]=NEG
via mask select. Mode 1 on scores (remat) + shared publish.
Tags: cf2v_e{E}_k{K}_t{L}_remat. Primary: E=256, K=8, lanes=64.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

NEG = float(-3.402823e38)


def build_cf2v(E: int, K: int, lanes: int):
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
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_shared((E,), "float32")
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        scores[i] = s[i]
                    for i in T.Parallel(E):
                        # mask select kill into shared home
                        s[i] = T.Select(scores[i] > thr, T.float32(NEG), s[i])
                for i in T.Parallel(E):
                    out_ub[i] = s[i]
            T.copy(out_ub, Out)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lanes = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    tag = f"cf2v_e{E}_k{K}_t{lanes}_remat"
    so = compile_prim(build_cf2v(E, K, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
