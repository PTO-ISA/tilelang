#!/usr/bin/env python3
"""ST-CF3 — remat_idx only: kill by rematted Parallel index i (no KEEP idxs).

Load scores KEEP; each k: v=victim[k]; if i == v: scores[i]=NEG.
Tags: cf3_e{E}_k{K}_t{T}_remat_idx. Primary: E=256, K=8, T=32.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import NEG, boot, compile_prim, set_out  # noqa: E402


def build_cf3(E: int, K: int, threads: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        Victim: T.Tensor((K,), "int32"),
        Out: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            vic_ub = T.alloc_shared((K,), "int32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, s)
            T.copy(Victim, vic_ub)
            with T.SimtVF(threads=threads):
                scores = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s[i]
                for k in T.serial(K):
                    v = vic_ub[k]
                    for i in T.Parallel(E):
                        if i == v:  # remat index at use — no idxs KEEP buffer
                            scores[i] = NEG
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    tag = f"cf3_e{E}_k{K}_t{threads}_remat_idx"
    so = compile_prim(build_cf3(E, K, threads), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
