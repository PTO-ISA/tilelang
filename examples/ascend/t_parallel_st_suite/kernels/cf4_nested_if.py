#!/usr/bin/env python3
"""ST-CF4 — nested if (2-level) divergence stress.

if x>hi → hi; else if x<lo → lo; else x*scale (fat path).
pfat is host-data skew for opsim gold (encoded in tag), not a kernel param.
Tags: cf4_e{E}_t{T}_pfat{P}. Primary: E=256, T=32, pfat∈{5,25}.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_cf4(E: int, threads: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        Scale: T.Tensor((E,), "float32"),
        Lo: T.Tensor((1,), "float32"),
        Hi: T.Tensor((1,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            sc_ub = T.alloc_shared((E,), "float32")
            lo_ub = T.alloc_shared((1,), "float32")
            hi_ub = T.alloc_shared((1,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(Scale, sc_ub)
            T.copy(Lo, lo_ub)
            T.copy(Hi, hi_ub)
            with T.SimtVF(threads=threads):
                x = T.alloc_fragment((E,), "float32")
                y = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    x[i] = a_ub[i]
                for i in T.Parallel(E):
                    hi = hi_ub[0]
                    lo = lo_ub[0]
                    if x[i] > hi:
                        y[i] = hi
                    else:
                        if x[i] < lo:
                            y[i] = lo
                        else:
                            y[i] = x[i] * sc_ub[i]  # fat path
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    pfat = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tag = f"cf4_e{E}_t{threads}_pfat{pfat}"
    so = compile_prim(build_cf4(E, threads), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
