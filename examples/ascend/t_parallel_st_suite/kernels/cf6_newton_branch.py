#!/usr/bin/env python3
"""ST-CF6 — Newton recip with near-0 branch + N_FAST=3 iters.

if abs(x)<eps → y=0; else seed + N_FAST iterations y = y*(2 - x*y).
Prefer T.max(v,-v). Tags: cf6_e{E}_t{T}_pnear{P}. Primary: E=256, T=32, pnear∈{5,25}.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import compile_target, vf_region  # noqa: E402

EPS = 1e-4
N_FAST = 3


def build_cf6(E: int, threads: int, eps: float = EPS, n_fast: int = N_FAST):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        X: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            with vf_region(threads):
                x = T.alloc_fragment((E,), "float32")
                y = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    x[i] = x_ub[i]
                for i in T.Parallel(E):
                    ax = T.max(x[i], -x[i])
                    if ax < eps:
                        y[i] = T.float32(0.0)
                    else:
                        # seed recip ~ 1/x via rough start then Newton
                        y[i] = T.float32(1.0) / x[i]
                        for _t in T.serial(n_fast):
                            y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
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
    pnear = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tag = f"cf6_e{E}_t{threads}_pnear{pnear}"
    so = compile_prim(build_cf6(E, threads), tag, target=compile_target())
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
