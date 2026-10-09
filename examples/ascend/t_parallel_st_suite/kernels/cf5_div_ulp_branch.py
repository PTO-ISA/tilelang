#!/usr/bin/env python3
"""ST-CF5 — div with near-0 range branch (ULP intent documented).

Main path: y = a/b. Near-0: if abs(a) < eps → y = 0.
eps fixed at 1e-4 in kernel; pnear is host-data skew (tag).
Gold checks rtol=2e-5 atol=1e-6 (not strict 0.5 ULP — camodel).
Prefer T.max(v,-v) not T.abs.
Tags: cf5_e{E}_t{T}_pnear{P}. Primary: E=256, T=32, pnear∈{5,25}.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402

EPS = 1e-4


def build_cf5(E: int, threads: int, eps: float = EPS):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with vf_region(threads):
                a = T.alloc_fragment((E,), "float32")
                b = T.alloc_fragment((E,), "float32")
                y = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    a[i] = a_ub[i]
                    b[i] = b_ub[i]
                for i in T.Parallel(E):
                    ax = T.max(a[i], -a[i])
                    if ax < eps:
                        y[i] = T.float32(0.0)
                    else:
                        y[i] = a[i] / b[i]
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
    tag = f"cf5_e{E}_t{threads}_pnear{pnear}"
    so = compile_prim(build_cf5(E, threads), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
