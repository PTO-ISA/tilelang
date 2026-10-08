#!/usr/bin/env python3
"""ST-CF4V — nested if via nested mask select (SimdVF / VMI twin of CF4).

Same 2-level classify: x>hi → hi; else x<lo → lo; else x*scale (fat).
Expressed as nested T.Select (lockstep pays fat path). pfat = host skew.
Tags: cf4v_e{E}_t{L}_pfat{P}. Primary: E=256, lanes=64, pfat∈{5,25}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402


def build_cf4v(E: int, lanes: int):
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
            with T.SimdVF(lanes=lanes):
                x = T.alloc_shared((E,), "float32")
                y = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    x[i] = a_ub[i]
                for i in T.Parallel(E):
                    hi = hi_ub[0]
                    lo = lo_ub[0]
                    # nested mask select ≡ nested if (no divergent schedule)
                    y[i] = T.Select(
                        x[i] > hi,
                        hi,
                        T.Select(x[i] < lo, lo, x[i] * sc_ub[i]),
                    )
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    pfat = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tag = f"cf4v_e{E}_t{lanes}_pfat{pfat}"
    so = compile_prim(build_cf4v(E, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
