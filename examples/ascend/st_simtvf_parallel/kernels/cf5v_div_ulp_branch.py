#!/usr/bin/env python3
"""ST-CF5V — div + near-0 branch via mask select (SimdVF / VMI twin of CF5).

Main path y=a/b; near-0 → y=0. VerifyParallelToPTO rejects Div/Max inside
Parallel; serial(Div) inside SimdVF fails AIV ("Unsupported scalar instruction").
ABI adaptation: remat quot = a/b in Kernel-scope serial *outside* SimdVF;
inside VF, Parallel only does abs-via-Select + near-0 mask Select on quot.
Tags: cf5v_e{E}_t{L}_pnear{P}. Primary: E=256, lanes=64, pnear∈{5,25}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

EPS = 1e-4


def build_cf5v(E: int, lanes: int, eps: float = EPS):
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
            quot = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            # ABI: Div outside SimdVF (not Parallel-whitelist; not AIV-scalar-in-VF)
            for i in T.serial(E):
                quot[i] = a_ub[i] / b_ub[i]
            with T.SimdVF(lanes=lanes):
                a = T.alloc_shared((E,), "float32")
                y = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    a[i] = a_ub[i]
                for i in T.Parallel(E):
                    ax = T.Select(a[i] >= T.float32(0.0), a[i], a[i] * T.float32(-1.0))
                    y[i] = T.Select(ax < eps, T.float32(0.0), quot[i])
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    pnear = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tag = f"cf5v_e{E}_t{lanes}_pnear{pnear}"
    so = compile_prim(build_cf5v(E, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
