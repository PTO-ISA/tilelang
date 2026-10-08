#!/usr/bin/env python3
"""LEGACY ST-SV5 — Bcast + multi-consumer (pre Case-3).

**LEGACY / retained for reference.** Active Case-3 SV5 is
``kernels/sv5_reduce_small_eltwise.py`` (tags ``sv5_r*_c*_g*_t*``, reduced Y[R,CG] only).

This file keeps the old vector bcast multi-consumer arms (tags ``sv5_e*_t*_{frag,reload}``).
Arm A: fragment amax bcast. Arm B: shared reload.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv5(E: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    if arm == "frag":

        @T.prim_func
        def main(
            X: T.Tensor((E,), "float32"),
            Y: T.Tensor((E,), "float32"),
            Z: T.Tensor((E,), "float32"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((E,), "float32")
                y_ub = T.alloc_shared((E,), "float32")
                z_ub = T.alloc_shared((E,), "float32")
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    x = T.alloc_fragment((E,), "float32")
                    amax = T.alloc_fragment((1,), "float32")
                    y = T.alloc_fragment((E,), "float32")
                    z = T.alloc_fragment((E,), "float32")
                    for i in T.Parallel(E):
                        x[i] = x_ub[i]
                    T.reduce_max(x, amax)
                    for i in T.Parallel(E):
                        y[i] = x[i] * (1.0 / amax[0])
                    for i in T.Parallel(E):
                        if x[i] == amax[0]:
                            z[i] = 1.0
                        else:
                            z[i] = 0.0
                    for i in T.Parallel(E):
                        y_ub[i] = y[i]
                        z_ub[i] = z[i]
                T.copy(y_ub, Y)
                T.copy(z_ub, Z)

        return main

    # arm == "reload": publish amax to shared; each consumer reloads
    @T.prim_func
    def main(
        X: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
        Z: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            z_ub = T.alloc_shared((E,), "float32")
            amax_s = T.alloc_shared((1,), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                x = T.alloc_fragment((E,), "float32")
                amax = T.alloc_fragment((1,), "float32")
                y = T.alloc_fragment((E,), "float32")
                z = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    x[i] = x_ub[i]
                T.reduce_max(x, amax)
                amax_s[0] = amax[0]
                for i in T.Parallel(E):
                    scale = amax_s[0]
                    y[i] = x[i] * (1.0 / scale)
                for i in T.Parallel(E):
                    peak = amax_s[0]
                    if x[i] == peak:
                        z[i] = 1.0
                    else:
                        z[i] = 0.0
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
                    z_ub[i] = z[i]
            T.copy(y_ub, Y)
            T.copy(z_ub, Z)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    arm = sys.argv[3] if len(sys.argv) > 3 else "frag"
    tag = f"sv5_e{E}_t{threads}_{arm}"
    so = compile_prim(build_sv5(E, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
