#!/usr/bin/env python3
"""ST-SV8 — Case-3 end-to-end: reduce + reduced eltwise + bcast mul.

Input ``X[R,C]`` fp16; output ``Out[R,C]`` fp16 =
``x * sf_inv[i, j//G]`` where
``sf_inv[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)``.

Defaults: R=64, C=128, G=16, T=32.

Arms (separate ``@T.prim_func`` bodies):
- ``live``: keep ``sf_inv[R,CG]`` in fragment; consumer uses
  ``Parallel(R,CG)+serial(G)`` reading live frag (avoid ``j//G`` InverseAffine).
- ``spill_dist``: after computing ``sf_inv`` in frag, **layout-spill** to shared
  by expanding each group scalar into G lanes
  (``scale_ub[R,C]``; ``for i,g in Parallel(R,CG): for t in serial(G):
  scale_ub[i,g*G+t]=sf_inv[i,g]``), then consumer reloads ``scale_ub[i,j]``.
  This is a VL-friendly **layout-transform spill**, not a capacity spill.

Tags: ``sv8_r{R}_c{C}_g{G}_t{T}_live`` and ``…_spill_dist``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402


def build_sv8(R: int, C: int, G: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    CG = C // G

    if arm == "spill_dist":

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Out: T.Tensor((R, C), "float16"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                out_ub = T.alloc_shared((R, C), "float16")
                # layout-transform spill: expand reduced [R,CG] → full [R,C]
                scale_ub = T.alloc_shared((R, C), "float32")
                T.copy(X, x_ub)
                with vf_region(threads):
                    sf_inv = T.alloc_fragment((R, CG), "float32")
                    for i, g in T.Parallel(R, CG):
                        m = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = T.Cast("float32", x_ub[i, g * G + t])
                            m = T.max(m, T.max(vv, -vv))
                        m = T.max(m, T.float32(1e-6))
                        sf_inv[i, g] = T.float32(1.0) / m
                    # expand each group scalar into G lanes (layout rehome)
                    for i, g in T.Parallel(R, CG):
                        s = T.alloc_var("float32")
                        s = sf_inv[i, g]
                        for t in T.serial(G):
                            scale_ub[i, g * G + t] = s
                    # consumer reloads VL-friendly scale_ub[i,j]
                    for i, j in T.Parallel(R, C):
                        s = T.alloc_var("float32")
                        s = scale_ub[i, j]
                        out_ub[i, j] = T.Cast(
                            "float16", T.Cast("float32", x_ub[i, j]) * s
                        )
                T.copy(out_ub, Out)

        return main

    # arm == "live" (default): sf_inv stays in fragment RF for consumer
    @T.prim_func
    def main(
        X: T.Tensor((R, C), "float16"),
        Out: T.Tensor((R, C), "float16"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float16")
            out_ub = T.alloc_shared((R, C), "float16")
            T.copy(X, x_ub)
            with vf_region(threads):
                sf_inv = T.alloc_fragment((R, CG), "float32")
                for i, g in T.Parallel(R, CG):
                    m = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        m = T.max(m, T.max(vv, -vv))
                    m = T.max(m, T.float32(1e-6))
                    sf_inv[i, g] = T.float32(1.0) / m
                # Parallel(R,CG)+serial(G) — avoid j//G InverseAffine layout bugs
                for i, g in T.Parallel(R, CG):
                    s = T.alloc_var("float32")
                    s = sf_inv[i, g]
                    for t in T.serial(G):
                        j = g * G + t
                        out_ub[i, j] = T.Cast(
                            "float16", T.Cast("float32", x_ub[i, j]) * s
                        )
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "live"
    if arm not in ("live", "spill_dist"):
        print(f"unknown arm {arm!r}; use live|spill_dist", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    tag = f"sv8_r{R}_c{C}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sv8(R, C, G, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
