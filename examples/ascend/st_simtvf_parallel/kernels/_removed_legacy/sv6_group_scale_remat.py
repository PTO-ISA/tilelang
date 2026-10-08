#!/usr/bin/env python3
"""LEGACY ST-SV6 — Reduce→scale→mul group remat (pre Case-3).

**LEGACY / retained for reference.** Active Case-3 SV6 is
``kernels/sv6_reduce_large_eltwise.py`` (tags ``sv6_r*_c*_g*_t*`` with large G,
reduced Y[R,CG] only — no bcast, no remat/reload arm suffix).

Old tags were ``sv6_r*_c*_g*_t*_{remat,reload}`` with full Out[R,C] fp16.
R=64 C=128 G=16 → amax[64,8].

Programming-style notes:
- alloc_var + name reassignment for mutable group absmax
- reduce/scale over Parallel(R, CG) + serial(G) reading shared (avoid fragment
  g*G+t / j//G layout-infer InverseAffine scale!=1)
- remat vs reload selected as two Python @T.prim_func bodies
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv6(R: int, C: int, G: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    CG = C // G

    if arm == "reload":

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Out: T.Tensor((R, C), "float16"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                out_ub = T.alloc_shared((R, C), "float16")
                amax_s = T.alloc_shared((R, CG), "float32")
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    amax = T.alloc_fragment((R, CG), "float32")
                    for i, g in T.Parallel(R, CG):
                        m = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = T.Cast("float32", x_ub[i, g * G + t])
                            m = T.max(m, T.max(vv, -vv))
                        amax[i, g] = m
                        amax_s[i, g] = m
                    for i, g in T.Parallel(R, CG):
                        scale = T.alloc_var("float32")
                        scale = amax_s[i, g]
                        scale = T.max(scale, T.float32(1e-6))
                        for t in T.serial(G):
                            j = g * G + t
                            out_ub[i, j] = T.Cast(
                                "float16", T.Cast("float32", x_ub[i, j]) / scale
                            )
                T.copy(out_ub, Out)

        return main

    @T.prim_func
    def main(
        X: T.Tensor((R, C), "float16"),
        Out: T.Tensor((R, C), "float16"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float16")
            out_ub = T.alloc_shared((R, C), "float16")
            amax_s = T.alloc_shared((R, CG), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                amax = T.alloc_fragment((R, CG), "float32")
                for i, g in T.Parallel(R, CG):
                    m = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        m = T.max(m, T.max(vv, -vv))
                    amax[i, g] = m
                for i, g in T.Parallel(R, CG):
                    scale = T.alloc_var("float32")
                    scale = amax[i, g]
                    scale = T.max(scale, T.float32(1e-6))
                    for t in T.serial(G):
                        j = g * G + t
                        out_ub[i, j] = T.Cast(
                            "float16", T.Cast("float32", x_ub[i, j]) / scale
                        )
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "remat"
    tag = f"sv6_r{R}_c{C}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sv6(R, C, G, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
