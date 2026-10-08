#!/usr/bin/env python3
"""ST-SV6 — Case-3 mid-G reduce → reduced eltwise (RF-capacity rung 2).

Input ``X[R,C]`` fp16; output ``Y[R,CG]`` fp32 with ``CG=C/G``:
``Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)``.

Stops at the reduced grid — **no bcast back to full C**.
Defaults: R=64, C=128, G=32 → CG=4, threads=32.
Tag: ``sv6_r{R}_c{C}_g{G}_t{T}_{keep_in_warp|reload}``.

Sensitivity (2026-10-06): **group-input RF capacity during absmax**.
G=32 is the mid rung — the group no longer fits ≈32 SIMD arch VL regs (so the
SIMD/VMI twin must reload), but SIMT with ``T=32=G`` can still hold the group
inside **one warp** and finish the reduce without ever writing a warp partial
to UB for a multi-warp merge.

- ``keep_in_warp`` : the group inputs are preloaded into a fragment
  (``xf[R,CG,G]`` fp32, RF/warp-local) and the absmax runs **from the
  fragment**; no partial ever lands in UB — only the final ``sf_inv`` is
  written to ``y_ub``.
- ``reload``       : baseline body — ``Parallel(R,CG) + serial(G)`` streaming
  the group out of shared ``x_ub`` every step. Same arithmetic, same gold.

Programming notes (green SV2/SV5):
- ``T.alloc_var`` + reassignment for mutable absmax
- abs via ``T.max(v, -v)`` (not ``T.abs``)
- 3-D ``[R,CG,G]`` fragment keeps every index affine (no ``j//G``)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

ARMS = ("keep_in_warp", "reload")


def build_sv6(R: int, C: int, G: int, threads: int, arm: str = "keep_in_warp"):
    import tilelang.ascend.language as T

    CG = C // G

    if arm == "keep_in_warp":

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Y: T.Tensor((R, CG), "float32"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                y_ub = T.alloc_shared((R, CG), "float32")
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    xf = T.alloc_fragment((R, CG, G), "float32")
                    # preload group into warp-local RF (separate Parallel loop:
                    # one ParallelOp records only ONE index pattern per buffer)
                    for i, g in T.Parallel(R, CG):
                        for t in T.serial(G):
                            xf[i, g, t] = T.Cast("float32", x_ub[i, g * G + t])
                    # warp-local reduce straight out of RF — no UB partials
                    for i, g in T.Parallel(R, CG):
                        m = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = xf[i, g, t]
                            m = T.max(m, T.max(vv, -vv))
                        m = T.max(m, T.float32(1e-6))
                        # final sf_inv only — no warp-partial scratch in UB
                        y_ub[i, g] = T.float32(1.0) / m
                T.copy(y_ub, Y)

        return main

    # arm == "reload": baseline stream from shared UB
    @T.prim_func
    def main(
        X: T.Tensor((R, C), "float16"),
        Y: T.Tensor((R, CG), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float16")
            y_ub = T.alloc_shared((R, CG), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                for i, g in T.Parallel(R, CG):
                    m = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        m = T.max(m, T.max(vv, -vv))
                    m = T.max(m, T.float32(1e-6))
                    y_ub[i, g] = T.float32(1.0) / m
            T.copy(y_ub, Y)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "keep_in_warp"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    tag = f"sv6_r{R}_c{C}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sv6(R, C, G, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
