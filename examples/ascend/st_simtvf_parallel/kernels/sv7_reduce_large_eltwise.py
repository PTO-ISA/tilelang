#!/usr/bin/env python3
"""ST-SV7 — Case-3 large-G reduce → reduced eltwise (RF-capacity rung 3).

Input ``X[R,C]`` fp16; output ``Y[R,CG]`` fp32 with ``CG=C/G``:
``Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)``.

No bcast. Primary: R=64, C=128, G=128 → CG=1 (also G=64 → CG=2), threads=32.
Tag: ``sv7_r{R}_c{C}_g{G}_t{T}_{multiwarp_ub|reload}``.

Sensitivity (2026-10-06): **group-input capacity during absmax**. G=128 is the
rung where the group needs ≈4 warps (NW=4 chunks of 32) and therefore a **UB
round-trip for the allreduce of the per-chunk partials**.

- ``multiwarp_ub`` : G is chunked into ``NW=4`` chunks of 32. Each chunk is
  reduced with ``Parallel(R,CG)`` into ``part_ub[R,CG,NW]`` (shared UB
  partials); a final ``Parallel(R,CG)`` takes the max over the NW partials and
  writes ``Y``. This is the "must write UB for the allreduce" arm. It runs with
  ``threads=32`` (the chunk/warp loop is serial inside the Parallel body), so
  the UB partial traffic — not the thread count — is the knob.
- ``reload``       : baseline single ``serial(G)`` stream out of shared
  ``x_ub``, no partial buffer. Same arithmetic, same gold.

Programming notes (green SV2/SV5):
- ``T.alloc_var`` + reassignment for mutable absmax
- abs via ``T.max(v, -v)`` (not ``T.abs``)
- partial index ``[i, g, w]`` stays affine in the Parallel vars
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

ARMS = ("multiwarp_ub", "reload")
NW = 4  # chunks of 32 → "warps" worth of group width


def build_sv7(R: int, C: int, G: int, threads: int, arm: str = "multiwarp_ub"):
    import tilelang.ascend.language as T

    CG = C // G

    if arm == "multiwarp_ub":
        CH = G // NW  # chunk width (32 for G=128, 16 for G=64)

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Y: T.Tensor((R, CG), "float32"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                y_ub = T.alloc_shared((R, CG), "float32")
                part_ub = T.alloc_shared((R, CG, NW), "float32")
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    # per-chunk ("per-warp") partials → UB
                    for i, g in T.Parallel(R, CG):
                        for w in T.serial(NW):
                            m = T.alloc_var("float32", init=0.0)
                            for t in T.serial(CH):
                                vv = T.Cast("float32", x_ub[i, g * G + w * CH + t])
                                m = T.max(m, T.max(vv, -vv))
                            part_ub[i, g, w] = m
                    # allreduce of the partials through UB → Y
                    for i, g in T.Parallel(R, CG):
                        mm = T.alloc_var("float32", init=0.0)
                        for w in T.serial(NW):
                            mm = T.max(mm, part_ub[i, g, w])
                        mm = T.max(mm, T.float32(1e-6))
                        y_ub[i, g] = T.float32(1.0) / mm
                T.copy(y_ub, Y)

        return main

    # arm == "reload": baseline single serial(G) stream
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

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "multiwarp_ub"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    if arm == "multiwarp_ub" and G % NW != 0:
        print(f"G={G} must be divisible by NW={NW}", flush=True)
        return 2
    tag = f"sv7_r{R}_c{C}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sv7(R, C, G, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
