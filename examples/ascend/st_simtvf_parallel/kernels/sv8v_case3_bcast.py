#!/usr/bin/env python3
"""ST-SV8V — Case-3 end-to-end reduce + bcast mul (SimdVF / VMI twin).

Same math/IO as Simt SV8:
  Out[R,C] fp16 = X * sf_inv[i, j//G]
  sf_inv[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)

Arms:
- ``live``: keep compact ``sf_inv[R,CG]`` in shared; consumer
  ``Parallel(R,CG)+serial(G)`` (avoid ``j//G`` InverseAffine).
- ``spill_dist``: expand each group scalar into G lanes in
  ``scale_ub[R,C]`` (layout-transform spill), then consumer reloads
  ``scale_ub[i,j]``.

Tags: ``sv8v_r{R}_c{C}_g{G}_t{L}_live`` / ``…_spill_dist``.
Defaults: R=64, C=128, G=16, lanes=64.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv8v(R: int, C: int, G: int, lanes: int, arm: str):
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
                scale_ub = T.alloc_shared((R, C), "float32")
                T.copy(X, x_ub)
                with T.SimdVF(lanes=lanes):
                    # Twin of Simt fragment sf_inv; scale_ub stays Kernel-scope (like Simt)
                    sf_inv = T.alloc_shared((R, CG), "float32")
                    for i, g in T.Parallel(R, CG):
                        m = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = T.Cast("float32", x_ub[i, g * G + t])
                            m = T.max(m, T.max(vv, -vv))
                        m = T.max(m, T.float32(1e-6))
                        sf_inv[i, g] = T.float32(1.0) / m
                    # layout-transform spill: expand [R,CG] → [R,C]
                    for i, g in T.Parallel(R, CG):
                        s = T.alloc_var("float32")
                        s = sf_inv[i, g]
                        for t in T.serial(G):
                            scale_ub[i, g * G + t] = s
                    for i, j in T.Parallel(R, C):
                        s = T.alloc_var("float32")
                        s = scale_ub[i, j]
                        out_ub[i, j] = T.Cast(
                            "float16", T.Cast("float32", x_ub[i, j]) * s
                        )
                T.copy(out_ub, Out)

        return main

    # arm == "live"
    @T.prim_func
    def main(
        X: T.Tensor((R, C), "float16"),
        Out: T.Tensor((R, C), "float16"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float16")
            out_ub = T.alloc_shared((R, C), "float16")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                # Twin of Simt fragment: shared working set under SimdVF
                sf_inv = T.alloc_shared((R, CG), "float32")
                for i, g in T.Parallel(R, CG):
                    m = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        m = T.max(m, T.max(vv, -vv))
                    m = T.max(m, T.float32(1e-6))
                    sf_inv[i, g] = T.float32(1.0) / m
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
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    lanes = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    arm = sys.argv[5] if len(sys.argv) > 5 else "live"
    if arm not in ("live", "spill_dist"):
        print(f"unknown arm {arm!r}; use live|spill_dist", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    tag = f"sv8v_r{R}_c{C}_g{G}_t{lanes}_{arm}"
    so = compile_prim(build_sv8v(R, C, G, lanes, arm), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
