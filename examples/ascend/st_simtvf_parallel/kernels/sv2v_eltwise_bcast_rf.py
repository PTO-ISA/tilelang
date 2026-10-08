#!/usr/bin/env python3
"""ST-SV2V — Fold scale KEEP; input KEEP vs STREAM (SimdVF twin of SV2 A).

Same fold gold as Simt SV2 / ptodsl SV2d. Scale always KEEP.

ABI note: SimdVF prefers ``alloc_shared`` for VF working (cannot claim fragment
RF KEEP). Structure mirrors A arms; scale lives in shared working buffer that
the consumer reuses (not rematted). Documented asymmetry vs A fragment KEEP.

Arms:
- ``input_keep``: KEEP x + scale in shared working (R≈16)
- ``input_stream`` (aliases frag_live/fold_frag_keep): STREAM x from x_ub;
  scale KEEP in shared working (R≥32)
- ``reload``: DEMOTEd scale-remat foil

Defaults: R=32 C=64 lanes=64.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

EPS = 1e-6
_STREAM_ARMS = ("input_stream", "frag_live", "fold_frag_keep")
_ARMS = ("input_keep",) + _STREAM_ARMS + ("reload",)


def build_sv2v(R: int, C: int, lanes: int, arm: str):
    import tilelang.ascend.language as T

    LANES = lanes
    if C % LANES != 0:
        raise ValueError(f"C={C} must be divisible by LANES=lanes={LANES}")
    NCH = C // LANES

    if arm == "reload":

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Out: T.Tensor((R, C), "float16"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                out_ub = T.alloc_shared((R, C), "float16")
                scale_s = T.alloc_shared((R, LANES), "float32")
                T.copy(X, x_ub)
                with T.SimdVF(lanes=lanes):
                    scale = T.alloc_shared((R, LANES), "float32")
                    for i, lane in T.Parallel(R, LANES):
                        m = T.alloc_var("float32", init=EPS)
                        for ch in T.serial(NCH):
                            vv = T.Cast("float32", x_ub[i, ch * LANES + lane])
                            m = T.max(m, T.max(vv, -vv))
                        scale[i, lane] = T.max(m, T.float32(EPS))
                        scale_s[i, lane] = scale[i, lane]
                    for i, lane in T.Parallel(R, LANES):
                        s = T.alloc_var("float32")
                        s = scale_s[i, lane]
                        for ch in T.serial(NCH):
                            j = ch * LANES + lane
                            out_ub[i, j] = T.Cast(
                                "float16", T.Cast("float32", x_ub[i, j]) / s
                            )
                T.copy(out_ub, Out)

        return main

    if arm == "input_keep":

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
                    x = T.alloc_shared((R, C), "float32")
                    scale = T.alloc_shared((R, LANES), "float32")
                    for i, lane in T.Parallel(R, LANES):
                        m = T.alloc_var("float32", init=EPS)
                        for ch in T.serial(NCH):
                            j = ch * LANES + lane
                            vv = T.Cast("float32", x_ub[i, j])
                            x[i, j] = vv
                            m = T.max(m, T.max(vv, -vv))
                        scale[i, lane] = T.max(m, T.float32(EPS))
                    for i, lane in T.Parallel(R, LANES):
                        s = T.alloc_var("float32")
                        s = scale[i, lane]
                        for ch in T.serial(NCH):
                            j = ch * LANES + lane
                            out_ub[i, j] = T.Cast("float16", x[i, j] / s)
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
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                scale = T.alloc_shared((R, LANES), "float32")
                for i, lane in T.Parallel(R, LANES):
                    m = T.alloc_var("float32", init=EPS)
                    for ch in T.serial(NCH):
                        vv = T.Cast("float32", x_ub[i, ch * LANES + lane])
                        m = T.max(m, T.max(vv, -vv))
                    scale[i, lane] = T.max(m, T.float32(EPS))
                for i, lane in T.Parallel(R, LANES):
                    s = T.alloc_var("float32")
                    s = scale[i, lane]
                    for ch in T.serial(NCH):
                        j = ch * LANES + lane
                        out_ub[i, j] = T.Cast(
                            "float16", T.Cast("float32", x_ub[i, j]) / s
                        )
            T.copy(out_ub, Out)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    lanes = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    arm = sys.argv[4] if len(sys.argv) > 4 else "input_stream"
    if arm not in _ARMS:
        print(
            f"unknown arm {arm!r}; use input_keep|input_stream|frag_live|fold_frag_keep|reload",
            flush=True,
        )
        return 2
    if C % lanes != 0:
        print(f"C={C} must be divisible by lanes/LANES={lanes}", flush=True)
        return 2
    tag = f"sv2v_r{R}_c{C}_t{lanes}_{arm}"
    so = compile_prim(build_sv2v(R, C, lanes, arm), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
