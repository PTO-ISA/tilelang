#!/usr/bin/env python3
"""ST-SV2 — Fold scale ALWAYS KEEP; input KEEP vs STREAM by R (Simt A).

Gold (fold along C; LANES = threads):
  scale[i, lane] = max(EPS, max over ch of abs(X[i, ch*LANES + lane]))
  out[i, j] = x[i, j] / scale[i, lane]   (j = ch*LANES + lane)

**Contract (Lok 2026-09-29):**
- Scale ALWAYS KEEP in ``T.alloc_fragment((R, LANES))`` across the consumer.
  Do NOT use scale remat/reload as the primary turning point.
- Sensitivity is **input (x) KEEP vs STREAM** by R:
  - R≈16 ``input_keep``: x + scale both live in fragment (fits RF).
  - R≥32 ``input_stream``: x STREAMs from x_ub; scale still KEEP in fragment.

Arms (separate ``@T.prim_func`` bodies):
- ``input_keep``: fold while filling ``x`` fragment; consumer reads ``x`` + ``scale``
  fragments. Intended geom R=16.
- ``input_stream`` (aliases ``frag_live`` / ``fold_frag_keep``): fold → scale
  fragment only; consumer STREAMs ``x_ub``. Intended geom R=32/64.
- ``reload``: DEMOTEd legacy scale-remat foil (not primary; not in oneshot).

Geom: C % LANES == 0. Prefer C=64 (multi-chunk when LANES=32 → NCH=2).
Defaults: R=32 C=64 T=32 arm=input_stream.

Tags: ``sv2_r{R}_c{C}_t{T}_{input_keep|input_stream|frag_live|...}``.

SV5/SV8: ``alloc_var`` + reassignment; abs via ``T.max(v,-v)``;
``Parallel(R,LANES)+serial(NCH)`` (avoid ``j//LANES`` InverseAffine).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402

EPS = 1e-6
_STREAM_ARMS = ("input_stream", "frag_live", "fold_frag_keep")
_ARMS = ("input_keep",) + _STREAM_ARMS + ("reload",)


def build_sv2(R: int, C: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    LANES = threads
    if C % LANES != 0:
        raise ValueError(f"C={C} must be divisible by LANES=threads={LANES}")
    NCH = C // LANES

    if arm == "reload":
        # DEMOTEd foil — scale remat; not the primary ST contrast
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
                with vf_region(threads):
                    scale = T.alloc_fragment((R, LANES), "float32")
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
        # R≈16: KEEP x fragment + scale fragment
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
                    x = T.alloc_fragment((R, C), "float32")
                    scale = T.alloc_fragment((R, LANES), "float32")
                    for i, lane in T.Parallel(R, LANES):
                        m = T.alloc_var("float32", init=EPS)
                        for ch in T.serial(NCH):
                            j = ch * LANES + lane
                            vv = T.Cast("float32", x_ub[i, j])
                            x[i, j] = vv  # KEEP input in fragment
                            m = T.max(m, T.max(vv, -vv))
                        scale[i, lane] = T.max(m, T.float32(EPS))
                    # consumer: both x and scale from fragment (scale ALWAYS KEEP)
                    for i, lane in T.Parallel(R, LANES):
                        s = T.alloc_var("float32")
                        s = scale[i, lane]
                        for ch in T.serial(NCH):
                            j = ch * LANES + lane
                            out_ub[i, j] = T.Cast("float16", x[i, j] / s)
                T.copy(out_ub, Out)

        return main

    # input_stream / frag_live / fold_frag_keep: STREAM x; KEEP scale fragment
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
                scale = T.alloc_fragment((R, LANES), "float32")
                for i, lane in T.Parallel(R, LANES):
                    m = T.alloc_var("float32", init=EPS)
                    for ch in T.serial(NCH):
                        vv = T.Cast("float32", x_ub[i, ch * LANES + lane])
                        m = T.max(m, T.max(vv, -vv))
                    scale[i, lane] = T.max(m, T.float32(EPS))
                # STREAM x from x_ub; KEEP scale in fragment
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
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    arm = sys.argv[4] if len(sys.argv) > 4 else "input_stream"
    if arm not in _ARMS:
        print(
            f"unknown arm {arm!r}; use input_keep|input_stream|frag_live|fold_frag_keep|reload",
            flush=True,
        )
        return 2
    if C % threads != 0:
        print(f"C={C} must be divisible by threads/LANES={threads}", flush=True)
        return 2
    tag = f"sv2_r{R}_c{C}_t{threads}_{arm}"
    so = compile_prim(build_sv2(R, C, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
