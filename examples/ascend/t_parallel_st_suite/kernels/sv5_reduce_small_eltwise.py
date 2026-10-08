#!/usr/bin/env python3
"""ST-SV5 — Case-3 small-block reduce → reduced eltwise (fused only).

Input ``X[R,C]`` fp16; output ``Y[R,CG]`` fp32 with ``CG=C/G``:
``Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)``.

Stops at the reduced grid — **no bcast back to full C**.
Defaults: R=64, C=128, G=16 → CG=8, threads=32.
Tag: ``sv5_r{R}_c{C}_g{G}_t{T}_{arm}``.

PRIMARY sensitivity (2026-10-06 rewrite): **group-input RF capacity during
absmax**, not post-reduce scale residency.

- ``keep_in_rf`` : the group **inputs** are preloaded into a fragment
  (``xf[R,CG,G]`` fp32, RF) and the absmax runs **from the fragment** — the
  inputs never go back to UB during the reduce. With G=16 the whole group fits
  in ≈32 SIMD arch VL regs / one SIMT thread's RF, so this is the rung where a
  keep is supposed to be free. ``Y = sf_inv`` written straight from the reduce.
- ``ub_stream`` : no preload. The absmax streams the group out of shared
  ``x_ub`` one ``serial(G)`` step at a time. This is the **reload foil** for
  small G: identical arithmetic, identical gold, only the input residency
  differs.

Legacy arms (post-reduce *scale* residency + a sink'd subsequent compute) are
kept working and still selectable, but they answer a different question:

- ``keep_reg``  : inverse scale stays live in a fragment (``sf_inv[R,CG]``)
  and a second ``Parallel`` loop reads it, then does ``serial(G)`` compute.
- ``ub_reload`` : inverse scale stored to shared UB and reloaded by a later
  ``Parallel`` loop before the identical subsequent compute.

Gold is IDENTICAL for all four arms and for the pre-arm no-suffix tag.

**Ordering is load-bearing** for the legacy arms (opsim 2026-10-05): the gold
store ``y_ub[i,g] = s`` must come *before* the ``serial(G)`` accumulate, else
SimtVF/bisheng clobbers the ``float sf_inv[…]`` local array (≈63% of lanes come
back as the 1e-6 clamp, maxabs≈999999.75). See
``reports/ST_SV5_REDUCE_SMALL.md`` §"RF clobber". The new ``keep_in_rf`` arm
reads its fragment inside the *same* Parallel body that filled it, so it is not
exposed to that cross-loop clobber.

Programming notes (green SV2/SV5):
- ``T.alloc_var`` + reassignment for mutable absmax
- abs via ``T.max(v, -v)`` (not ``T.abs``)
- avoid ``j//G`` in fragment indexing (InverseAffine) — the 3-D ``[R,CG,G]``
  fragment keeps every index affine in the Parallel vars
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

ARMS = ("keep_in_rf", "ub_stream", "keep_reg", "ub_reload")


def build_sv5(R: int, C: int, G: int, threads: int, arm: str = "keep_in_rf"):
    import tilelang.ascend.language as T

    CG = C // G

    if arm == "keep_in_rf":
        # PRIMARY keep: group inputs preloaded into RF (fragment), absmax from RF.
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
                    # preload the whole group into RF
                    # (separate Parallel loop: one ParallelOp may record only ONE
                    #  index pattern per buffer — a write+read of xf inside the
                    #  same Parallel body trips
                    #  `RecordBufferAccess: xf: (i,g,t) and (i,g,t)`)
                    for i, g in T.Parallel(R, CG):
                        for t in T.serial(G):
                            xf[i, g, t] = T.Cast("float32", x_ub[i, g * G + t])
                    # absmax FROM the fragment (inputs stay in RF)
                    for i, g in T.Parallel(R, CG):
                        m = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = xf[i, g, t]
                            m = T.max(m, T.max(vv, -vv))
                        m = T.max(m, T.float32(1e-6))
                        y_ub[i, g] = T.float32(1.0) / m
                T.copy(y_ub, Y)

        return main

    if arm == "ub_stream":
        # PRIMARY foil: no preload — absmax streams from shared UB each step.
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

    if arm == "ub_reload":

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Y: T.Tensor((R, CG), "float32"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                y_ub = T.alloc_shared((R, CG), "float32")
                # post-reduce scale parked in UB (shared) instead of RF
                scale_ub = T.alloc_shared((R, CG), "float32")
                # sink for the subsequent compute — keeps it alive, off the gold path
                sink_ub = T.alloc_shared((R, CG), "float32")
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    for i, g in T.Parallel(R, CG):
                        m = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = T.Cast("float32", x_ub[i, g * G + t])
                            m = T.max(m, T.max(vv, -vv))
                        m = T.max(m, T.float32(1e-6))
                        scale_ub[i, g] = T.float32(1.0) / m
                    for i, g in T.Parallel(R, CG):
                        s = T.alloc_var("float32")
                        s = scale_ub[i, g]
                        y_ub[i, g] = s          # gold store FIRST (see docstring)
                        acc = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = T.Cast("float32", x_ub[i, g * G + t])
                            acc = acc + T.max(vv, -vv) * s
                        sink_ub[i, g] = acc     # subsequent compute, off gold path
                T.copy(y_ub, Y)

        return main

    # arm == "keep_reg" (legacy): post-reduce scale stays live in the fragment / RF
    @T.prim_func
    def main(
        X: T.Tensor((R, C), "float16"),
        Y: T.Tensor((R, CG), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((R, C), "float16")
            y_ub = T.alloc_shared((R, CG), "float32")
            sink_ub = T.alloc_shared((R, CG), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                sf_inv = T.alloc_fragment((R, CG), "float32")
                for i, g in T.Parallel(R, CG):
                    m = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        m = T.max(m, T.max(vv, -vv))
                    m = T.max(m, T.float32(1e-6))
                    sf_inv[i, g] = T.float32(1.0) / m
                for i, g in T.Parallel(R, CG):
                    s = T.alloc_var("float32")
                    s = sf_inv[i, g]        # live from RF, no round-trip
                    y_ub[i, g] = s          # gold store FIRST (see docstring)
                    acc = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        acc = acc + T.max(vv, -vv) * s
                    sink_ub[i, g] = acc     # subsequent compute, off gold path
            T.copy(y_ub, Y)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "keep_in_rf"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    tag = f"sv5_r{R}_c{C}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sv5(R, C, G, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
