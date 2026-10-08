#!/usr/bin/env python3
"""ST-SV5V — Case-3 small-G reduce → reduced eltwise (SimdVF / VMI twin of SV5).

Same math/IO as Simt SV5 (gold identical, seed 5):
  X[R,C] fp16 → Y[R,CG] fp32,  Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)
No bcast. Defaults: R=64, C=128, G=16 → CG=8, lanes=64.
Tag: ``sv5v_r{R}_c{C}_g{G}_t{L}_{arm}`` (``t`` = SimdVF lanes).

PRIMARY arms (twin of the 2026-10-06 RF-capacity rewrite):
- ``keep_in_rf`` : the f32 group working set lives in a **fragment** (RF / VL
  regs) and the absmax reads it from there — expect VL pressure, that is the
  point of the rung.
- ``ub_stream``  : the working set stays in **shared UB** (``alloc_shared``) and
  the absmax streams it one ``serial(G)`` step at a time (mapping mode 1).

Legacy arms ``keep_reg`` / ``ub_reload`` (post-reduce *scale* residency) are
kept for the historical rows; they use the pre-2026-10-06 2-D body and are known
to hit the Case-3 ABI walls.

VMI ABI (three stacked walls found on 2026-10-06, see reports/ST_CASE3_VMI.md):
1. ``xf[n*G + t]`` — reducing a group over ``t`` with lanes over ``n`` is neither
   lane-continuous nor lane-uniform → ``[VerifyParallelToPTO] cannot prove the
   address is continuous``. Fix: stage a **transposed** working set ``xt[t, n]``
   so every VF access is continuous over the lane domain ``n``.
2. the transpose cannot be a strided GM→UB copy (``T.copy(X[:, t], xt[t, :])`` →
   ``PTO GM->UB MTE destination row stride must be 32-byte aligned, got 2``), and
   the UB→UB column copy crashes the overlay. It is therefore done as a **scalar
   transpose outside the VF region** (prologue cost; it is NOT inside the VF
   cycles that the theory model compares against).
3. ``Div`` is not on the Parallel→PTO whitelist (CF5V lesson) → the reciprocal
   and the 1e-6 clamp run in a Kernel-scope ``serial`` loop **outside** SimdVF.
Lane legality (SV3V lesson): every ``Parallel`` is 1-D over ``N = R*CG``
(512/256/128/64) or ``lanes``, all multiples of ``lanes``; no ``alloc_var`` in
the VF body (the running absmax lives in UB); no guards in the body.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

ARMS = ("keep_in_rf", "ub_stream", "keep_reg", "ub_reload")


def build_sv5v(R: int, C: int, G: int, lanes: int, arm: str = "ub_stream"):
    import tilelang.ascend.language as T

    CG = C // G
    N = R * CG

    if arm in ("keep_in_rf", "ub_stream"):
        keep = arm == "keep_in_rf"

        @T.prim_func
        def main(
            X: T.Tensor((N, G), "float16"),
            Y: T.Tensor((N,), "float32"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((N, G), "float16")
                xt = T.alloc_shared((G, N), "float16")
                m_ub = T.alloc_shared((N,), "float32")
                y_ub = T.alloc_shared((N,), "float32")
                T.copy(X, x_ub)
                # scalar transpose, outside the VF region (ABI wall #2)
                for n in T.serial(N):
                    for t in T.serial(G):
                        xt[t, n] = x_ub[n, t]
                with T.SimdVF(lanes=lanes):
                    if keep:
                        xf = T.alloc_fragment((G * N,), "float32")   # inputs in RF
                    else:
                        xf = T.alloc_shared((G * N,), "float32")     # inputs in UB
                    for t in T.serial(G):
                        for n in T.Parallel(N):
                            xf[t * N + n] = T.Cast("float32", xt[t, n])
                    for n in T.Parallel(N):
                        m_ub[n] = T.float32(0.0)
                    for t in T.serial(G):
                        for n in T.Parallel(N):
                            vv = xf[t * N + n]
                            m_ub[n] = T.max(m_ub[n], T.max(vv, -vv))
                # Div outside the VF region (ABI wall #3)
                for n in T.serial(N):
                    y_ub[n] = T.float32(1.0) / T.max(m_ub[n], T.float32(1e-6))
                T.copy(y_ub, Y)

        return main

    # ---- legacy post-reduce scale-residency arms (historical rows) ----
    if arm == "ub_reload":

        @T.prim_func
        def main(
            X: T.Tensor((R, C), "float16"),
            Y: T.Tensor((R, CG), "float32"),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((R, C), "float16")
                y_ub = T.alloc_shared((R, CG), "float32")
                scale_ub = T.alloc_shared((R, CG), "float32")
                sink_ub = T.alloc_shared((R, CG), "float32")
                T.copy(X, x_ub)
                with T.SimdVF(lanes=lanes):
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
                        y_ub[i, g] = s          # gold store FIRST
                        acc = T.alloc_var("float32", init=0.0)
                        for t in T.serial(G):
                            vv = T.Cast("float32", x_ub[i, g * G + t])
                            acc = acc + T.max(vv, -vv) * s
                        sink_ub[i, g] = acc
                T.copy(y_ub, Y)

        return main

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
            with T.SimdVF(lanes=lanes):
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
                    s = sf_inv[i, g]
                    y_ub[i, g] = s
                    acc = T.alloc_var("float32", init=0.0)
                    for t in T.serial(G):
                        vv = T.Cast("float32", x_ub[i, g * G + t])
                        acc = acc + T.max(vv, -vv) * s
                    sink_ub[i, g] = acc
            T.copy(y_ub, Y)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    lanes = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    arm = sys.argv[5] if len(sys.argv) > 5 else "ub_stream"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    if arm in ("keep_in_rf", "ub_stream"):
        N = R * (C // G)
        if N % lanes != 0:
            print(f"lane-illegal: R*CG={N} must be a multiple of lanes={lanes}", flush=True)
            return 2
    tag = f"sv5v_r{R}_c{C}_g{G}_t{lanes}_{arm}"
    so = compile_prim(build_sv5v(R, C, G, lanes, arm), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
