#!/usr/bin/env python3
"""ST-SV7V — Case-3 large-G reduce → reduced eltwise (SimdVF / VMI twin of SV7).

Same math/IO as Simt SV7 (gold identical, seed 6):
  X[R,C] fp16 → Y[R,CG] fp32,  Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)
No bcast. Defaults: R=64, C=128, G=128 → CG=1 (also G=64 → CG=2), lanes=64.
Tag: ``sv7v_r{R}_c{C}_g{G}_t{L}_{reload|multiwarp_ub}``.

- ``reload``       : PRIMARY (mapping mode **1**) — single ``serial(G)`` stream
  of the group out of the shared UB working set.
- ``multiwarp_ub`` : chunked multi-pass twin of the Simt arm — NW=4 chunks of
  G/4 reduced into shared ``part_ub[NW, N]`` partials, then a final pass over
  the NW partials. Expresses the UB-allreduce cost on the SIMD side.

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

ARMS = ("reload", "multiwarp_ub")
NW = 4


def build_sv7v(R: int, C: int, G: int, lanes: int, arm: str = "reload"):
    import tilelang.ascend.language as T

    CG = C // G
    N = R * CG
    CH = G // NW

    @T.prim_func
    def main(
        X: T.Tensor((N, G), "float16"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N, G), "float16")
            xt = T.alloc_shared((G, N), "float16")
            xf = T.alloc_shared((G * N,), "float32")
            part_ub = T.alloc_shared((NW * N,), "float32")
            m_ub = T.alloc_shared((N,), "float32")
            y_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for n in T.serial(N):
                for t in T.serial(G):
                    xt[t, n] = x_ub[n, t]
            with T.SimdVF(lanes=lanes):
                for t in T.serial(G):
                    for n in T.Parallel(N):
                        xf[t * N + n] = T.Cast("float32", xt[t, n])
                for n in T.Parallel(N):
                    m_ub[n] = T.float32(0.0)
                if arm == "multiwarp_ub":
                    for w in T.serial(NW):
                        for n in T.Parallel(N):
                            part_ub[w * N + n] = T.float32(0.0)
                        for t in T.serial(CH):
                            for n in T.Parallel(N):
                                vv = xf[(w * CH + t) * N + n]
                                part_ub[w * N + n] = T.max(
                                    part_ub[w * N + n], T.max(vv, -vv)
                                )
                    for w in T.serial(NW):
                        for n in T.Parallel(N):
                            m_ub[n] = T.max(m_ub[n], part_ub[w * N + n])
                else:
                    for t in T.serial(G):
                        for n in T.Parallel(N):
                            vv = xf[t * N + n]
                            m_ub[n] = T.max(m_ub[n], T.max(vv, -vv))
            for n in T.serial(N):
                y_ub[n] = T.float32(1.0) / T.max(m_ub[n], T.float32(1e-6))
            T.copy(y_ub, Y)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    lanes = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    arm = sys.argv[5] if len(sys.argv) > 5 else "reload"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    if arm == "multiwarp_ub" and G % NW != 0:
        print(f"G={G} must be divisible by NW={NW}", flush=True)
        return 2
    N = R * (C // G)
    if N % lanes != 0:
        print(f"lane-illegal: R*CG={N} must be a multiple of lanes={lanes}", flush=True)
        return 2
    tag = f"sv7v_r{R}_c{C}_g{G}_t{lanes}_{arm}"
    so = compile_prim(build_sv7v(R, C, G, lanes, arm), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
