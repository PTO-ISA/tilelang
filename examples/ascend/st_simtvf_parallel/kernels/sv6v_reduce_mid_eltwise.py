#!/usr/bin/env python3
"""ST-SV6V — Case-3 mid-G reduce → reduced eltwise (SimdVF / VMI twin of SV6).

Same math/IO as Simt SV6 (gold identical, seed 6):
  X[R,C] fp16 → Y[R,CG] fp32,  Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)
No bcast. Defaults: R=64, C=128, G=32 → CG=4, lanes=64.
Tag: ``sv6v_r{R}_c{C}_g{G}_t{L}_{reload|keep_in_warp}``.

PRIMARY arm is ``reload`` (mapping mode **1**): at G=32 the group does not fit
≈32 SIMD arch VL regs, so the SIMD/VMI side must slice / reload from UB. We do
NOT fake a VL-keep of G=32 in 32 regs — the optional ``keep_in_warp`` arm only
swaps the working set to ``alloc_fragment`` to record where the overlay gives up.

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

ARMS = ("reload", "keep_in_warp")


def build_sv6v(R: int, C: int, G: int, lanes: int, arm: str = "reload"):
    import tilelang.ascend.language as T

    CG = C // G
    N = R * CG
    keep = arm == "keep_in_warp"

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
            for n in T.serial(N):
                for t in T.serial(G):
                    xt[t, n] = x_ub[n, t]
            with T.SimdVF(lanes=lanes):
                if keep:
                    xf = T.alloc_fragment((G * N,), "float32")
                else:
                    xf = T.alloc_shared((G * N,), "float32")
                for t in T.serial(G):
                    for n in T.Parallel(N):
                        xf[t * N + n] = T.Cast("float32", xt[t, n])
                for n in T.Parallel(N):
                    m_ub[n] = T.float32(0.0)
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
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    lanes = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    arm = sys.argv[5] if len(sys.argv) > 5 else "reload"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    if C % G != 0:
        print(f"C={C} must be divisible by G={G}", flush=True)
        return 2
    N = R * (C // G)
    if N % lanes != 0:
        print(f"lane-illegal: R*CG={N} must be a multiple of lanes={lanes}", flush=True)
        return 2
    tag = f"sv6v_r{R}_c{C}_g{G}_t{lanes}_{arm}"
    so = compile_prim(build_sv6v(R, C, G, lanes, arm), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
