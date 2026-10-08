#!/usr/bin/env python3
"""Probe: VMI (SimdVF) Case-3 reduce with a TRANSPOSED UB working set.

VerifyParallelToPTO rejects the natural layout: reducing group g over t reads
``xf[n*G + t]`` which is neither lane-continuous nor lane-uniform across the
Parallel var ``n``. This probe stages a transposed copy ``xt[t, n]`` with G
strided GM→UB copies outside the VF region, so every VF access is continuous
over the lane domain ``n``.

Usage: _probe_sv6v_t.py R C G lanes [variant]   variant = copyslice | copy2d
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402


def build(R, C, G, lanes, variant):
    import tilelang.ascend.language as T

    CG = C // G
    N = R * CG

    @T.prim_func
    def main(
        X: T.Tensor((N, G), "float16"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N, G), "float16")
            xt = T.alloc_shared((G, N), "float16")
            xf = T.alloc_shared((G * N,), "float32")
            m_ub = T.alloc_shared((N,), "float32")
            y_ub = T.alloc_shared((N,), "float32")
            if variant == "ubub":
                # contiguous GM->UB, then UB->UB column copies for the transpose
                T.copy(X, x_ub)
                for t in T.serial(G):
                    T.copy(x_ub[:, t], xt[t, :])
            elif variant == "scalar":
                # contiguous GM->UB, then a scalar transpose OUTSIDE the VF region
                T.copy(X, x_ub)
                for n in T.serial(N):
                    for t in T.serial(G):
                        xt[t, n] = x_ub[n, t]
            else:  # copyslice: strided GM->UB column copies
                for t in T.serial(G):
                    T.copy(X[:, t], xt[t, :])
            with T.SimdVF(lanes=lanes):
                for t in T.serial(G):
                    for n in T.Parallel(N):
                        xf[t * N + n] = T.Cast("float32", xt[t, n])
                for n in T.Parallel(N):
                    m_ub[n] = T.float32(0.0)
                for t in T.serial(G):
                    for n in T.Parallel(N):
                        vv = xf[t * N + n]
                        m_ub[n] = T.max(m_ub[n], T.max(vv, -vv))
            # CF5V lesson: Div is not on the Parallel->PTO whitelist - do the
            # reciprocal (and the 1e-6 clamp) in a Kernel-scope serial loop
            # OUTSIDE the SimdVF region.
            for n in T.serial(N):
                y_ub[n] = T.float32(1.0) / T.max(m_ub[n], T.float32(1e-6))
            T.copy(y_ub, Y)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    R = int(sys.argv[1]); C = int(sys.argv[2]); G = int(sys.argv[3]); lanes = int(sys.argv[4])
    variant = sys.argv[5] if len(sys.argv) > 5 else "copyslice"
    tag = f"probe_sv6v_t_r{R}_c{C}_g{G}_t{lanes}_{variant}"
    so = compile_prim(build(R, C, G, lanes, variant), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
