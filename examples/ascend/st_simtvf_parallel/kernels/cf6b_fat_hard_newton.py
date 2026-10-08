#!/usr/bin/env python3
"""ST-CF6b — fat hard-band Newton + thin easy path (divergence stress).

Design: reports/ST_CF6_DIVERGENCE_CASE.md
vs current CF6 (near-0→0 else fixed N_FAST=3): this micro makes the
HARD arm (eps ≤ |x| < HARD_HI) much fatter than the EASY arm so Simt
SHOULD beat lockstep CF6bd when phard is low.

Gold:
  ax = abs(x)
  if ax < EPS:          y = 0
  elif ax < HARD_HI:    y = seed 1/x; N_HARD Newton; N_POLY Horner polish
  else:                 y = seed 1/x; N_FAST Newton

phard is host-data skew only (tag). Tags: cf6b_e{E}_t{T}_phard{P}.
Primary: E=256, T=32, phard∈{5,25}.

Wired: oneshot_cf6b_cf6bd.sh + run_opsim_generic cf6b_ branch.
Keep all Newton/poly INSIDE SimtVF Parallel (do not hoist → CF6v tax).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

EPS = 1e-4
HARD_HI = 1e-1
N_FAST = 2
N_HARD = 12
# Horner coeffs for hard-arm polish (fixed; gold must match)
POLY_C = (0.0, 1e-6, -2e-6, 3e-6, -4e-6, 5e-6, -6e-6, 7e-6)  # c0..c7
N_POLY = len(POLY_C)


def build_cf6b(
    E: int,
    threads: int,
    eps: float = EPS,
    hard_hi: float = HARD_HI,
    n_fast: int = N_FAST,
    n_hard: int = N_HARD,
):
    import tilelang.ascend.language as T

    c0, c1, c2, c3, c4, c5, c6, c7 = [float(c) for c in POLY_C]

    @T.prim_func
    def main(
        X: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                x = T.alloc_fragment((E,), "float32")
                y = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    x[i] = x_ub[i]
                for i in T.Parallel(E):
                    ax = T.max(x[i], -x[i])
                    if ax < eps:
                        y[i] = T.float32(0.0)
                    elif ax < hard_hi:
                        # FAT hard arm: many Newton iters + Horner polish
                        y[i] = T.float32(1.0) / x[i]
                        for _t in T.serial(n_hard):
                            y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
                        # Horner: ((((c7*x+c6)*x+c5)*x+...) + y  (alloc_var: mutable)
                        p = T.alloc_var("float32")
                        p = T.float32(c7)
                        p = p * x[i] + T.float32(c6)
                        p = p * x[i] + T.float32(c5)
                        p = p * x[i] + T.float32(c4)
                        p = p * x[i] + T.float32(c3)
                        p = p * x[i] + T.float32(c2)
                        p = p * x[i] + T.float32(c1)
                        p = p * x[i] + T.float32(c0)
                        y[i] = y[i] + p
                    else:
                        # THIN easy arm
                        y[i] = T.float32(1.0) / x[i]
                        for _t in T.serial(n_fast):
                            y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def reference(x, eps=EPS, hard_hi=HARD_HI, n_fast=N_FAST, n_hard=N_HARD):
    """Host gold matching Simt CF6b."""
    import numpy as np

    out = np.zeros_like(x, dtype=np.float32)
    cs = [np.float32(c) for c in POLY_C]
    for i in range(x.shape[0]):
        xi = np.float32(x[i])
        ax = abs(float(xi))
        if ax < float(eps):
            out[i] = 0.0
        elif ax < float(hard_hi):
            y = np.float32(1.0) / xi
            for _ in range(n_hard):
                y = y * (np.float32(2.0) - xi * y)
            p = cs[7]
            for k in range(6, -1, -1):
                p = p * xi + cs[k]
            out[i] = y + p
        else:
            y = np.float32(1.0) / xi
            for _ in range(n_fast):
                y = y * (np.float32(2.0) - xi * y)
            out[i] = y
    return out


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    phard = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tag = f"cf6b_e{E}_t{threads}_phard{phard}"
    so = compile_prim(build_cf6b(E, threads), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
