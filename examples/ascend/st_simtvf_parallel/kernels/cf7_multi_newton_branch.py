#!/usr/bin/env python3
"""ST-CF7 — multi-range slope-adaptive Newton recip (heavy uncommon / 3-arm CF).

Predicate on |x| magnitude bands (realistic recip / special-fn / solver pattern):
  |x| < EPS_FLOOR → y=0 (guard)
  EPS_FLOOR ≤ |x| < EPS_NEAR → fat arm: rough seed + max_iters Newton (uncommon)
  EPS_NEAR ≤ |x| < EPS_MID  → mid arm: rough seed + n_mid Newton
  |x| ≥ EPS_MID             → far arm: closed-form 1/x, 0 Newton iters

Simt: nested divergent if inside Parallel — fat/mid lanes stay in serial Newton;
far lanes early-exit (0 iters). Prefer T.max(v,-v).

Tags: cf7_e{E}_t{T}_pfat{P}_pmid{M}  (host skew; max_iters fixed in kernel).
Primary: E=256, T=32, pfat∈{5,25}, pmid=20, max_iters=8, n_mid=2.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

EPS_FLOOR = 1e-6
EPS_NEAR = 0.1
EPS_MID = 1.0
MAX_ITERS = 8
N_MID = 2


def build_cf7(
    E: int,
    threads: int,
    eps_floor: float = EPS_FLOOR,
    eps_near: float = EPS_NEAR,
    eps_mid: float = EPS_MID,
    max_iters: int = MAX_ITERS,
    n_mid: int = N_MID,
):
    import tilelang.ascend.language as T

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
                    if ax < eps_floor:
                        y[i] = T.float32(0.0)
                    else:
                        if ax < eps_near:
                            # fat / uncommon: rough sign seed + max_iters Newton
                            if x[i] >= T.float32(0.0):
                                y[i] = T.float32(1.0)
                            else:
                                y[i] = T.float32(-1.0)
                            for _t in T.serial(max_iters):
                                y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
                        else:
                            if ax < eps_mid:
                                # mid: rough seed + n_mid iters
                                if x[i] >= T.float32(0.0):
                                    y[i] = T.float32(1.0)
                                else:
                                    y[i] = T.float32(-1.0)
                                for _t in T.serial(n_mid):
                                    y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
                            else:
                                # far: closed form, 0 Newton (early exit)
                                y[i] = T.float32(1.0) / x[i]
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    pfat = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    pmid = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    tag = f"cf7_e{E}_t{threads}_pfat{pfat}_pmid{pmid}"
    so = compile_prim(build_cf7(E, threads), tag, target="ascend")
    if so is None:
        return 1
    print("SO", so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
