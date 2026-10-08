#!/usr/bin/env python3
"""ST-CF6V — Newton recip + near-0 via mask select (SimdVF / VMI twin of CF6).

Near-0 → y=0; else seed 1/x + N_FAST Newton. Div/Sub/Max not Parallel-whitelist;
serial Newton inside SimdVF fails AIV. ABI: Newton in Kernel-scope serial outside
SimdVF (carry via y_work[i] stores — no immutable local); VF Parallel publishes.
Tags: cf6v_e{E}_t{L}_pnear{P}. Primary: E=256, lanes=64.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

EPS = 1e-4
N_FAST = 3


def build_cf6v(E: int, lanes: int, eps: float = EPS, n_fast: int = N_FAST):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        X: T.Tensor((E,), "float32"),
        Y: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_work = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            for i in T.serial(E):
                xi = x_ub[i]
                ax = T.Select(xi >= T.float32(0.0), xi, xi * T.float32(-1.0))
                safe_x = T.Select(ax < eps, T.float32(1.0), xi)
                y_work[i] = T.float32(1.0) / safe_x
                for _t in T.serial(n_fast):
                    y_work[i] = y_work[i] * (
                        T.float32(2.0) + (safe_x * y_work[i]) * T.float32(-1.0)
                    )
                y_work[i] = T.Select(ax < eps, T.float32(0.0), y_work[i])
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = y_work[i]
            T.copy(y_ub, Y)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    pnear = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    tag = f"cf6v_e{E}_t{lanes}_pnear{pnear}"
    so = compile_prim(build_cf6v(E, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
