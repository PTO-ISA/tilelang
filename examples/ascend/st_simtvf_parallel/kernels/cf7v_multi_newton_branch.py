#!/usr/bin/env python3
"""ST-CF7V — multi-range Newton via nested Select + always-max lockstep (VMI twin of CF7).

Same 3 magnitude arms as CF7 (floor / fat max_iters / mid n_mid / far closed-form).
Faithful Simd translation: Nested T.Select picks arm; Newton body must live in
Kernel-scope serial *outside* SimdVF (Div/Sub/Max not Parallel-whitelist; serial-in-VF
→ AIV "Unsupported scalar instruction") — same ABI as CF5V/CF6V.

Lockstep tax (even with ABI hoist): every lane pays fat+mid+far compute, then Select.
Carry Newton via shared stores (no immutable local rebind — same as CF6V y_work[i]).
Simt CF7 early-exits per-lane (far = 0 Newton; mid = n_mid only). No AABBCC/vf_fuse.

Tags: cf7v_e{E}_t{L}_pfat{P}_pmid{M}. Primary: E=256, lanes=64, pfat∈{5,25}, pmid=20.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

EPS_FLOOR = 1e-6
EPS_NEAR = 0.1
EPS_MID = 1.0
MAX_ITERS = 8
N_MID = 2


def build_cf7v(
    E: int,
    lanes: int,
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
            y_fat_ub = T.alloc_shared((E,), "float32")
            y_mid_ub = T.alloc_shared((E,), "float32")
            y_work = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            # ABI: Div/Sub/Max Newton outside SimdVF; always compute all arms (lockstep)
            # Carry via shared stores (immutable local rebind fails eager builder).
            for i in T.serial(E):
                xi = x_ub[i]
                ax = T.Select(xi >= T.float32(0.0), xi, xi * T.float32(-1.0))
                safe_x = T.Select(ax < eps_floor, T.float32(1.0), xi)
                sign_seed = T.Select(xi >= T.float32(0.0), T.float32(1.0), T.float32(-1.0))
                # fat arm: rough seed + max_iters
                y_fat_ub[i] = sign_seed
                for _t in T.serial(max_iters):
                    y_fat_ub[i] = y_fat_ub[i] * (
                        T.float32(2.0) + (safe_x * y_fat_ub[i]) * T.float32(-1.0)
                    )
                # mid arm: rough seed + n_mid
                y_mid_ub[i] = sign_seed
                for _t in T.serial(n_mid):
                    y_mid_ub[i] = y_mid_ub[i] * (
                        T.float32(2.0) + (safe_x * y_mid_ub[i]) * T.float32(-1.0)
                    )
                # far arm: closed form
                y_far = T.float32(1.0) / safe_x
                # nested Select ≡ nested if (no divergent early exit)
                y_work[i] = T.Select(
                    ax < eps_floor,
                    T.float32(0.0),
                    T.Select(
                        ax < eps_near,
                        y_fat_ub[i],
                        T.Select(ax < eps_mid, y_mid_ub[i], y_far),
                    ),
                )
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
    pfat = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    pmid = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    tag = f"cf7v_e{E}_t{lanes}_pfat{pfat}_pmid{pmid}"
    so = compile_prim(build_cf7v(E, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
