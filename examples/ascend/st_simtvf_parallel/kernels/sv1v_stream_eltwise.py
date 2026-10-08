#!/usr/bin/env python3
"""ST-SV1V — Stream Parallel eltwise baseline (SimdVF / VMI twin of SV1).

Same math/IO as Simt SV1:
  C[E] = A[E] + B[E]  (fp32), no loop-carry, no bcast.
Defaults: E=256, lanes=64. Matrix: E∈{256,2048} × lanes=64.
Tag: ``sv1v_e{E}_t{L}`` (``t`` = SimdVF lanes; parallel to Simt T).

VMI notes (green SimdVF AB):
- ``T.SimdVF(lanes=…)`` + ``target=pto`` + overlay lib
- Prefer ``alloc_shared`` for VF working set (not fragment)
- No ``T.abs``; no loop-carried state here
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv1v(E: int, lanes: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
        C: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            c_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=lanes):
                # SimdVF-green: stream temps live in shared (not fragment)
                t1 = T.alloc_shared((E,), "float32")
                t2 = T.alloc_shared((E,), "float32")
                t3 = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    t1[i] = a_ub[i]
                    t2[i] = b_ub[i]
                    t3[i] = t1[i] + t2[i]
                    c_ub[i] = t3[i]
            T.copy(c_ub, C)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    tag = f"sv1v_e{E}_t{lanes}"
    so = compile_prim(build_sv1v(E, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
