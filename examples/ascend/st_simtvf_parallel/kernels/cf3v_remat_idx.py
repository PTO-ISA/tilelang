#!/usr/bin/env python3
"""ST-CF3V — remat_idx only (SimdVF / VMI twin of CF3).

KEEP scores; kill by rematted Parallel index i == victim[k] via mask select.
No idxs KEEP buffer. Mode 3 on scores; index rematerialized at use.
Tags: cf3v_e{E}_k{K}_t{L}_remat_idx. Primary: E=256, K=8, lanes=64.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

NEG = float(-3.402823e38)


def build_cf3v(E: int, K: int, lanes: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        Victim: T.Tensor((K,), "int32"),
        Out: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            vic_ub = T.alloc_shared((K,), "int32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, s)
            T.copy(Victim, vic_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s[i]
                for k in T.serial(K):
                    v = vic_ub[k]
                    for i in T.Parallel(E):
                        # remat index i in mask predicate
                        scores[i] = T.Select(i == v, T.float32(NEG), scores[i])
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lanes = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    tag = f"cf3v_e{E}_k{K}_t{lanes}_remat_idx"
    so = compile_prim(build_cf3v(E, K, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
