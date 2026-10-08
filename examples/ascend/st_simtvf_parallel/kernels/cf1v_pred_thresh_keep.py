#!/usr/bin/env python3
"""ST-CF1V — thresh kill KEEP (SimdVF / VMI twin of CF1).

Same loops/arms as Simt CF1; control flow via mask select (T.Select),
NOT divergent if. Scores stay-alive across serial(K) (mode 3).
Tags: cf1v_e{E}_k{K}_t{L}_keep. Primary: E=256, K∈{1,8}, lanes=64.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402

NEG = float(-3.402823e38)


def build_cf1v(E: int, K: int, lanes: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        Thresh: T.Tensor((K,), "float32"),
        Out: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, s)
            T.copy(Thresh, thr_ub)
            with T.SimdVF(lanes=lanes):
                # mode-3 stay-alive scores (shared under SimdVF ABI)
                scores = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s[i]
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        # mask select: predicated kill (not divergent if)
                        scores[i] = T.Select(scores[i] > thr, T.float32(NEG), scores[i])
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
    tag = f"cf1v_e{E}_k{K}_t{lanes}_keep"
    so = compile_prim(build_cf1v(E, K, lanes), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
