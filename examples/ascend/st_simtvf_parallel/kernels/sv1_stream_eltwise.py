#!/usr/bin/env python3
"""ST-SV1 — Stream Parallel eltwise baseline (no loop-carry). Matrix: E∈{256,2048} × T=32."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv1(E: int, threads: int):
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
            with T.SimtVF(threads=threads):
                t1 = T.alloc_fragment((E,), "float32")
                t2 = T.alloc_fragment((E,), "float32")
                t3 = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    t1[i] = a_ub[i]
                    t2[i] = b_ub[i]
                    t3[i] = t1[i] + t2[i]
                    c_ub[i] = t3[i]
            T.copy(c_ub, C)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    tag = f"sv1_e{E}_t{threads}"
    so = compile_prim(build_sv1(E, threads), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
