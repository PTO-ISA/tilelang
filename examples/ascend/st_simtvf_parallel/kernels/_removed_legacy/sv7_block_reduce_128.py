#!/usr/bin/env python3
"""ST-SV7 — Block reduce map N=128 (1D). threads∈{32,128}."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv7(N: int, threads: int):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        In: T.Tensor((N,), "float32"),
        Out: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            in_ub = T.alloc_shared((N,), "float32")
            out_ub = T.alloc_shared((1,), "float32")
            T.copy(In, in_ub)
            with T.SimtVF(threads=threads):
                v = T.alloc_fragment((N,), "float32")
                red = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(N):
                    v[i] = in_ub[i]
                T.reduce_max(v, red)
                for _ in T.Parallel(1):
                    out_ub[0] = red[0]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    tag = f"sv7_n{N}_t{threads}"
    so = compile_prim(build_sv7(N, threads), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
