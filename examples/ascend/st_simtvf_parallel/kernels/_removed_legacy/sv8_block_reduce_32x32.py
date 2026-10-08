#!/usr/bin/env python3
"""LEGACY ST-SV8 — Block reduce map 32×32 (pre Case-3).

**LEGACY / retained for reference.** Active Case-3 SV8 is
``kernels/sv8_case3_bcast.py`` (tags ``sv8_r*_c*_g*_t*_{live,spill_dist}``).

Old tags ``sv8_32x32_t*``: 2D hierarchical reduce_max + normalize bcast.

Uses T.reduce_max(..., dim=1) like green SV7 / flash-attn examples — avoids
serial scalar rebind and mixed tile[i,0]/tile[i,j] layout conflict.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv8(threads: int):
    import tilelang.ascend.language as T

    N = 32

    @T.prim_func
    def main(
        In: T.Tensor((N, N), "float32"),
        Out: T.Tensor((N, N), "float32"),
    ):
        with T.Kernel(1):
            in_ub = T.alloc_shared((N, N), "float32")
            out_ub = T.alloc_shared((N, N), "float32")
            T.copy(In, in_ub)
            with T.SimtVF(threads=threads):
                tile = T.alloc_fragment((N, N), "float32")
                row_max = T.alloc_fragment((N,), "float32")
                all_max = T.alloc_fragment((1,), "float32")
                for i, j in T.Parallel(N, N):
                    tile[i, j] = in_ub[i, j]
                T.reduce_max(tile, row_max, dim=1)
                T.reduce_max(row_max, all_max)
                for i, j in T.Parallel(N, N):
                    out_ub[i, j] = tile[i, j] / all_max[0]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    tag = f"sv8_32x32_t{threads}"
    so = compile_prim(build_sv8(threads), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
