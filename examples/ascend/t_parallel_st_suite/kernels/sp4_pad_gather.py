#!/usr/bin/env python3
"""ST-SP4 — Pad gather: Simt early-skip vs Simd mask.

**Single sensitivity:** predicated / divergent gather over pad holes
(Simt early-skip vs Simd mask on twin). Alignment waste vs dense SV4.

**Workload remark:** fused alignment holes from ``get_fused_mapping``.

Pad fraction encoded in tag ``pad{Pct}`` (host synthesizes ``Expert[p]<0``).
In-place mask: ``Out[p]=Buf[p] if Expert[p]>=0 else 0``.

Tags: ``sp4_e{Eexp}_h{H}_t{Thr}_pad{Pct}``.
Primary: Eexp=64,H=128,Thr=32,pad25 (≈25% holes).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402


def build_sp4(Eexp: int, H: int, threads: int):
    """Gather live rows: Out[p]=Buf[p] if Expert[p]>=0 else 0 (in-place mask).

    Keeps shape [Eexp,H] so gold is simple; pad slots contribute zeros —
    measures predicated gather / early-skip vs dense copy.
    """
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        Buf: T.Tensor((Eexp, H), "float32"),
        Expert: T.Tensor((Eexp,), "int32"),
        Out: T.Tensor((Eexp, H), "float32"),
    ):
        with T.Kernel(1):
            buf_ub = T.alloc_shared((Eexp, H), "float32")
            exp_ub = T.alloc_shared((Eexp,), "int32")
            out_ub = T.alloc_shared((Eexp, H), "float32")
            T.copy(Buf, buf_ub)
            T.copy(Expert, exp_ub)
            with vf_region(threads):
                for p, j in T.Parallel(Eexp, H):
                    if exp_ub[p] >= 0:
                        out_ub[p, j] = buf_ub[p, j]
                    else:
                        out_ub[p, j] = T.float32(0.0)
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    Eexp = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    pad_pct = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    tag = f"sp4_e{Eexp}_h{H}_t{threads}_pad{pad_pct}"
    so = compile_prim(build_sp4(Eexp, H, threads), tag, target="ascend")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
