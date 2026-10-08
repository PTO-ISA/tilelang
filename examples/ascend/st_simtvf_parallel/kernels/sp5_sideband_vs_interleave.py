#!/usr/bin/env python3
"""ST-SP5 — Sideband (2 indexed loads) vs interleave (1 slot).

**Single sensitivity:** RF/MTE schedule when gather uses **two** tensors
under one index vs **one** interleaved slot then split. Same gold.

**Workload remark:** SFA / KV-ish scale+value layout; TileKernels sideband
``QuantTensor=(V,Sf)``.

Arms (``Out[q,:] = V[slot,:] * Sf[slot]`` group-bcast):
- ``sideband``   — gather V[slot] and Sf[slot] as two tensors
- ``interleave`` — gather Pack[slot] = concat(V row, Sf row); split then mul

Tags: ``sp5_n{N}_h{H}_g{G}_qg{Qg}_t{Thr}_{sideband,interleave}``.
Primary: N=64,H=128,G=32,Qg=32,Thr=32.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sp5(N: int, H: int, G: int, Qg: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    Hs = H // G
    PackW = H + Hs  # flat interleave: [N, H+Hs]

    if arm == "interleave":

        @T.prim_func
        def main(
            Pack: T.Tensor((N, PackW), "float32"),
            Idx: T.Tensor((Qg,), "int32"),
            Out: T.Tensor((Qg, H), "float32"),
        ):
            with T.Kernel(1):
                pack_ub = T.alloc_shared((N, PackW), "float32")
                idx_ub = T.alloc_shared((Qg,), "int32")
                out_ub = T.alloc_shared((Qg, H), "float32")
                T.copy(Pack, pack_ub)
                T.copy(Idx, idx_ub)
                with T.SimtVF(threads=threads):
                    for q in T.serial(Qg):
                        slot = idx_ub[q]
                        # one indexed load stream; split v + sf from Pack[slot]
                        for g in T.Parallel(Hs):
                            s = pack_ub[slot, H + g]
                            for tt in T.serial(G):
                                j = g * G + tt
                                out_ub[q, j] = pack_ub[slot, j] * s
                T.copy(out_ub, Out)

        return main

    # arm == "sideband"
    @T.prim_func
    def main(
        V: T.Tensor((N, H), "float32"),
        Sf: T.Tensor((N, Hs), "float32"),
        Idx: T.Tensor((Qg,), "int32"),
        Out: T.Tensor((Qg, H), "float32"),
    ):
        with T.Kernel(1):
            v_ub = T.alloc_shared((N, H), "float32")
            sf_ub = T.alloc_shared((N, Hs), "float32")
            idx_ub = T.alloc_shared((Qg,), "int32")
            out_ub = T.alloc_shared((Qg, H), "float32")
            T.copy(V, v_ub)
            T.copy(Sf, sf_ub)
            T.copy(Idx, idx_ub)
            with T.SimtVF(threads=threads):
                for q in T.serial(Qg):
                    slot = idx_ub[q]
                    for g in T.Parallel(Hs):
                        s = sf_ub[slot, g]
                        for tt in T.serial(G):
                            j = g * G + tt
                            out_ub[q, j] = v_ub[slot, j] * s
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    Qg = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    arm = sys.argv[6] if len(sys.argv) > 6 else "sideband"
    if arm not in ("sideband", "interleave"):
        print(f"unknown arm {arm!r}; use sideband|interleave", flush=True)
        return 2
    tag = f"sp5_n{N}_h{H}_g{G}_qg{Qg}_t{threads}_{arm}"
    so = compile_prim(build_sp5(N, H, G, Qg, threads, arm), tag, target="ascend")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
