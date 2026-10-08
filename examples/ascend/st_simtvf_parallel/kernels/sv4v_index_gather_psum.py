#!/usr/bin/env python3
"""ST-SV4V — Index gather + partial-sum across batch B (SimdVF / VMI twin of SV4).

Replaced keep2: former twin used identity idxs[i]=i and Out[B,E]; active twin
matches Simt SV4 gather_psum — non-identity Idx[E], Out[E] partial sum.

Same math/IO as Simt SV4:
  Out[i] = sum_{b=0..B-1} A[b,i] * W[Idx[i]]

Arms:
- ``keep_idx``  — load Idx once into live idxs; KEEP idxs + KEEP acc across B
- ``remat_idx`` — NO live idxs across B; remat idx_i from shared Idx each gather;
                  KEEP acc across B

Tags: ``sv4v_e{E}_b{B}_t{L}_{keep_idx,remat_idx}``.
Defaults: E=256, B=8, lanes=64.

VMI notes: prefer ``alloc_shared`` for idxs/acc working set under SimdVF.
Compile: target=pto via common_pto_harness.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv4v(E: int, B: int, lanes: int, arm: str):
    import tilelang.ascend.language as T

    if arm == "remat_idx":
        @T.prim_func
        def main(
            A: T.Tensor((B, E), "float32"),
            W: T.Tensor((E,), "float32"),
            Idx: T.Tensor((E,), "int32"),
            Out: T.Tensor((E,), "float32"),
        ):
            with T.Kernel(1):
                a_ub = T.alloc_shared((B, E), "float32")
                w_ub = T.alloc_shared((E,), "float32")
                idx_ub = T.alloc_shared((E,), "int32")
                out_ub = T.alloc_shared((E,), "float32")
                T.copy(A, a_ub)
                T.copy(W, w_ub)
                T.copy(Idx, idx_ub)
                with T.SimdVF(lanes=lanes):
                    # NO live idxs buffer across B — remat from shared each gather
                    # SimdVF-green: acc in shared (Simt twin used fragment)
                    acc = T.alloc_shared((E,), "float32")
                    for i in T.Parallel(E):
                        acc[i] = T.float32(0.0)
                    for b in T.serial(B):
                        for i in T.Parallel(E):
                            idx_i = idx_ub[i]  # remat index (VCI) from shared Idx
                            acc[i] = acc[i] + a_ub[b, i] * w_ub[idx_i]
                    for i in T.Parallel(E):
                        out_ub[i] = acc[i]
                T.copy(out_ub, Out)

        return main

    # arm == "keep_idx"
    @T.prim_func
    def main(
        A: T.Tensor((B, E), "float32"),
        W: T.Tensor((E,), "float32"),
        Idx: T.Tensor((E,), "int32"),
        Out: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((B, E), "float32")
            w_ub = T.alloc_shared((E,), "float32")
            idx_ub = T.alloc_shared((E,), "int32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(W, w_ub)
            T.copy(Idx, idx_ub)
            with T.SimdVF(lanes=lanes):
                # SimdVF-green: idxs/acc in shared (Simt twin used fragment)
                idxs = T.alloc_shared((E,), "int32")
                acc = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    idxs[i] = idx_ub[i]  # load Idx once into live working set
                for i in T.Parallel(E):
                    acc[i] = T.float32(0.0)
                for b in T.serial(B):
                    for i in T.Parallel(E):
                        # KEEP idxs + KEEP acc across outer B
                        acc[i] = acc[i] + a_ub[b, i] * w_ub[idxs[i]]
                for i in T.Parallel(E):
                    out_ub[i] = acc[i]
            T.copy(out_ub, Out)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lanes = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    arm = sys.argv[4] if len(sys.argv) > 4 else "keep_idx"
    if arm not in ("keep_idx", "remat_idx"):
        print(f"unknown arm {arm!r}; use keep_idx|remat_idx", flush=True)
        return 2
    tag = f"sv4v_e{E}_b{B}_t{lanes}_{arm}"
    so = compile_prim(build_sv4v(E, B, lanes, arm), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
