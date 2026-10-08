#!/usr/bin/env python3
"""ST-SV4 — Index gather + partial-sum across batch B (keep_idx / remat_idx).

Replaced keep2: former ST was identity idxs[i]=i with Out[B,E]=A*W[idxs];
active ST is non-identity VCI table Idx[E] with Out[E] = sum_b A[b,i]*W[Idx[i]].

GPU analogy: keep a long VCI index live across batch while a large register
keep-alive (partial sum across B) already occupies RF — or remat the index
each gather while still KEEP-ing the accumulator.

Arms:
- ``keep_idx``       — load Idx once into live idxs; KEEP idxs + KEEP acc across B
- ``remat_idx``      — NO live idxs across B; remat idx_i from shared Idx each gather;
                       KEEP acc across B (mode 1 on index, mode 3 on acc).
                       **Blocked on SimtVF/ascend today** — see note below.
- ``remat_idx_calc`` — NO live idxs across B and no shared re-read: recompute the
                       VCI arithmetically ((i*7+3) % E, the suite's fixed table)
                       at each gather; KEEP acc across B. Same gold as the other
                       two arms; this is the green remat arm on SimtVF.

Remat-from-shared (``remat_idx``) blocker (2026-10-05, deps-native libtilelang):
  per-lane ``w_ub[idx_ub[i]]`` inside ``T.Parallel`` makes the gather address a
  vector lane value. The SimtVF path then fails in one of two places:
    * stock:                 TileLangThreadSync → arith::EvalSet ICHECK(eval_vec_)
    * tl.disable_thread_storage_sync: codegen "Cannot convert type int64x2
      (lanes=2) to Ascend type"
  (tl.disable_vectorize_256 does not defuse it; tl.config_index_bitwidth=32 trades
  it for "variables (idx_i,) ... not passed in as API arguments").
  keep_idx is unaffected because its index comes from a per-thread fragment.

No topk CF (that stays SV9). No AABBCC / token_tile / vf_fuse.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

ARMS = ("keep_idx", "remat_idx", "remat_idx_calc")


def build_sv4(E: int, B: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    if arm == "remat_idx_calc":
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
                T.copy(Idx, idx_ub)  # same MTE traffic as the other arms
                with T.SimtVF(threads=threads):
                    # NO live idxs buffer across B — recompute the VCI each gather
                    acc = T.alloc_fragment((E,), "float32")
                    for i in T.Parallel(E):
                        acc[i] = T.float32(0.0)
                    for b in T.serial(B):
                        for i in T.Parallel(E):
                            # (i*7+3) % E is the suite's fixed Idx table
                            acc[i] = acc[i] + a_ub[b, i] * w_ub[(i * 7 + 3) % E]
                    for i in T.Parallel(E):
                        out_ub[i] = acc[i]
                T.copy(out_ub, Out)

        return main

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
                with T.SimtVF(threads=threads):
                    # NO live idxs buffer across B — remat from shared each gather
                    acc = T.alloc_fragment((E,), "float32")
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

    # arm == "keep_idx": live idxs + live acc across serial(B)
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
            with T.SimtVF(threads=threads):
                idxs = T.alloc_fragment((E,), "int32")
                acc = T.alloc_fragment((E,), "float32")
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
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    arm = sys.argv[4] if len(sys.argv) > 4 else "keep_idx"
    if arm not in ARMS:
        print(f"unknown arm {arm!r}; use {'|'.join(ARMS)}", flush=True)
        return 2
    tag = f"sv4_e{E}_b{B}_t{threads}_{arm}"
    so = compile_prim(build_sv4(E, B, threads, arm), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
