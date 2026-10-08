#!/usr/bin/env python3
"""ST-SV3 (former SV2G) — GEMV / thin-GEMM loop-carried fp32 partial-sum RF stress (SV3).

Replaces topk CF as the SV3 sensitivity story. Legacy topk micros archived under ``kernels/_removed_legacy/``; e2e is SV9.

Math (batched strip / thin GEMM):
  Acc[M, VL] += A[k, M, VL] * x[k]   (broadcast x[k]) for k in 0..K-1
  then write Acc → Out[M, VL].

Sensitivity (Lok):
- Loop-carried **fp32 partial sum** stays in fragment RF across outer K.
- Other inputs (A tile / x) **reload** each K from shared (not KEEP).
- Tile M×VL with VL=64; try M=24 and M=32 near the SIMD ~32 VL-reg cliff.
- When Acc too large: **split** chunk M (CM), flush to shared once per chunk
  (expected membar count O(num_chunks), not O(K)).

Arms (separate Python ``@T.prim_func`` bodies):
- ``keep``  — Acc[M, VL] live across K
- ``split`` — Acc[CM, VL] live across K; outer chunk loop flushes to out_ub

SV6/SV1B lessons:
- Mutable accum: ``acc[i,j] = acc[i,j] + ...`` on fragment (OK)
- Prefer fp32 Out (avoid half uint2 packs)
- ``target=ascend`` + cython via ``compile_prim``
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv3(M: int, VL: int, K: int, threads: int, arm: str, CM: int = 16):
    import tilelang.ascend.language as T

    if arm == "split":
        NC = (M + CM - 1) // CM  # chunk count; prefer M % CM == 0

        @T.prim_func
        def main(
            A: T.Tensor((K, M, VL), "float16"),
            X: T.Tensor((K,), "float16"),
            Out: T.Tensor((M, VL), "float32"),
        ):
            with T.Kernel(1):
                a_ub = T.alloc_shared((K, M, VL), "float16")
                x_ub = T.alloc_shared((K,), "float16")
                out_ub = T.alloc_shared((M, VL), "float32")
                T.copy(A, a_ub)
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    # Partial-sum tile fits in CM×VL (folded RF); reused per chunk.
                    # Flush to shared once per chunk → expected MTE bar ≈ O(NC), not O(K).
                    acc_c = T.alloc_fragment((CM, VL), "float32")
                    for c in T.serial(NC):
                        for i, j in T.Parallel(CM, VL):
                            acc_c[i, j] = T.float32(0.0)
                        for k in T.serial(K):
                            # x reload each K (scalar into alloc_var)
                            xk = T.alloc_var("float32")
                            xk = T.Cast("float32", x_ub[k])
                            # A reload from shared each K; Acc KEEP across K
                            for i, j in T.Parallel(CM, VL):
                                row = c * CM + i
                                # Guard for non-multiple M (oneshot uses M%CM==0)
                                if row < M:
                                    acc_c[i, j] = acc_c[i, j] + (
                                        T.Cast("float32", a_ub[k, row, j]) * xk
                                    )
                        # One flush per chunk (not per K)
                        for i, j in T.Parallel(CM, VL):
                            row = c * CM + i
                            if row < M:
                                out_ub[row, j] = acc_c[i, j]
                T.copy(out_ub, Out)

        return main

    # arm == "keep" (default): full Acc[M, VL] live across outer K
    @T.prim_func
    def main(
        A: T.Tensor((K, M, VL), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((M, VL), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K, M, VL), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((M, VL), "float32")
            T.copy(A, a_ub)
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                acc = T.alloc_fragment((M, VL), "float32")
                for i, j in T.Parallel(M, VL):
                    acc[i, j] = T.float32(0.0)
                for k in T.serial(K):
                    # Reload x each K; Acc KEEP in fragment RF
                    xk = T.alloc_var("float32")
                    xk = T.Cast("float32", x_ub[k])
                    # Reload A tile from shared each K (not kept across K)
                    for i, j in T.Parallel(M, VL):
                        acc[i, j] = acc[i, j] + (
                            T.Cast("float32", a_ub[k, i, j]) * xk
                        )
                for i, j in T.Parallel(M, VL):
                    out_ub[i, j] = acc[i, j]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    VL = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "keep"
    CM = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    if arm not in ("keep", "split"):
        print(f"unknown arm {arm!r}; use keep|split", flush=True)
        return 2
    if arm == "split":
        tag = f"sv3_m{M}_vl{VL}_k{K}_t{threads}_split_cm{CM}"
    else:
        tag = f"sv3_m{M}_vl{VL}_k{K}_t{threads}_keep"
    so = compile_prim(build_sv3(M, VL, K, threads, arm, CM), tag)
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
