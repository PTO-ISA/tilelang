#!/usr/bin/env python3
"""ST-SV3V — GEMV / thin-GEMM Acc KEEP / split (SimdVF / VMI twin of SV3).

Same math/IO as Simt SV3 (``sv3_gemv_partial_keep.py``):
  Acc[M, VL] += A[k, M, VL] * x[k]  for k in 0..K-1 -> Out[M, VL] fp32

Arms:
- ``keep``  — Acc[M*VL] live across the outer K (shared working set under SimdVF)
- ``split`` — Acc[CM*VL] live across K; one flush to out_ub per chunk

Tags: ``sv3v_m{M}_vl{VL}_k{K}_t{L}_{keep,split_cm{CM}}``.
Defaults: M=32, VL=64, K=16, lanes=64, CM=16.

VMI / SimdVF ABI notes (2026-10-05 REGRESSION-2 fix) — four separate blockers,
all ABI-level; the loop nest / arms / KEEP semantics are unchanged:

1. ``T.alloc_var`` for the ``xk`` reload became a ``local.var`` buffer:
   ``[VerifyParallelToPTO] memory access: buffer `xk` has scope `local.var`;
   the first version supports only UB (shared/shared.dyn) accesses``.
   -> the scalar is read inline from UB (``x_f32[k]``); its index does not
   depend on the vectorized Parallel var, so it lowers to a broadcast load.

2. Scalar ``T.Cast("float32", x_ub[k])`` hit
   ``Unsupported PTO cast: float16 -> float32`` (codegen_pto only has scalar
   f32<->bf16).  -> x is upcast **vectorially** once, in a lanes-wide
   ``T.Parallel`` (vload f16 + vcvt f32), into ``x_f32``; the K loop then
   broadcasts an f32 scalar.  ``x16`` is padded to ``lanes`` so that cast is
   lane-aligned; only the first K lanes are ever read back.

3. 2-D ``T.Parallel(M, VL)`` with a non-lane-aligned outer extent makes
   SimdVFExpandParallelDomain emit a domain mask, and PTODSL (ptoas_vmi
   0.1.8) has no ``pto.vmi.mask_and``:
   ``AttributeError: '_VMINamespace' object has no attribute 'mask_and'``.
   -> every Parallel here is 1-D over a lane-aligned extent
   (M*VL = 2048 / 1536, CM*VL = 1024, lanes = 64), i.e. the (i, j) iteration
   space is linearised, exactly the same element order the 2-D nest had.
   Buffers are declared flat for the same reason (identical GM/UB bytes).

4. The ``if row < M`` guard in the split arm made the Parallel body a
   predicated / non-continuous store.  -> guard dropped; ``M % CM == 0`` is
   now required (the oneshot shape m32/CM16 satisfies it).

Kept identical to Simt SV3: A and x are re-read from shared every K, Acc stays
live across the outer K (mapping mode 3), and the split arm flushes once per
chunk so barriers scale with chunk count, not K.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_pto_harness import boot, compile_prim, set_out  # noqa: E402


def build_sv3v(M: int, VL: int, K: int, lanes: int, arm: str, CM: int = 16):
    import tilelang.ascend.language as T

    MVL = M * VL
    if MVL % lanes != 0:
        raise ValueError(f"M*VL={MVL} must be a multiple of lanes={lanes} (no domain mask on PTODSL)")

    if arm == "split":
        if M % CM != 0:
            raise ValueError(f"split arm needs M % CM == 0 (no lane guard inside VF); got M={M} CM={CM}")
        NC = M // CM
        CMVL = CM * VL
        if CMVL % lanes != 0:
            raise ValueError(f"CM*VL={CMVL} must be a multiple of lanes={lanes}")

        @T.prim_func
        def main(
            A: T.Tensor((K, MVL), "float16"),
            X: T.Tensor((K,), "float16"),
            Out: T.Tensor((MVL,), "float32"),
        ):
            with T.Kernel(1):
                a_ub = T.alloc_shared((K, MVL), "float16")
                x16 = T.alloc_shared((lanes,), "float16")
                x_f32 = T.alloc_shared((lanes,), "float32")
                out_ub = T.alloc_shared((MVL,), "float32")
                T.copy(A, a_ub)
                T.copy(X, x16[0:K])
                with T.SimdVF(lanes=lanes):
                    # vector f16->f32 of x once (scalar f16->f32 has no PTO cast)
                    for t in T.Parallel(lanes):
                        x_f32[t] = T.Cast("float32", x16[t])
                    acc_c = T.alloc_shared((CMVL,), "float32")
                    for c in T.serial(NC):
                        for t in T.Parallel(CMVL):
                            acc_c[t] = T.float32(0.0)
                        for k in T.serial(K):
                            # A reload from shared each K; x[k] broadcast from
                            # UB each K; Acc[CM,VL] KEEP across K
                            for t in T.Parallel(CMVL):
                                acc_c[t] = acc_c[t] + (
                                    T.Cast("float32", a_ub[k, c * CMVL + t]) * x_f32[k]
                                )
                        # one flush per chunk (not per K)
                        for t in T.Parallel(CMVL):
                            out_ub[c * CMVL + t] = acc_c[t]
                T.copy(out_ub, Out)

        return main

    # arm == "keep"
    @T.prim_func
    def main(
        A: T.Tensor((K, MVL), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((MVL,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K, MVL), "float16")
            x16 = T.alloc_shared((lanes,), "float16")
            x_f32 = T.alloc_shared((lanes,), "float32")
            out_ub = T.alloc_shared((MVL,), "float32")
            T.copy(A, a_ub)
            T.copy(X, x16[0:K])
            with T.SimdVF(lanes=lanes):
                # vector f16->f32 of x once (scalar f16->f32 has no PTO cast)
                for t in T.Parallel(lanes):
                    x_f32[t] = T.Cast("float32", x16[t])
                acc = T.alloc_shared((MVL,), "float32")
                for t in T.Parallel(MVL):
                    acc[t] = T.float32(0.0)
                for k in T.serial(K):
                    # A reload from shared each K; x[k] broadcast from UB each
                    # K; Acc[M,VL] KEEP across K
                    for t in T.Parallel(MVL):
                        acc[t] = acc[t] + (
                            T.Cast("float32", a_ub[k, t]) * x_f32[k]
                        )
                for t in T.Parallel(MVL):
                    out_ub[t] = acc[t]
            T.copy(out_ub, Out)

    return main


def main():
    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    VL = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    lanes = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    arm = sys.argv[5] if len(sys.argv) > 5 else "keep"
    CM = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    if arm not in ("keep", "split"):
        print(f"unknown arm {arm!r}; use keep|split", flush=True)
        return 2
    if arm == "split":
        tag = f"sv3v_m{M}_vl{VL}_k{K}_t{lanes}_split_cm{CM}"
    else:
        tag = f"sv3v_m{M}_vl{VL}_k{K}_t{lanes}_keep"
    so = compile_prim(build_sv3v(M, VL, K, lanes, arm, CM), tag, target="pto")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
