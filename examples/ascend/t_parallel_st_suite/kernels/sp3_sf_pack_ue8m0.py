#!/usr/bin/env python3
"""ST-SP3 — fp32 SF vs A5 bit-reinterpret e8m0→fp compose (dense).

Single sensitivity: fp32 SF sideband load vs packed e8m0 load + **A5
bit-reinterpret compose** (NOT a soft Pow2 LUT table; NOT native e8m0 vcvt).

Same gold: ``Out[m,j] = V[m,j] * Sf[m, j//G]`` (group bcast).

Arms:
- ``fp32``  — Sf[M,Hs] float32 sideband (baseline; historical PASS)
- ``e8m0``  — Sf[M,Hs] uint8 e8m0 exponents; decode via bit compose
              ``ui8 → ui32 → <<23 → reinterpret f32`` then mul.
              Tag alias: ``ue8m0`` → same arm.

**A5 ISA (pto-isa authoritative, Ascend950PR):**
- There is **NO** dedicated ``vcvt f8e8m0→fp`` / TSCALE on A5 (A6 may add that vcvt).
- Real A5 vector path = bit-reinterpret compose (VMI twin preferred):
    ``vload ui8 → vcvt→ui32 → vshls(23) → vinterpret_cast→f32
     → vload(..., dist_mode="brc") + vmul``
  Evidence: ``pto-vmi/.../AntiMxQuantDequantKernel/..._case0_fp8_fp32_32_256.py``
  (``target="a5"``). AscendC cousin: load as uint8 → Interleave zeros →
  ``MicroAPI::ShiftRights(..., 1)`` → bf16 ``2^(E−127)``.
- Cube MX (TMATMUL_MX keeps ``float8_e8m0_t``) = **OUT OF SCOPE** for SP3.

**Forbidden:** soft ``Pow2[e]`` LUT table (that stays **SP6** soft 4-bit e2m1 only).

**SimtVF note:** scalar bit compose via ``T.Cast`` + ``<< 23`` + ``T.reinterpret``
(no LUT). Special VMI mnemonics (``vshls`` / ``vinterpret_cast`` / ``dist_mode=brc``)
live on the *d twin. Status until opsim: may be COMPILE_FAIL / UNRUN if Ascend
lowering rejects reinterpret inside SimtVF — still prefer that over a LUT.

**PTO-DSL twin:** ``kernels_ptodsl/sp3d_sf_pack_e8m0.py`` (VMI bit-shift compose).

Workload remark: MX / UE8M0 scale path (pack/ISA micro, not KEEP/remat schedule).

Tags: ``sp3_m{M}_h{H}_g{G}_t{Thr}_{fp32,e8m0}``.
Primary: M=32,H=128,G=32,Thr=32.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import compile_target, vf_region  # noqa: E402

# Soft Pow2 LUT path REMOVED. Do not reintroduce.


def build_sp3(M: int, H: int, G: int, threads: int, arm: str):
    """Build SP3 prim_func. e8m0 = A5 bit-reinterpret compose (no LUT)."""
    import tilelang.ascend.language as T

    Hs = H // G

    if arm in ("e8m0", "ue8m0"):
        # A5 bit-reinterpret: e8m0 byte → (e<<23) bits → f32 power-of-two scale.
        # No Pow2[e] table. No native f8e8m0 vcvt (A5).

        @T.prim_func
        def main(
            V: T.Tensor((M, H), "float32"),
            Sf_u8: T.Tensor((M, Hs), "uint8"),
            Out: T.Tensor((M, H), "float32"),
        ):
            with T.Kernel(1):
                v_ub = T.alloc_shared((M, H), "float32")
                sf_ub = T.alloc_shared((M, Hs), "uint8")
                out_ub = T.alloc_shared((M, H), "float32")
                T.copy(V, v_ub)
                T.copy(Sf_u8, sf_ub)
                with vf_region(threads):
                    for m, g in T.Parallel(M, Hs):
                        e = T.Cast("uint32", sf_ub[m, g])
                        bits = e << 23
                        # IEEE f32 with exponent field = e  ⇒  scale = 2^(e-127)
                        s = T.reinterpret(bits, "float32")
                        for tt in T.serial(G):
                            j = g * G + tt
                            out_ub[m, j] = v_ub[m, j] * s
                T.copy(out_ub, Out)

        return main

    @T.prim_func
    def main(
        V: T.Tensor((M, H), "float32"),
        Sf: T.Tensor((M, Hs), "float32"),
        Out: T.Tensor((M, H), "float32"),
    ):
        with T.Kernel(1):
            v_ub = T.alloc_shared((M, H), "float32")
            sf_ub = T.alloc_shared((M, Hs), "float32")
            out_ub = T.alloc_shared((M, H), "float32")
            T.copy(V, v_ub)
            T.copy(Sf, sf_ub)
            with vf_region(threads):
                for m, g in T.Parallel(M, Hs):
                    s = sf_ub[m, g]
                    for tt in T.serial(G):
                        j = g * G + tt
                        out_ub[m, j] = v_ub[m, j] * s
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "fp32"
    if arm == "ue8m0":
        arm = "e8m0"
    if arm not in ("fp32", "e8m0"):
        print(f"unknown arm {arm!r}; use fp32|e8m0 (ue8m0 alias→e8m0)", flush=True)
        return 2
    tag = f"sp3_m{M}_h{H}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sp3(M, H, G, threads, arm), tag, target=compile_target())
    print("SO", so) if so else None
    if arm == "e8m0" and not so:
        print(
            f"UNRUN/COMPILE_FAIL {tag}: A5 bit-reinterpret e8m0→fp (no LUT, no native "
            "e8m0 vcvt). Prefer *d twin VMI: vload ui8→vcvt ui32→vshls(23)→"
            "vinterpret_cast f32→vload brc+vmul. Soft Pow2 LUT forbidden.",
            flush=True,
        )
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
