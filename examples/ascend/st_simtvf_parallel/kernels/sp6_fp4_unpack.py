#!/usr/bin/env python3
"""ST-SP6 — Soft LUT 4-bit (e2m1) dequant (pack-decode appendix).

**Not** a Simt/Simd schedule peer of SP1–SP5. **Not** HW ``float4_e2m1x2_t`` /
``castFp4toBf16`` (pto-isa). Soft nibble extract + 16-entry e2m1 LUT only.

Alias name: soft LUT FP4 dequant (``sp6_lut_fp4_dequant``).

Arms:
- ``unpack``    — decode only (int8 byte → 2× LUT[nibble] → fp32)
- ``unpack_sf`` — decode × Sf[n, j//G] group bcast

Single sensitivity: value-pack density vs soft LUT ALU (± SF mul tax).

Workload remark: TileKernels ``unpack_from_e2m1fn_x2`` / per_token_cast e2m1.
Orthogonal to SP3 (HW e8m0 SF pack). Soft Pow2 UE8M0 LUT was retired from
SP3 — do not reintroduce scale LUT here either; SF is fp32 sideband.

Tags: ``sp6_n{N}_h{H}_g{G}_t{Thr}_{unpack,unpack_sf}``.
Primary: N=32,H=128,G=32,Thr=32.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402

# Soft e2m1 LUT (nibble 0..15) — matches TileKernels unpack_from_e2m1fn_x2.
# This is intentionally software; HW FP4 = separate future ST.
_E2M1 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def build_sp6(N: int, H: int, G: int, threads: int, arm: str):
    import tilelang.ascend.language as T

    Hs = H // G
    Hbytes = H // 2

    # Lut[16] as kernel arg (Ascend-friendly; avoid nested defs in prim_func).
    if arm == "unpack_sf":

        @T.prim_func
        def main(
            V_i8: T.Tensor((N, Hbytes), "int8"),
            Sf: T.Tensor((N, Hs), "float32"),
            Lut: T.Tensor((16,), "float32"),
            Out: T.Tensor((N, H), "float32"),
        ):
            with T.Kernel(1):
                v_ub = T.alloc_shared((N, Hbytes), "int8")
                sf_ub = T.alloc_shared((N, Hs), "float32")
                lut_ub = T.alloc_shared((16,), "float32")
                out_ub = T.alloc_shared((N, H), "float32")
                T.copy(V_i8, v_ub)
                T.copy(Sf, sf_ub)
                T.copy(Lut, lut_ub)
                with T.SimtVF(threads=threads):
                    for n, b in T.Parallel(N, Hbytes):
                        byte = T.Cast("int32", v_ub[n, b])
                        lo = byte & 0x0F
                        hi = (byte >> 4) & 0x0F
                        out_ub[n, b * 2] = lut_ub[lo]
                        out_ub[n, b * 2 + 1] = lut_ub[hi]
                    for n, g in T.Parallel(N, Hs):
                        s = sf_ub[n, g]
                        for tt in T.serial(G):
                            j = g * G + tt
                            out_ub[n, j] = out_ub[n, j] * s
                T.copy(out_ub, Out)

        return main

    @T.prim_func
    def main(
        V_i8: T.Tensor((N, Hbytes), "int8"),
        Lut: T.Tensor((16,), "float32"),
        Out: T.Tensor((N, H), "float32"),
    ):
        with T.Kernel(1):
            v_ub = T.alloc_shared((N, Hbytes), "int8")
            lut_ub = T.alloc_shared((16,), "float32")
            out_ub = T.alloc_shared((N, H), "float32")
            T.copy(V_i8, v_ub)
            T.copy(Lut, lut_ub)
            with T.SimtVF(threads=threads):
                for n, b in T.Parallel(N, Hbytes):
                    byte = T.Cast("int32", v_ub[n, b])
                    lo = byte & 0x0F
                    hi = (byte >> 4) & 0x0F
                    out_ub[n, b * 2] = lut_ub[lo]
                    out_ub[n, b * 2 + 1] = lut_ub[hi]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/st_simtvf_parallel"))
    boot()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    arm = sys.argv[5] if len(sys.argv) > 5 else "unpack"
    if arm not in ("unpack", "unpack_sf"):
        print(f"unknown arm {arm!r}; use unpack|unpack_sf", flush=True)
        return 2
    tag = f"sp6_n{N}_h{H}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sp6(N, H, G, threads, arm), tag, target="ascend")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
