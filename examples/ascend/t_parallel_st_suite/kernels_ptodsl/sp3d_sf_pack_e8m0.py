#!/usr/bin/env python3
"""ST-SP3d — PTO-DSL twin of SP3: A5 bit-reinterpret e8m0→fp compose.

Simt twin: kernels/sp3_sf_pack_ue8m0.py
Primary tags: sp3d_m32_h128_g32_t32_{fp32,e8m0}

**A5 path (pto-isa authoritative — Ascend950PR):**
NO dedicated ``vcvt f8e8m0→fp`` / TSCALE on A5 (A6 may). Real vector decode:

    vload ui8 → vcvt→ui32 → vshl(23) → vinterpret_cast→f32
    → vload(..., dist_mode="brc") + vmul

Evidence cite:
  ``pto-vmi/.../AntiMxQuantDequantKernel/..._case0_fp8_fp32_32_256.py`` (target="a5")
AscendC cousin: load fp8_e8m0 as uint8 → Interleave w/ zeros →
  ``MicroAPI::ShiftRights(..., 1)`` → bf16 ``2^(E−127)``.

**Forbidden:** soft Pow2[e] LUT table (SP6 soft 4-bit e2m1 only).
**Out of scope:** Cube MX TMATMUL_MX (scales stay float8_e8m0_t — no vector decode).

Alignment (2026-10-08 Lok): continuous unaligned → conceptually vldu/vstu
(`pto.vldas`/`pto.vldus`). Those return non-VMI vregs; e8m0 arm uses AntiMx
VL-aligned continuous `vmi.vload` of the packed ui8 SF (no gather).
"""
import argparse
import ctypes
import sys
import time

import numpy as np

M = 32
H = 128
G = 32
THR = 32
Hs = H // G  # 4
VL = 64
N_V_CHUNKS = H // VL  # 2

V_BYTES = M * H * 4
SF_F32_BYTES = M * Hs * 4
SF_U8_BYTES = M * Hs * 1
# Pad ui8 SF to VL-friendly scratch for vcvt widen (optional staging)
SF_U32_BYTES = M * Hs * 4
SF_F_BYTES = M * Hs * 4
OUT_BYTES = M * H * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _try_import_pto():
    try:
        from ptodsl import pto

        return pto
    except ImportError as e:
        return None


def build_fp32_kernel(pto):
    """fp32 SF sideband — dense group-bcast mul (compare baseline)."""

    OFF_SF = V_BYTES
    OFF_OUT = OFF_SF + SF_F32_BYTES

    @pto.jit(
        name="SP3d_m32_h128_g32_fp32",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
    )
    def kernel(
        out_gm: pto.ptr(pto.f32, "gm"),
        v_gm: pto.ptr(pto.f32, "gm"),
        sf_gm: pto.ptr(pto.f32, "gm"),
    ):
        v_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        sf_ub = pto.castptr(pto.const(OFF_SF, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(v_gm, v_ub, 0, V_BYTES, nburst=(1, V_BYTES, V_BYTES))
        pto.mte_load(sf_gm, sf_ub, 0, SF_F32_BYTES, nburst=(1, SF_F32_BYTES, SF_F32_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        # Per (m,g): brc one f32 scale across G elems (here VL chunks of V row)
        with pto.for_(0, M, step=1) as m:
            for g in range(Hs):
                # size=1 load + vbrc — cousin of dist_mode="brc"
                s1 = pto.vmi.vload(sf_ub, m * Hs + g, size=1)
                s_brc = pto.vmi.vbrc(s1, size=VL)
                base = m * H + g * G
                # G=32 < VL=64: one masked chunk covers the group
                mask_g = pto.vmi.create_mask(G, size=VL)
                v = pto.vmi.vload(v_ub, base, size=VL)
                o = pto.vmi.vmul(v, s_brc, mask_g)
                pto.vmi.vstore(o, out_ub, base, mask_g)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def build_e8m0_kernel(pto):
    """e8m0 arm — A5 bit-reinterpret compose (no soft Pow2 LUT, no native e8m0 vcvt).

    Exact VMI sequence (cite AntiMx case0_fp8_fp32 target=a5):
      vload ui8 → vcvt→ui32 → vshl(23) → vinterpret_cast→f32
      → vload(..., dist_mode="brc") + vmul

    Bindings may alias ``vshls``↔``vshl`` / ``vinterpret_cast``↔``vbitcast`` /
    ``vload(..., dist_mode="brc")``↔``vload size=1 + vbrc``. Adjust at compile
    time on pto-b10 if needed; do **not** fall back to a Pow2 LUT table.
    """

    OFF_SF = V_BYTES
    OFF_SF_F = OFF_SF + ((SF_U8_BYTES + 63) // 64) * 64
    OFF_OUT = OFF_SF_F + SF_F_BYTES

    @pto.jit(
        name="SP3d_m32_h128_g32_e8m0",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
    )
    def kernel(
        out_gm: pto.ptr(pto.f32, "gm"),
        v_gm: pto.ptr(pto.f32, "gm"),
        sf_gm: pto.ptr(pto.ui8, "gm"),
    ):
        v_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        sf_u8 = pto.castptr(pto.const(OFF_SF, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
        sf_f = pto.castptr(pto.const(OFF_SF_F, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(v_gm, v_ub, 0, V_BYTES, nburst=(1, V_BYTES, V_BYTES))
        pto.mte_load(sf_gm, sf_u8, 0, SF_U8_BYTES, nburst=(1, SF_U8_BYTES, SF_U8_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        # Continuous e8m0 decode — AntiMx VL-aligned bulk (no gather).
        # Prior per-row `vload(sf_u8, m*Hs, size=VL)` hit camodel RV_VLD at
        # OFF_SF+4 (Hs=4). VMI has no vmi.vldu; low-level pto.vldas/vldus return
        # !pto.vreg which does not compose with pto.vmi.vcvt. Keep continuous
        # via VL-aligned vmi.vload of the packed ui8 buffer (SCALE_N=M*Hs=128).
        mask_64 = pto.vmi.create_mask(VL, size=VL)
        SCALE_N = M * Hs
        n_chunks = (SCALE_N + VL - 1) // VL
        for i in range(n_chunks):
            off = i * VL
            e_u8 = pto.vmi.vload(sf_u8, off, size=VL)
            e_u32 = pto.vmi.vcvt(e_u8, pto.ui32)
            # vshl(23) — place exponent into IEEE f32 exponent field
            e_shl = pto.vmi.vshl(e_u32, 23, mask_64)
            scale_f = pto.vmi.vinterpret_cast(e_shl, pto.f32)
            pto.vmi.vstore(scale_f, sf_f, off, mask_64)

        with pto.for_(0, M, step=1) as m:
            for g in range(Hs):
                # broadcast one decoded scale across the group (brc: no 32B need)
                s_brc = pto.vmi.vload(sf_f, m * Hs + g, size=VL, dist_mode="brc")
                mask_g = pto.vmi.create_mask(G, size=VL)
                base = m * H + g * G
                v = pto.vmi.vload(v_ub, base, size=VL)
                o = pto.vmi.vmul(v, s_brc, mask_g)
                pto.vmi.vstore(o, out_ub, base, mask_g)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def e8m0_to_f32(e: np.ndarray) -> np.ndarray:
    """Host gold: bit-reinterpret compose (same as device; NOT a LUT table)."""
    bits = (e.astype(np.uint32) << np.uint32(23))
    return bits.view(np.float32)


def reference_fp32(V, Sf):
    scale = np.repeat(Sf, G, axis=1)
    return (V * scale).astype(np.float32)


def reference_e8m0(V, Sf_u8):
    Sf = e8m0_to_f32(Sf_u8)
    return reference_fp32(V, Sf)


def _load_acl():
    lib = ctypes.CDLL("libascendcl.so")
    for n, t in [
        ("aclInit", ctypes.c_char_p),
        ("aclFinalize", None),
        ("aclrtSetDevice", ctypes.c_int),
        ("aclrtResetDevice", ctypes.c_int),
    ]:
        getattr(lib, n).argtypes = [t] if t else []
        getattr(lib, n).restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def run_acl(arm: str, tag: str) -> int:
    pto = _try_import_pto()
    if pto is None:
        print(
            f"UNRUN {tag}: ptodsl not importable on this host. "
            "Kernel source is full *d-style (A5 bit-reinterpret). Run on pto-b10.",
            flush=True,
        )
        _print_sequence()
        return 1

    rng = np.random.default_rng(103)
    V = rng.standard_normal((M, H)).astype(np.float32)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    if arm == "fp32":
        Sf = rng.uniform(0.25, 4.0, size=(M, Hs)).astype(np.float32)
        ref = reference_fp32(V, Sf)
        kernel = build_fp32_kernel(pto)
        sf_bytes = SF_F32_BYTES
        sf_host = Sf
    else:
        # exponents in a safe pow2 range near 1.0 (bias 127)
        Sf_u8 = rng.integers(120, 130, size=(M, Hs), dtype=np.uint8)
        ref = reference_e8m0(V, Sf_u8)
        kernel = build_e8m0_kernel(pto)
        sf_bytes = SF_U8_BYTES
        sf_host = Sf_u8

    out = np.zeros((M, H), np.float32)
    vd = ctypes.c_void_p()
    sd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(vd), V_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mv")
    _chk(lib.aclrtMalloc(ctypes.byref(sd), sf_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    _chk(lib.aclrtMalloc(ctypes.byref(od), OUT_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            vd, V_BYTES, np.ascontiguousarray(V).ctypes.data_as(ctypes.c_void_p), V_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "h2d_v",
    )
    _chk(
        lib.aclrtMemcpy(
            sd,
            sf_bytes,
            np.ascontiguousarray(sf_host).ctypes.data_as(ctypes.c_void_p),
            sf_bytes,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_sf",
    )

    t0 = time.perf_counter()
    try:
        comp = kernel.compile()
    except Exception as e:
        print(f"UNRUN/COMPILE_FAIL {tag}: {e}", flush=True)
        _print_sequence()
        return 1
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(vd.value), int(sd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    _chk(
        lib.aclrtMemcpy(out.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(vd)
    lib.aclrtFree(sd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(out, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(out - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def _print_sequence():
    print(
        "A5 e8m0 VMI sequence (no soft Pow2 LUT; no native e8m0 vcvt):\n"
        "  vload ui8 → vcvt→ui32 → vshl(23) → vinterpret_cast→f32\n"
        "  → vload(..., dist_mode=\"brc\") + vmul\n"
        "Cite: pto-vmi AntiMxQuantDequantKernel case0_fp8_fp32_32_256.py target=a5",
        flush=True,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SP3d A5 bit-reinterpret e8m0 twin")
    ap.add_argument("--arm", choices=("fp32", "e8m0"), default="e8m0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("--sequence-only", action="store_true", help="print ISA sequence and exit UNRUN")
    args = ap.parse_args(argv)
    tag = args.tag or f"sp3d_m{M}_h{H}_g{G}_t{THR}_{args.arm}"
    if args.sequence_only:
        print(f"UNRUN {tag} (sequence-only)", flush=True)
        _print_sequence()
        return 1
    if args.emit_mlir:
        pto = _try_import_pto()
        if pto is None:
            print(f"UNRUN {tag}: no ptodsl for --emit-mlir", flush=True)
            _print_sequence()
            return 1
        k = build_e8m0_kernel(pto) if args.arm == "e8m0" else build_fp32_kernel(pto)
        print(k.compile().mlir_text())
        return 0
    return run_acl(args.arm, tag)


if __name__ == "__main__":
    raise SystemExit(main())
