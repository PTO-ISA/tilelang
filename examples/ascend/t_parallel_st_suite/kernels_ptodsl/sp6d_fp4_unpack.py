#!/usr/bin/env python3
"""ST-SP6d — SIMD (PTO-DSL) soft 4-bit e2m1 LUT dequant (pack-decode appendix).

Simt twin: kernels/sp6_fp4_unpack.py
Primary tags:
  sp6d_n32_h128_g32_t32_{unpack|unpack_sf}_{gather|vselr}

**Not** a Simt/Simd schedule peer of SP1–SP5. Soft nibble + 16-entry e2m1
LUT only — **not** HW ``float4_e2m1x2_t`` / ``castFp4toBf16``.

Sensitivity (schedule × decode):
  1. lut=gather — nibble indices → ``pto.vmi.vgather(lut_ub, idx, mask)``
  2. lut=vselr  — register-table select by nibble (if ``pto.vmi.vselr`` binds)

Arms matching Simt: ``unpack`` / ``unpack_sf`` (optional Sf[n,j//G] mul).

Alignment: packed ui8 dense N×Hbytes is VL-aligned continuous OK; Lut[16]
padded to 64 B; Hs-stride SF uses size=1 + vbrc (SP3d style).
"""
import argparse
import ctypes
import time
from typing import Optional

import numpy as np

N = 32
H = 128
G = 32
THR = 32
Hs = H // G  # 4
Hbytes = H // 2  # 64
VL = 64

V_BYTES = N * Hbytes  # ui8
# Lut[16] live values; pad UB to full VL f32 (256 B) so vselr source lane-count
# matches index VL=64 (vselr requires index lanes == result/source lanes).
LUT_BYTES = VL * 4  # 256
SF_BYTES = N * Hs * 4
OUT_BYTES = N * H * 4

_E2M1 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _try_import_pto():
    try:
        from ptodsl import pto
        from ptodsl import scalar  # noqa: F401

        return pto
    except ImportError:
        return None


def _has_vselr(pto) -> bool:
    return hasattr(pto.vmi, "vselr")


def _layout(with_sf: bool):
    off_v = 0
    off_lut = off_v + V_BYTES
    off_sf = off_lut + LUT_BYTES
    off_out = off_sf + (SF_BYTES if with_sf else 0)
    return dict(off_v=off_v, off_lut=off_lut, off_sf=off_sf, off_out=off_out)


def build_kernel(pto, with_sf: bool, lut_mode: str):
    """lut_mode: 'gather' | 'vselr'."""
    from ptodsl import scalar

    L = _layout(with_sf)
    arm = "unpack_sf" if with_sf else "unpack"
    name = f"SP6d_n{N}_h{H}_g{G}_{arm}_{lut_mode}"

    @pto.jit(
        name=name,
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(
        out_gm: pto.ptr(pto.f32, "gm"),
        v_gm: pto.ptr(pto.ui8, "gm"),
        lut_gm: pto.ptr(pto.f32, "gm"),
        sf_gm: pto.ptr(pto.f32, "gm"),
    ):
        v_ub = pto.castptr(pto.const(L["off_v"], dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
        lut_ub = pto.castptr(pto.const(L["off_lut"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(L["off_out"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        sf_ub = pto.castptr(pto.const(L["off_sf"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(v_gm, v_ub, 0, V_BYTES, nburst=(1, V_BYTES, V_BYTES))
        pto.mte_load(lut_gm, lut_ub, 0, LUT_BYTES, nburst=(1, LUT_BYTES, LUT_BYTES))
        if with_sf:
            pto.mte_load(sf_gm, sf_ub, 0, SF_BYTES, nburst=(1, SF_BYTES, SF_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)
        two = pto.vmi.vbrc(pto.i32(2), size=VL)
        one_i = pto.vmi.vbrc(pto.i32(1), size=VL)
        even = pto.vmi.vmul(arange, two, mask)
        odd = pto.vmi.vadd(even, one_i, mask)
        mask15 = pto.vmi.vbrc(pto.i32(0x0F), size=VL)

        # vselr: preload Lut[0:16] into VRF once (sig: vselr(source, index) — no mask)
        if lut_mode == "vselr":
            # Full-VL table: lanes 0..15 = e2m1 Lut; rest padded zeros (indices only use 0..15)
            table = pto.vmi.vload(lut_ub, 0, size=VL)

        with pto.for_(0, N, step=1) as n:
            # One VL of packed ui8 = full row (Hbytes=64)
            # ui8→i32 widen (vcvt same-width ui32↔i32 is illegal); nibble in i32
            raw_u8 = pto.vmi.vload(v_ub, n * Hbytes, size=VL)
            raw_i = pto.vmi.vcvt(raw_u8, pto.i32)
            lo_i = pto.vmi.vand(raw_i, mask15, mask)
            hi_i = pto.vmi.vand(pto.vmi.vshr(raw_i, 4, mask), mask15, mask)

            if lut_mode == "gather":
                lo_f = pto.vmi.vgather(lut_ub, lo_i, mask)
                hi_f = pto.vmi.vgather(lut_ub, hi_i, mask)
            else:
                lo_f = pto.vmi.vselr(table, lo_i)
                hi_f = pto.vmi.vselr(table, hi_i)

            # Interleave into Out[n, 0:128]: even=lo, odd=hi
            n_i32 = scalar.index_cast(pto.i32, n)
            base = pto.vmi.vbrc(n_i32 * H, size=VL)
            outs_lo = pto.vmi.vadd(base, even, mask)
            outs_hi = pto.vmi.vadd(base, odd, mask)
            pto.vmi.vscatter(lo_f, out_ub, outs_lo, mask)
            pto.vmi.vscatter(hi_f, out_ub, outs_hi, mask)

            if with_sf:
                for g in range(Hs):
                    s1 = pto.vmi.vload(sf_ub, n * Hs + g, size=1)
                    s_brc = pto.vmi.vbrc(s1, size=VL)
                    mask_g = pto.vmi.create_mask(G, size=VL)
                    base_g = n * H + g * G
                    v = pto.vmi.vload(out_ub, base_g, size=VL)
                    o = pto.vmi.vmul(v, s_brc, mask_g)
                    pto.vmi.vstore(o, out_ub, base_g, mask_g)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def reference(V_u8, Lut, Sf):
    """Host gold — same e2m1 table as Simt."""
    out = np.zeros((N, H), dtype=np.float32)
    for n in range(N):
        for b in range(Hbytes):
            byte = int(V_u8[n, b]) & 0xFF
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            out[n, b * 2] = Lut[lo]
            out[n, b * 2 + 1] = Lut[hi]
        if Sf is not None:
            for g in range(Hs):
                s = Sf[n, g]
                out[n, g * G : (g + 1) * G] *= s
    return out


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
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def run_acl(with_sf: bool, lut_mode: str, tag: str) -> int:
    pto = _try_import_pto()
    if pto is None:
        print(
            f"UNRUN {tag}: ptodsl not importable on this host. Run on pto-b10.",
            flush=True,
        )
        return 1
    if lut_mode == "vselr" and not _has_vselr(pto):
        print(
            f"UNRUN {tag}: pto.vmi.vselr not in binding (dir(pto.vmi) has no vselr).",
            flush=True,
        )
        return 1

    rng = np.random.default_rng(16)
    V_u8 = rng.integers(0, 256, size=(N, Hbytes), dtype=np.uint8)
    Lut = np.array(_E2M1, dtype=np.float32)
    # Pad Lut host buffer to LUT_BYTES
    Lut_pad = np.zeros(LUT_BYTES // 4, dtype=np.float32)
    Lut_pad[:16] = Lut
    Sf = rng.uniform(0.25, 4.0, size=(N, Hs)).astype(np.float32) if with_sf else None
    ref = reference(V_u8, Lut, Sf)

    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    def malloc(n):
        p = ctypes.c_void_p()
        _chk(lib.aclrtMalloc(ctypes.byref(p), n, _ACL_MEM_MALLOC_HUGE_FIRST), "malloc")
        return p

    vd = malloc(V_BYTES)
    ld = malloc(LUT_BYTES)
    od = malloc(OUT_BYTES)
    sd = malloc(SF_BYTES) if with_sf else malloc(4)  # dummy when no sf

    def h2d(dst, arr, n):
        _chk(
            lib.aclrtMemcpy(
                dst,
                n,
                np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p),
                n,
                _ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "h2d",
        )

    h2d(vd, V_u8, V_BYTES)
    h2d(ld, Lut_pad, LUT_BYTES)
    if with_sf:
        h2d(sd, Sf, SF_BYTES)
    else:
        h2d(sd, np.zeros(1, np.float32), 4)

    try:
        kernel = build_kernel(pto, with_sf, lut_mode)
    except Exception as e:
        print(f"UNRUN/COMPILE_FAIL {tag}: build {e}", flush=True)
        return 1

    t0 = time.perf_counter()
    try:
        comp = kernel.compile()
    except Exception as e:
        print(f"UNRUN/COMPILE_FAIL {tag}: {e}", flush=True)
        return 1
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(vd.value), int(ld.value), int(sd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    out = np.zeros((N, H), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(
            out.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "d2h",
    )
    for p in (vd, ld, od, sd):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(out, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={float(np.max(np.abs(out - ref)))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SP6d soft e2m1 LUT (SIMD PTO-DSL)")
    ap.add_argument("N", nargs="?", type=int, default=N)
    ap.add_argument("H", nargs="?", type=int, default=H)
    ap.add_argument("G", nargs="?", type=int, default=G)
    ap.add_argument("Thr", nargs="?", type=int, default=THR)
    ap.add_argument(
        "arm_pos",
        nargs="?",
        default=None,
        choices=("unpack", "unpack_sf", "gather", "vselr"),
    )
    ap.add_argument(
        "lut_pos",
        nargs="?",
        default=None,
        choices=("gather", "vselr", "unpack", "unpack_sf"),
    )
    ap.add_argument("--arm", default="unpack", choices=("unpack", "unpack_sf"))
    ap.add_argument("--lut", default="gather", choices=("gather", "vselr"))
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    # Positional flexibility for sim_dsl.sh: N H G Thr arm lut
    arm = args.arm
    lut = args.lut
    if args.arm_pos in ("unpack", "unpack_sf"):
        arm = args.arm_pos
        if args.lut_pos in ("gather", "vselr"):
            lut = args.lut_pos
    elif args.arm_pos in ("gather", "vselr"):
        lut = args.arm_pos
        if args.lut_pos in ("unpack", "unpack_sf"):
            arm = args.lut_pos

    with_sf = arm == "unpack_sf"
    tag = args.tag or f"sp6d_n{N}_h{H}_g{G}_t{THR}_{arm}_{lut}"
    if args.emit_mlir:
        pto = _try_import_pto()
        if pto is None:
            print(f"UNRUN {tag}: no ptodsl for --emit-mlir", flush=True)
            return 1
        if lut == "vselr" and not _has_vselr(pto):
            print(f"UNRUN {tag}: no vselr binding", flush=True)
            return 1
        k = build_kernel(pto, with_sf, lut)
        print(k.compile().mlir_text())
        return 0
    return run_acl(with_sf, lut, tag)


if __name__ == "__main__":
    raise SystemExit(main())
