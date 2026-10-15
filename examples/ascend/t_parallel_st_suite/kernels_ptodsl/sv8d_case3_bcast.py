#!/usr/bin/env python3
"""ST-SV8d — PTO-DSL Layer D twin of Simt SV8 (Case-3 bcast live vs spill).

Tags: sv8d_r{R}_c{C}_g{G}_t{T}_{live|spill_dist}
Only size∈{1,64}. Out stored as VL=64 chunks (C multiple of 64).
live: sf_inv size-1 KEEP; expand 4 group scales/VL via scratch+vsel range masks.
spill_dist: expand sf to scale_ub[R,C] via same pack; reload VL mul.
Assumes G divides VL (G=16, VL=64 → 4 groups per VL chunk).
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

EPS = 1e-6
T_DEFAULT = 32
VL = 64

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _layout(R, C, G, spill: bool):
    CG = C // G
    assert C % VL == 0
    assert VL % G == 0
    x_bytes = R * C * 4
    out_bytes = R * C * 4
    off_out = x_bytes
    # scratch: one VL slot for scale expand roundtrip
    off_scratch = x_bytes + out_bytes
    off_scale = off_scratch + VL * 4
    return CG, x_bytes, out_bytes, off_out, off_scale, off_scratch


def _reduce_sf_row(x_ub, i, C, G, CG, mask1, neg1):
    """Return list of CG size-1 inv vregs."""
    sf = []
    for g in range(CG):
        m = pto.vmi.vbrc(pto.f32(0.0), size=1)
        for t in range(G):
            v = pto.vmi.vload(x_ub, i * C + g * G + t, size=1)
            absv = pto.vmi.vmax(v, pto.vmi.vmul(v, neg1, mask1), mask1)
            m = pto.vmi.vmax(m, absv, mask1)
        m = pto.vmi.vmax(m, pto.vmi.vbrc(pto.f32(EPS), size=1), mask1)
        sf.append(pto.vmi.vdiv(pto.vmi.vbrc(pto.f32(1.0), size=1), m, mask1))
    return sf


def _range_mask(arange, lo, hi, mask_vl):
    """Predicate true for lanes in [lo, hi)."""
    ge = pto.vmi.vcmp(arange, pto.vmi.vbrc(pto.i32(lo), size=VL), mask_vl, "ge")
    lt = pto.vmi.vcmp(arange, pto.vmi.vbrc(pto.i32(hi), size=VL), mask_vl, "lt")
    # vsel(ge, lt, zeros_as_false): keep lt where ge else false
    # Use vsel on i32 1/0 then vcmp eq 1 — or nest: vsel(ge, lt_as_select_on_ones, zero_mask)
    # Simpler: vsel(ge, ones, zeros) AND via vmin on mask? masks may be special.
    # Practical: vsel(ge, vsel(lt, one_f, zero_f), zero_f) then vcmp eq one — heavy.
    # Use: out = vsel(lt, vsel(ge, src, dst), dst) pattern at call site instead.
    return ge, lt


def _expand_scale_vl(sf_chunk, scratch_ub, arange, mask_vl, G):
    """sf_chunk: list of (VL/G) size-1 scales for one VL. Build dense VL scale."""
    ngrp = VL // G
    sc = pto.vmi.vbrc(pto.f32(0.0), size=VL)
    for gi in range(ngrp):
        # aligned VST + reload avoids ensure_layout on vbrc(sz1)→vsel
        pto.vmi.vstore(pto.vmi.vbrc(sf_chunk[gi], size=VL), scratch_ub, 0, mask_vl)
        pto.mem_bar("VST_VLD")
        sb = pto.vmi.vload(scratch_ub, 0, size=VL)
        lo = gi * G
        hi = lo + G
        ge = pto.vmi.vcmp(arange, pto.vmi.vbrc(pto.i32(lo), size=VL), mask_vl, "ge")
        lt = pto.vmi.vcmp(arange, pto.vmi.vbrc(pto.i32(hi), size=VL), mask_vl, "lt")
        # lanes in [lo,hi): vsel(lt, vsel(ge, sb, sc), sc)
        mid = pto.vmi.vsel(ge, sb, sc)
        sc = pto.vmi.vsel(lt, mid, sc)
    return sc


def _kernel_live(R: int, C: int, G: int):
    CG, x_bytes, out_bytes, off_out, _, off_scratch = _layout(R, C, G, False)
    NCH = C // VL
    ngrp = VL // G

    @pto.jit(
        name=f"SV8d_r{R}_c{C}_g{G}_live",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        scratch_ub = pto.castptr(pto.const(off_scratch, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask1 = pto.vmi.create_mask(1, size=1)
        mask_vl = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=1)
        arange = pto.vmi.vci(pto.i32(0), size=VL)

        with pto.for_(0, R, step=1) as i:
            sf = _reduce_sf_row(x_ub, i, C, G, CG, mask1, neg1)
            for ch in range(NCH):
                base_g = (ch * VL) // G
                chunk = [sf[base_g + gi] for gi in range(ngrp)]
                sc = _expand_scale_vl(chunk, scratch_ub, arange, mask_vl, G)
                x = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)
                pto.vmi.vstore(pto.vmi.vmul(x, sc, mask_vl), out_ub, i * C + ch * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel_spill_dist(R: int, C: int, G: int):
    CG, x_bytes, out_bytes, off_out, off_scale, off_scratch = _layout(R, C, G, True)
    NCH = C // VL
    ngrp = VL // G

    @pto.jit(
        name=f"SV8d_r{R}_c{C}_g{G}_spill_dist",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        scale_ub = pto.castptr(pto.const(off_scale, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        scratch_ub = pto.castptr(pto.const(off_scratch, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask1 = pto.vmi.create_mask(1, size=1)
        mask_vl = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=1)
        arange = pto.vmi.vci(pto.i32(0), size=VL)

        with pto.for_(0, R, step=1) as i:
            sf = _reduce_sf_row(x_ub, i, C, G, CG, mask1, neg1)
            for ch in range(NCH):
                base_g = (ch * VL) // G
                chunk = [sf[base_g + gi] for gi in range(ngrp)]
                sc = _expand_scale_vl(chunk, scratch_ub, arange, mask_vl, G)
                pto.vmi.vstore(sc, scale_ub, i * C + ch * VL, mask_vl)
            pto.mem_bar("VST_VLD")
            for ch in range(NCH):
                s = pto.vmi.vload(scale_ub, i * C + ch * VL, size=VL)
                x = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)
                pto.vmi.vstore(pto.vmi.vmul(x, s, mask_vl), out_ub, i * C + ch * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def reference(X, R, C, G):
    CG = C // G
    Out = np.empty_like(X)
    for i in range(R):
        for g in range(CG):
            chunk = X[i, g * G : (g + 1) * G].astype(np.float32)
            m = max(float(np.max(np.abs(chunk))), EPS)
            inv = np.float32(1.0 / m)
            Out[i, g * G : (g + 1) * G] = (chunk * inv).astype(np.float32)
    return Out


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


def run_acl(kernel, tag, R, C, G) -> int:
    _, x_bytes, out_bytes, _, _, _ = _layout(R, C, G, True)
    rng = np.random.RandomState(8)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref = reference(X, R, C, G)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), out_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros((R, C), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), out_bytes, od, out_bytes,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    maxabs = float(np.max(np.abs(oh - ref)))
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={maxabs}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s maxabs={maxabs:.3e}")
    return 0


_ARM_BUILDERS = {
    "live": _kernel_live,
    "spill_dist": _kernel_spill_dist,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("R", nargs="?", type=int, default=64)
    ap.add_argument("C", nargs="?", type=int, default=128)
    ap.add_argument("G", nargs="?", type=int, default=16)
    ap.add_argument("T", nargs="?", type=int, default=T_DEFAULT)
    ap.add_argument("arm", nargs="?", default="live", choices=tuple(_ARM_BUILDERS))
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.C % args.G != 0:
        raise SystemExit(f"C={args.C} must be divisible by G={args.G}")
    if VL % args.G != 0:
        raise SystemExit(f"VL={VL} must be divisible by G={args.G}")
    tag = f"sv8d_r{args.R}_c{args.C}_g{args.G}_t{args.T}_{args.arm}"
    kernel = _ARM_BUILDERS[args.arm](args.R, args.C, args.G)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, args.R, args.C, args.G)


if __name__ == "__main__":
    raise SystemExit(main())
