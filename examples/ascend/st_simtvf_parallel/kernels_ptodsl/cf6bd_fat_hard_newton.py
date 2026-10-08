#!/usr/bin/env python3
"""ST-CF6bd — PTO-DSL lockstep twin of CF6b (fat hard / thin easy).

Simt twin: kernels/cf6b_fat_hard_newton.py
Design: reports/ST_CF6_DIVERGENCE_CASE.md

Fair vector rooftop: ALWAYS run the fat hard pipeline (seed + N_HARD Newton
+ N_POLY Horner) in vregs, plus a thin easy path (seed + N_FAST), then
vsel by near/hard masks. Wall should be ~flat vs phard.

Primary tags: cf6bd_e256_t32_phard5 / cf6bd_e256_t32_phard25
Use camodel .camodel_deps_vmi018 on pto-b10 (NOT bare PTOAS-vmi/ptodsl).
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

E = 256
T = 32  # mirrored in tag; schedule uses VL chunks
VL = 64
N_CHUNKS = E // VL
EPS = 1e-4
HARD_HI = 1e-1
N_FAST = 2
N_HARD = 12
POLY_C = (0.0, 1e-6, -2e-6, 3e-6, -4e-6, 5e-6, -6e-6, 7e-6)
X_BYTES = E * 4
OUT_BYTES = E * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="CF6bd_e256",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel(
    out_gm: pto.ptr(pto.f32, "gm"),
    x_gm: pto.ptr(pto.f32, "gm"),
):
    x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(X_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(x_gm, x_ub, 0, X_BYTES, nburst=(1, X_BYTES, X_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
    zero = pto.vmi.vbrc(pto.f32(0.0), size=VL)
    one = pto.vmi.vbrc(pto.f32(1.0), size=VL)
    two = pto.vmi.vbrc(pto.f32(2.0), size=VL)
    eps_v = pto.vmi.vbrc(pto.f32(EPS), size=VL)
    hard_hi_v = pto.vmi.vbrc(pto.f32(HARD_HI), size=VL)
    c0 = pto.vmi.vbrc(pto.f32(POLY_C[0]), size=VL)
    c1 = pto.vmi.vbrc(pto.f32(POLY_C[1]), size=VL)
    c2 = pto.vmi.vbrc(pto.f32(POLY_C[2]), size=VL)
    c3 = pto.vmi.vbrc(pto.f32(POLY_C[3]), size=VL)
    c4 = pto.vmi.vbrc(pto.f32(POLY_C[4]), size=VL)
    c5 = pto.vmi.vbrc(pto.f32(POLY_C[5]), size=VL)
    c6 = pto.vmi.vbrc(pto.f32(POLY_C[6]), size=VL)
    c7 = pto.vmi.vbrc(pto.f32(POLY_C[7]), size=VL)

    # Lockstep rooftop: always fat + thin, then nested vsel (near → 0, hard → yh, else ye)
    for i in range(N_CHUNKS):
        off = i * VL
        x = pto.vmi.vload(x_ub, off, size=VL)
        ax = pto.vmi.vmax(x, pto.vmi.vmul(x, neg1, mask), mask)
        near = pto.vmi.vcmp(ax, eps_v, mask, "lt")
        hard_lt = pto.vmi.vcmp(ax, hard_hi_v, mask, "lt")  # includes near; near wins below
        safe_x = pto.vmi.vsel(near, one, x)

        # FAT hard pipeline (always)
        y_h = pto.vmi.vdiv(one, safe_x, mask)
        for _t in range(N_HARD):
            xy = pto.vmi.vmul(safe_x, y_h, mask)
            y_h = pto.vmi.vmul(y_h, pto.vmi.vsub(two, xy, mask), mask)
        # Horner polish: ((((c7*x+c6)*x+...)+c0)
        p = c7
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c6, mask)
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c5, mask)
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c4, mask)
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c3, mask)
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c2, mask)
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c1, mask)
        p = pto.vmi.vadd(pto.vmi.vmul(p, safe_x, mask), c0, mask)
        y_h = pto.vmi.vadd(y_h, p, mask)

        # THIN easy pipeline (always)
        y_e = pto.vmi.vdiv(one, safe_x, mask)
        for _t in range(N_FAST):
            xy = pto.vmi.vmul(safe_x, y_e, mask)
            y_e = pto.vmi.vmul(y_e, pto.vmi.vsub(two, xy, mask), mask)

        # y = near ? 0 : (ax < HARD_HI ? y_h : y_e)
        mid = pto.vmi.vsel(hard_lt, y_h, y_e)
        y = pto.vmi.vsel(near, zero, mid)
        pto.vmi.vstore(y, out_ub, off, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


def reference(x, eps=EPS, hard_hi=HARD_HI, n_fast=N_FAST, n_hard=N_HARD):
    """Host gold matching Simt CF6b."""
    out = np.zeros_like(x, dtype=np.float32)
    cs = [np.float32(c) for c in POLY_C]
    for i in range(x.shape[0]):
        xi = np.float32(x[i])
        ax = abs(float(xi))
        if ax < float(eps):
            out[i] = 0.0
        elif ax < float(hard_hi):
            y = np.float32(1.0) / xi
            for _ in range(n_hard):
                y = y * (np.float32(2.0) - xi * y)
            p = cs[7]
            for k in range(6, -1, -1):
                p = p * xi + cs[k]
            out[i] = y + p
        else:
            y = np.float32(1.0) / xi
            for _ in range(n_fast):
                y = y * (np.float32(2.0) - xi * y)
            out[i] = y
    return out


def _make_inputs(phard: int, seed: int = 26):
    """phard% lanes forced into hard band [EPS, HARD_HI); rest easy."""
    rng = np.random.default_rng(seed)
    n_hard = max(1, int(round(E * phard / 100.0)))
    x = rng.uniform(0.5, 2.0, size=E).astype(np.float32)  # easy
    lo = np.float32(EPS * 2.0)
    hi = np.float32(HARD_HI * 0.9)
    x[:n_hard] = rng.uniform(lo, hi, size=n_hard).astype(np.float32)
    mask_bad = (np.abs(x) < EPS) & (np.arange(E) >= n_hard)
    x = np.where(mask_bad, np.float32(1.0), x).astype(np.float32)
    return x


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


def run_acl(tag: str, phard: int) -> int:
    x = _make_inputs(phard)
    ref = reference(x)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), X_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), OUT_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            xd, X_BYTES, np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p), X_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_x",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(E, dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(oh, ref, rtol=2e-5, atol=1e-6):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(oh - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s maxabs={np.max(np.abs(oh - ref))}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument("phard", nargs="?", type=int, default=5)
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("--gold-check", action="store_true", help="numpy gold sanity only")
    args = ap.parse_args(argv)
    if args.E != E or args.T != T:
        print(f"note: binary fixed E={E} T={T}; got E={args.E} T={args.T}", flush=True)
    tag = f"cf6bd_e{E}_t{T}_phard{args.phard}"
    if args.gold_check:
        x = _make_inputs(args.phard)
        y = reference(x)
        n_hard = int(np.sum((np.abs(x) >= EPS) & (np.abs(x) < HARD_HI)))
        print(f"OK {tag} gold n_hard_lanes={n_hard} y[0]={y[0]} y_easy≈{y[-1]}")
        return 0
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag, args.phard)


if __name__ == "__main__":
    raise SystemExit(main())
