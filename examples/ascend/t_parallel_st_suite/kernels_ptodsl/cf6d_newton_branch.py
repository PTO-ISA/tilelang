#!/usr/bin/env python3
"""ST-CF6d — PTO-DSL explicit twin of Simt CF6 (Newton recip + near-0).

Simt twin: kernels/cf6_newton_branch.py
CF6v ABI tax (DO NOT copy): Newton in Kernel-serial *outside* SimdVF → ~12.35µs.
This *d* keeps seed recip + N_FAST=3 Newton iters entirely in vregs
(vdiv/vmul/vsub + vmax abs + vcmp/vsel) — no outside-VF serial / UB remat of y.

Primary tag: cf6d_e256_t32_pnear5
Mapping (2)/(3): fused near-0 Select + stay-alive y across Newton iters.
EPS=1e-4; N_FAST=3; pnear is host-data skew only (tag).
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
N_FAST = 3
X_BYTES = E * 4
OUT_BYTES = E * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="CF6d_e256",
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

    # All 4 VL chunks: near-0 → 0; else seed 1/x + 3 Newton y=y*(2-x*y) in RF
    for i in range(N_CHUNKS):
        off = i * VL
        x = pto.vmi.vload(x_ub, off, size=VL)
        # abs via max(v, -v)
        ax = pto.vmi.vmax(x, pto.vmi.vmul(x, neg1, mask), mask)
        near = pto.vmi.vcmp(ax, eps_v, mask, "lt")
        # safe denom so near-0 lanes never Inf/NaN the Newton pipeline
        safe_x = pto.vmi.vsel(near, one, x)
        # seed recip in vreg (no UB remat)
        y = pto.vmi.vdiv(one, safe_x, mask)
        # N_FAST=3 Newton iters entirely in vregs: y = y * (2 - x*y)
        for _t in range(N_FAST):
            xy = pto.vmi.vmul(safe_x, y, mask)
            two_m_xy = pto.vmi.vsub(two, xy, mask)
            y = pto.vmi.vmul(y, two_m_xy, mask)
        y = pto.vmi.vsel(near, zero, y)
        pto.vmi.vstore(y, out_ub, off, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


def reference(x, eps=EPS, n_fast=N_FAST):
    """Host gold matching Simt CF6 Newton semantics."""
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        ax = abs(float(x[i]))
        if ax < float(eps):
            out[i] = 0.0
        else:
            y = np.float32(1.0) / x[i]
            for _ in range(n_fast):
                y = y * (np.float32(2.0) - x[i] * y)
            out[i] = y
    return out


def _make_inputs(pnear: int, seed: int = 16):
    """pnear% lanes forced near-0 (same skew recipe as run_opsim_generic)."""
    eps = np.float32(EPS)
    rng = np.random.default_rng(seed)
    n_near = max(1, int(round(E * pnear / 100.0)))
    x = rng.standard_normal((E,)).astype(np.float32) * 2.0 + 1.0
    x[:n_near] = rng.uniform(-eps * 0.5, eps * 0.5, size=n_near).astype(np.float32)
    mask_far = (np.abs(x) < eps) & (np.arange(E) >= n_near)
    x = np.where(mask_far, np.sign(x) * (eps * 2.0 + 0.5), x).astype(np.float32)
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


def run_acl(tag: str, pnear: int) -> int:
    x = _make_inputs(pnear)
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
    ap.add_argument("pnear", nargs="?", type=int, default=5)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.E != E or args.T != T:
        print(f"note: binary fixed E={E} T={T}; got E={args.E} T={args.T}", flush=True)
    tag = f"cf6d_e{E}_t{T}_pnear{args.pnear}"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag, args.pnear)


if __name__ == "__main__":
    raise SystemExit(main())
