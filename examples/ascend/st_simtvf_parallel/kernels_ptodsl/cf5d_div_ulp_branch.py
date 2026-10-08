#!/usr/bin/env python3
"""ST-CF5d — PTO-DSL explicit twin of Simt CF5 (div + near-0 branch).

Simt twin: kernels/cf5_div_ulp_branch.py
CF5v ABI tax (DO NOT copy): Kernel-serial Div *outside* SimdVF → ~6.23µs.
This *d* keeps Div in vector RF: vmax abs + vcmp lt eps + vsel/vdiv — no outside-VF serial.

Primary tag: cf5d_e256_t32_pnear5
Mapping (2) fused compute with near-0 mask Select. All 4 VL chunks.
EPS=1e-4 fixed in kernel; pnear is host-data skew only (tag).
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
A_BYTES = E * 4
B_BYTES = E * 4
OUT_BYTES = E * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="CF5d_e256",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel(
    out_gm: pto.ptr(pto.f32, "gm"),
    a_gm: pto.ptr(pto.f32, "gm"),
    b_gm: pto.ptr(pto.f32, "gm"),
):
    a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    b_ub = pto.castptr(pto.const(A_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(
        pto.const(A_BYTES + B_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub")
    )

    pto.mte_load(a_gm, a_ub, 0, A_BYTES, nburst=(1, A_BYTES, A_BYTES))
    pto.mte_load(b_gm, b_ub, 0, B_BYTES, nburst=(1, B_BYTES, B_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
    zero = pto.vmi.vbrc(pto.f32(0.0), size=VL)
    eps_v = pto.vmi.vbrc(pto.f32(EPS), size=VL)

    # Mapping (2): fused abs+vcmp+vsel+vdiv per VL chunk — all 4 chunks in RF
    for i in range(N_CHUNKS):
        off = i * VL
        a = pto.vmi.vload(a_ub, off, size=VL)
        b = pto.vmi.vload(b_ub, off, size=VL)
        # abs via max(v, -v) — prefer over vabs
        ax = pto.vmi.vmax(a, pto.vmi.vmul(a, neg1, mask), mask)
        near = pto.vmi.vcmp(ax, eps_v, mask, "lt")
        quot = pto.vmi.vdiv(a, b, mask)
        y = pto.vmi.vsel(near, zero, quot)
        pto.vmi.vstore(y, out_ub, off, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


def reference(a, b, eps=EPS):
    """Host gold matching Simt CF5: y=a/b if abs(a)>=eps else 0."""
    ax = np.maximum(a, -a)
    return np.where(ax < np.float32(eps), np.float32(0.0), (a / b).astype(np.float32)).astype(
        np.float32
    )


def _make_inputs(pnear: int, seed: int = 15):
    """pnear% lanes forced near-0 in a (same skew recipe as run_opsim_generic)."""
    eps = np.float32(EPS)
    rng = np.random.default_rng(seed)
    n_near = max(1, int(round(E * pnear / 100.0)))
    a = rng.standard_normal((E,)).astype(np.float32) * 2.0
    b = rng.standard_normal((E,)).astype(np.float32) * 2.0 + 0.5
    a[:n_near] = rng.uniform(-eps * 0.5, eps * 0.5, size=n_near).astype(np.float32)
    mask = np.abs(a) < eps
    a = np.where(mask & (np.arange(E) >= n_near), np.sign(a) * (eps * 2.0 + 0.1), a).astype(
        np.float32
    )
    return a, b


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
    a, b = _make_inputs(pnear)
    ref = reference(a, b)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    ad = ctypes.c_void_p()
    bd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(ad), A_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "ma")
    _chk(lib.aclrtMalloc(ctypes.byref(bd), B_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mb")
    _chk(lib.aclrtMalloc(ctypes.byref(od), OUT_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            ad, A_BYTES, np.ascontiguousarray(a).ctypes.data_as(ctypes.c_void_p), A_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_a",
    )
    _chk(
        lib.aclrtMemcpy(
            bd, B_BYTES, np.ascontiguousarray(b).ctypes.data_as(ctypes.c_void_p), B_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_b",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(ad.value), int(bd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(E, dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(ad)
    lib.aclrtFree(bd)
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
    tag = f"cf5d_e{E}_t{T}_pnear{args.pnear}"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag, args.pnear)


if __name__ == "__main__":
    raise SystemExit(main())
