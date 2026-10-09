#!/usr/bin/env python3
"""ST-SV1d — PTO-DSL explicit twin of Simt SV1 (stream eltwise).

Simt twin: kernels/sv1_stream_eltwise.py
Primary tag: sv1d_e256_t32
Mapping (1) ld+st+compute per VL chunk. No AABBCC / vf_fuse.
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
A_BYTES = E * 4
B_BYTES = E * 4
C_BYTES = E * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="SV1d_e256",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def kernel(
    c_gm: pto.ptr(pto.f32, "gm"),
    a_gm: pto.ptr(pto.f32, "gm"),
    b_gm: pto.ptr(pto.f32, "gm"),
):
    a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    b_ub = pto.castptr(pto.const(A_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    c_ub = pto.castptr(pto.const(A_BYTES + B_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(a_gm, a_ub, 0, A_BYTES, nburst=(1, A_BYTES, A_BYTES))
    pto.mte_load(b_gm, b_ub, 0, B_BYTES, nburst=(1, B_BYTES, B_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    for i in range(N_CHUNKS):
        off = i * VL
        a = pto.vmi.vload(a_ub, off, size=VL)
        b = pto.vmi.vload(b_ub, off, size=VL)
        c = pto.vmi.vadd(a, b, mask)
        pto.vmi.vstore(c, c_ub, off, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(c_ub, c_gm, C_BYTES, nburst=(1, C_BYTES, C_BYTES))


def reference(a, b):
    return (a + b).astype(np.float32)


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


def run_acl(tag: str) -> int:
    rng = np.random.RandomState(2026)
    a = rng.uniform(-3, 3, E).astype(np.float32)
    b = rng.uniform(-3, 3, E).astype(np.float32)
    ref = reference(a, b)
    nbytes = E * 4
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    ad = ctypes.c_void_p()
    bd = ctypes.c_void_p()
    cd = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(ad), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "ma")
    _chk(lib.aclrtMalloc(ctypes.byref(bd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mb")
    _chk(lib.aclrtMalloc(ctypes.byref(cd), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mc")
    _chk(
        lib.aclrtMemcpy(ad, nbytes, np.ascontiguousarray(a).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d_a",
    )
    _chk(
        lib.aclrtMemcpy(bd, nbytes, np.ascontiguousarray(b).ctypes.data_as(ctypes.c_void_p), nbytes, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d_b",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(cd.value), int(ad.value), int(bd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    ch = np.zeros(E, dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(ch.ctypes.data_as(ctypes.c_void_p), nbytes, cd, nbytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(ad)
    lib.aclrtFree(bd)
    lib.aclrtFree(cd)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(ch, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(ch - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.E != E or args.T != T:
        print(f"note: Phase1 binary is fixed E={E} T={T}; got E={args.E} T={args.T}", flush=True)
    tag = f"sv1d_e{E}_t{T}"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag)


if __name__ == "__main__":
    raise SystemExit(main())
