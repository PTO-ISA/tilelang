#!/usr/bin/env python3
"""ST-CF4d — PTO-DSL explicit twin of Simt CF4 (nested if via nested vsel).

Simt twin: kernels/cf4_nested_if.py
Primary tags: cf4d_e256_t32_pfat5 / cf4d_e256_t32_pfat25
Mapping (2): nested vsel — x>hi → hi; else x<lo → lo; else x*scale (fat).
pfat is host-data skew only (tag), not a kernel param. No AABBCC / vf_fuse.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

E = 256
T = 32
VL = 64
N_CHUNKS = E // VL
A_BYTES = E * 4
SC_BYTES = E * 4
LO_BYTES = 4
HI_BYTES = 4
OUT_BYTES = E * 4
# UB layout: a | scale | lo(pad32) | hi(pad32) | out — 32B aligned (Ascend MTE/VSTI)
OFF_SC = A_BYTES
OFF_LO = A_BYTES + SC_BYTES          # 2048
OFF_HI = OFF_LO + 32                 # 2080
OFF_OUT = OFF_HI + 32                # 2112

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="CF4d_e256",
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
    sc_gm: pto.ptr(pto.f32, "gm"),
    lo_gm: pto.ptr(pto.f32, "gm"),
    hi_gm: pto.ptr(pto.f32, "gm"),
):
    a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    sc_ub = pto.castptr(pto.const(OFF_SC, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    lo_ub = pto.castptr(pto.const(OFF_LO, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    hi_ub = pto.castptr(pto.const(OFF_HI, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(a_gm, a_ub, 0, A_BYTES, nburst=(1, A_BYTES, A_BYTES))
    pto.mte_load(sc_gm, sc_ub, 0, SC_BYTES, nburst=(1, SC_BYTES, SC_BYTES))
    pto.mte_load(lo_gm, lo_ub, 0, LO_BYTES, nburst=(1, LO_BYTES, LO_BYTES))
    pto.mte_load(hi_gm, hi_ub, 0, HI_BYTES, nburst=(1, HI_BYTES, HI_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    lo_s = pto.vmi.vload(lo_ub, 0, size=1)
    hi_s = pto.vmi.vload(hi_ub, 0, size=1)
    lo_b = pto.vmi.vbrc(lo_s, size=VL)
    hi_b = pto.vmi.vbrc(hi_s, size=VL)

    # Mapping (2): nested vsel ≡ nested if (lockstep pays fat path)
    for i in range(N_CHUNKS):
        off = i * VL
        x = pto.vmi.vload(a_ub, off, size=VL)
        sc = pto.vmi.vload(sc_ub, off, size=VL)
        fat = pto.vmi.vmul(x, sc, mask)
        # y = Select(x>hi, hi, Select(x<lo, lo, x*scale))
        mid = pto.vmi.vsel(pto.vmi.vcmp(x, lo_b, mask, "lt"), lo_b, fat)
        y = pto.vmi.vsel(pto.vmi.vcmp(x, hi_b, mask, "gt"), hi_b, mid)
        pto.vmi.vstore(y, out_ub, off, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


def reference(a, scale, lo=-1.0, hi=1.0):
    lo = np.float32(lo)
    hi = np.float32(hi)
    exp = np.empty_like(a)
    for i in range(a.shape[0]):
        x = a[i]
        if x > hi:
            exp[i] = hi
        elif x < lo:
            exp[i] = lo
        else:
            exp[i] = x * scale[i]
    return exp.astype(np.float32)


def _make_inputs(pfat: int, seed: int = 14):
    """Same skew recipe as run_opsim_generic CF4 (seed 14)."""
    lo, hi = np.float32(-1.0), np.float32(1.0)
    rng = np.random.default_rng(seed)
    scale = (rng.standard_normal((E,)).astype(np.float32) * 0.5 + 1.0)
    n_fat = max(1, int(round(E * pfat / 100.0)))
    a = np.empty(E, np.float32)
    a[:n_fat] = rng.uniform(-0.99, 0.99, size=n_fat).astype(np.float32)
    n_hi = (E - n_fat) // 2
    a[n_fat : n_fat + n_hi] = rng.uniform(1.01, 3.0, size=n_hi).astype(np.float32)
    a[n_fat + n_hi :] = rng.uniform(-3.0, -1.01, size=E - n_fat - n_hi).astype(np.float32)
    rng.shuffle(a)
    return a, scale, np.array([lo], np.float32), np.array([hi], np.float32)


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


def run_acl(tag: str, pfat: int) -> int:
    a, scale, lo_t, hi_t = _make_inputs(pfat)
    ref = reference(a, scale, float(lo_t[0]), float(hi_t[0]))
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    ad = ctypes.c_void_p()
    sd = ctypes.c_void_p()
    lod = ctypes.c_void_p()
    hid = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(ad), A_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "ma")
    _chk(lib.aclrtMalloc(ctypes.byref(sd), SC_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    _chk(lib.aclrtMalloc(ctypes.byref(lod), LO_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mlo")
    _chk(lib.aclrtMalloc(ctypes.byref(hid), HI_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mhi")
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
            sd, SC_BYTES, np.ascontiguousarray(scale).ctypes.data_as(ctypes.c_void_p), SC_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_sc",
    )
    _chk(
        lib.aclrtMemcpy(
            lod, LO_BYTES, np.ascontiguousarray(lo_t).ctypes.data_as(ctypes.c_void_p), LO_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_lo",
    )
    _chk(
        lib.aclrtMemcpy(
            hid, HI_BYTES, np.ascontiguousarray(hi_t).ctypes.data_as(ctypes.c_void_p), HI_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_hi",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(ad.value), int(sd.value), int(lod.value), int(hid.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(E, dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(ad)
    lib.aclrtFree(sd)
    lib.aclrtFree(lod)
    lib.aclrtFree(hid)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(oh - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s maxabs={np.max(np.abs(oh - ref))}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument("pfat", nargs="?", type=int, default=5)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.E != E or args.T != T:
        print(f"note: binary fixed E={E} T={T}; got E={args.E} T={args.T}", flush=True)
    tag = f"cf4d_e{E}_t{T}_pfat{args.pfat}"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag, args.pfat)


if __name__ == "__main__":
    raise SystemExit(main())
