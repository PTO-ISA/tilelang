#!/usr/bin/env python3
"""ST-CF2d — PTO-DSL explicit twin of Simt CF2 (remat + kill-in-shared).

Simt twin: kernels/cf2_remat_thresh_kill_shared.py
Primary tag: cf2d_e256_k8_t32_remat
Mapping (1): remat scores from UB each K; kill writes back to shared UB.
Contrast CF1d mapping (3) KEEP scores in vregs across K.
No AABBCC / vf_fuse.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

E = 256
K = 8
T = 32
VL = 64
N_CHUNKS = E // VL
SCORES_BYTES = E * 4
THR_BYTES = K * 4
OUT_BYTES = E * 4
NEG_INF_F32 = -3.40282347e38

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="CF2d_e256_k8",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def kernel(
    out_gm: pto.ptr(pto.f32, "gm"),
    scores_gm: pto.ptr(pto.f32, "gm"),
    thr_gm: pto.ptr(pto.f32, "gm"),
):
    scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    thr_ub = pto.castptr(pto.const(SCORES_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(
        pto.const(SCORES_BYTES + THR_BYTES, dtype=pto.i64), pto.ptr(pto.f32, "ub")
    )

    pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
    pto.mte_load(thr_gm, thr_ub, 0, THR_BYTES, nburst=(1, THR_BYTES, THR_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(VL, size=VL)
    neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF_F32), size=VL)

    # Mapping (1): remat from shared each k; kill-in-shared (store back to scores_ub)
    for k in range(K):
        thr_vec = pto.vmi.vload(thr_ub, k, size=1)
        thr_b = pto.vmi.vbrc(thr_vec, size=VL)
        for i in range(N_CHUNKS):
            off = i * VL
            s = pto.vmi.vload(scores_ub, off, size=VL)  # remat
            s = pto.vmi.vsel(pto.vmi.vcmp(s, thr_b, mask, "gt"), neg_inf, s)
            pto.vmi.vstore(s, scores_ub, off, mask)  # kill-in-shared

    # final publish to out_ub
    for i in range(N_CHUNKS):
        off = i * VL
        s = pto.vmi.vload(scores_ub, off, size=VL)
        pto.vmi.vstore(s, out_ub, off, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


def reference(scores, thresh):
    out = scores.copy()
    for thr in thresh:
        out = np.where(out > thr, np.float32(NEG_INF_F32), out)
    return out.astype(np.float32)


def _make_inputs(seed: int = 12):
    """Same recipe as run_opsim_generic CF2 (seed 12)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((E,)).astype(np.float32)
    thr = (np.sort(a.copy())[::-1][:K] * 0.99).astype(np.float32)
    return a, thr


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
    scores, thresh = _make_inputs()
    ref = reference(scores, thresh)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    sd = ctypes.c_void_p()
    td = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(sd), SCORES_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    _chk(lib.aclrtMalloc(ctypes.byref(td), THR_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mt")
    _chk(lib.aclrtMalloc(ctypes.byref(od), OUT_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            sd, SCORES_BYTES, np.ascontiguousarray(scores).ctypes.data_as(ctypes.c_void_p),
            SCORES_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_s",
    )
    _chk(
        lib.aclrtMemcpy(
            td, THR_BYTES, np.ascontiguousarray(thresh).ctypes.data_as(ctypes.c_void_p),
            THR_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d_t",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(sd.value), int(td.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(E, dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(sd)
    lib.aclrtFree(td)
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
    ap.add_argument("K", nargs="?", type=int, default=K)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.E != E or args.K != K or args.T != T:
        print(f"note: binary fixed E={E} K={K} T={T}; got E={args.E} K={args.K} T={args.T}", flush=True)
    tag = f"cf2d_e{E}_k{K}_t{T}_remat"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag)


if __name__ == "__main__":
    raise SystemExit(main())
