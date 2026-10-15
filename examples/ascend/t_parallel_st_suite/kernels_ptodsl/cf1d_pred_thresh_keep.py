#!/usr/bin/env python3
"""ST-CF1d — PTO-DSL explicit twin of Simt CF1 (thresh kill KEEP).

Simt twin: kernels/cf1_pred_thresh_keep.py
Primary tag: cf1d_e256_k8_t32_keep
Mapping (3) scores KEEP in vregs; vcmp+vsel. No AABBCC / vf_fuse.
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
    name="CF1d_e256_k8",
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

    # KEEP scores across K (mapping 3)
    s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
    s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
    s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
    s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

    for k in range(K):
        thr_vec = pto.vmi.vload(thr_ub, k, size=1)
        thr_b = pto.vmi.vbrc(thr_vec, size=VL)
        s0 = pto.vmi.vsel(pto.vmi.vcmp(s0, thr_b, mask, "gt"), neg_inf, s0)
        s1 = pto.vmi.vsel(pto.vmi.vcmp(s1, thr_b, mask, "gt"), neg_inf, s1)
        s2 = pto.vmi.vsel(pto.vmi.vcmp(s2, thr_b, mask, "gt"), neg_inf, s2)
        s3 = pto.vmi.vsel(pto.vmi.vcmp(s3, thr_b, mask, "gt"), neg_inf, s3)

    pto.vmi.vstore(s0, out_ub, 0 * VL, mask)
    pto.vmi.vstore(s1, out_ub, 1 * VL, mask)
    pto.vmi.vstore(s2, out_ub, 2 * VL, mask)
    pto.vmi.vstore(s3, out_ub, 3 * VL, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))


def reference(scores, thresh):
    out = scores.copy()
    for thr in thresh:
        out = np.where(out > thr, np.float32(NEG_INF_F32), out)
    return out.astype(np.float32)


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
    scores = rng.uniform(-3, 3, E).astype(np.float32)
    thresh = rng.uniform(-1, 1, K).astype(np.float32)
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
        lib.aclrtMemcpy(sd, SCORES_BYTES, np.ascontiguousarray(scores).ctypes.data_as(ctypes.c_void_p), SCORES_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d_s",
    )
    _chk(
        lib.aclrtMemcpy(td, THR_BYTES, np.ascontiguousarray(thresh).ctypes.data_as(ctypes.c_void_p), THR_BYTES, _ACL_MEMCPY_HOST_TO_DEVICE),
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
    _chk(lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST), "d2h")
    lib.aclrtFree(sd)
    lib.aclrtFree(td)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(oh - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("K", nargs="?", type=int, default=K)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    tag = f"cf1d_e{E}_k{K}_t{T}_keep"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag)


if __name__ == "__main__":
    raise SystemExit(main())
