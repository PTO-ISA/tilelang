#!/usr/bin/env python3
"""ST-SV9d — PTO-DSL twin of Simt SV9 (topk e2e KEEP / remat).

Simt: kernels/sv9_topk_e2e.py
Phase2: keep @ sv9d_e256_k8_t32_keep; also remat_scores, remat_idx

Mapping:
  keep         (3): scores+idxs stay-alive; kill in RF
  remat_scores (1+3): reload scores from UB each K; kill in UB; idxs KEEP
  remat_idx    (3): scores KEEP; remat index via vci each use; kill in RF
No AABBCC / vf_fuse. Pattern from pto-vmi TopkGateVfKernel (E=256 / 4 VL).
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
N = E // VL

NEG_INF = -3.40282347e38
INT_MAX = 0x7FFFFFFF

SCORES_BYTES = E * 4
OUT_BYTES = K * 4
OFF_OUT = SCORES_BYTES

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _kernel_keep():
    @pto.jit(
        name="SV9d_e256_k8_keep",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.i32, "gm"), scores_gm: pto.ptr(pto.f32, "gm")):
        scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.i32, "ub"))

        pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF), size=VL)
        int_max = pto.vmi.vbrc(pto.i32(INT_MAX), size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)

        s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
        s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
        s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
        s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
        i0 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(0 * VL), size=VL), arange, mask)
        i1 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(1 * VL), size=VL), arange, mask)
        i2 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(2 * VL), size=VL), arange, mask)
        i3 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(3 * VL), size=VL), arange, mask)

        for k in range(K):
            m01 = pto.vmi.vmax(s0, s1, mask)
            m23 = pto.vmi.vmax(s2, s3, mask)
            top_s = pto.vmi.vcmax(pto.vmi.vmax(m01, m23, mask), mask)
            top_b = pto.vmi.vbrc(top_s, size=VL)
            c0 = pto.vmi.vsel(pto.vmi.vcmp(s0, top_b, mask, "eq"), i0, int_max)
            c1 = pto.vmi.vsel(pto.vmi.vcmp(s1, top_b, mask, "eq"), i1, int_max)
            c2 = pto.vmi.vsel(pto.vmi.vcmp(s2, top_b, mask, "eq"), i2, int_max)
            c3 = pto.vmi.vsel(pto.vmi.vcmp(s3, top_b, mask, "eq"), i3, int_max)
            n01 = pto.vmi.vmin(c0, c1, mask)
            n23 = pto.vmi.vmin(c2, c3, mask)
            win_s = pto.vmi.vcmin(pto.vmi.vmin(n01, n23, mask), mask)
            pto.vmi.vstore(win_s, out_ub, k)
            win_b = pto.vmi.vbrc(win_s, size=VL)
            s0 = pto.vmi.vsel(pto.vmi.vcmp(i0, win_b, mask, "eq"), neg_inf, s0)
            s1 = pto.vmi.vsel(pto.vmi.vcmp(i1, win_b, mask, "eq"), neg_inf, s1)
            s2 = pto.vmi.vsel(pto.vmi.vcmp(i2, win_b, mask, "eq"), neg_inf, s2)
            s3 = pto.vmi.vsel(pto.vmi.vcmp(i3, win_b, mask, "eq"), neg_inf, s3)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def _kernel_remat_scores():
    @pto.jit(
        name="SV9d_e256_k8_remat_scores",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.i32, "gm"), scores_gm: pto.ptr(pto.f32, "gm")):
        scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.i32, "ub"))

        pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF), size=VL)
        int_max = pto.vmi.vbrc(pto.i32(INT_MAX), size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)
        i0 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(0 * VL), size=VL), arange, mask)
        i1 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(1 * VL), size=VL), arange, mask)
        i2 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(2 * VL), size=VL), arange, mask)
        i3 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(3 * VL), size=VL), arange, mask)

        for k in range(K):
            s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
            s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
            s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
            s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)
            m01 = pto.vmi.vmax(s0, s1, mask)
            m23 = pto.vmi.vmax(s2, s3, mask)
            top_s = pto.vmi.vcmax(pto.vmi.vmax(m01, m23, mask), mask)
            top_b = pto.vmi.vbrc(top_s, size=VL)
            c0 = pto.vmi.vsel(pto.vmi.vcmp(s0, top_b, mask, "eq"), i0, int_max)
            c1 = pto.vmi.vsel(pto.vmi.vcmp(s1, top_b, mask, "eq"), i1, int_max)
            c2 = pto.vmi.vsel(pto.vmi.vcmp(s2, top_b, mask, "eq"), i2, int_max)
            c3 = pto.vmi.vsel(pto.vmi.vcmp(s3, top_b, mask, "eq"), i3, int_max)
            n01 = pto.vmi.vmin(c0, c1, mask)
            n23 = pto.vmi.vmin(c2, c3, mask)
            win_s = pto.vmi.vcmin(pto.vmi.vmin(n01, n23, mask), mask)
            pto.vmi.vstore(win_s, out_ub, k)
            win_b = pto.vmi.vbrc(win_s, size=VL)
            pto.vmi.vstore(pto.vmi.vsel(pto.vmi.vcmp(i0, win_b, mask, "eq"), neg_inf, s0), scores_ub, 0 * VL, mask)
            pto.vmi.vstore(pto.vmi.vsel(pto.vmi.vcmp(i1, win_b, mask, "eq"), neg_inf, s1), scores_ub, 1 * VL, mask)
            pto.vmi.vstore(pto.vmi.vsel(pto.vmi.vcmp(i2, win_b, mask, "eq"), neg_inf, s2), scores_ub, 2 * VL, mask)
            pto.vmi.vstore(pto.vmi.vsel(pto.vmi.vcmp(i3, win_b, mask, "eq"), neg_inf, s3), scores_ub, 3 * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def _kernel_remat_idx():
    @pto.jit(
        name="SV9d_e256_k8_remat_idx",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.i32, "gm"), scores_gm: pto.ptr(pto.f32, "gm")):
        scores_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.i32, "ub"))

        pto.mte_load(scores_gm, scores_ub, 0, SCORES_BYTES, nburst=(1, SCORES_BYTES, SCORES_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg_inf = pto.vmi.vbrc(pto.f32(NEG_INF), size=VL)
        int_max = pto.vmi.vbrc(pto.i32(INT_MAX), size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)

        s0 = pto.vmi.vload(scores_ub, 0 * VL, size=VL)
        s1 = pto.vmi.vload(scores_ub, 1 * VL, size=VL)
        s2 = pto.vmi.vload(scores_ub, 2 * VL, size=VL)
        s3 = pto.vmi.vload(scores_ub, 3 * VL, size=VL)

        for k in range(K):
            m01 = pto.vmi.vmax(s0, s1, mask)
            m23 = pto.vmi.vmax(s2, s3, mask)
            top_s = pto.vmi.vcmax(pto.vmi.vmax(m01, m23, mask), mask)
            top_b = pto.vmi.vbrc(top_s, size=VL)
            i0 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(0 * VL), size=VL), arange, mask)
            i1 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(1 * VL), size=VL), arange, mask)
            i2 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(2 * VL), size=VL), arange, mask)
            i3 = pto.vmi.vadd(pto.vmi.vbrc(pto.i32(3 * VL), size=VL), arange, mask)
            c0 = pto.vmi.vsel(pto.vmi.vcmp(s0, top_b, mask, "eq"), i0, int_max)
            c1 = pto.vmi.vsel(pto.vmi.vcmp(s1, top_b, mask, "eq"), i1, int_max)
            c2 = pto.vmi.vsel(pto.vmi.vcmp(s2, top_b, mask, "eq"), i2, int_max)
            c3 = pto.vmi.vsel(pto.vmi.vcmp(s3, top_b, mask, "eq"), i3, int_max)
            n01 = pto.vmi.vmin(c0, c1, mask)
            n23 = pto.vmi.vmin(c2, c3, mask)
            win_s = pto.vmi.vcmin(pto.vmi.vmin(n01, n23, mask), mask)
            pto.vmi.vstore(win_s, out_ub, k)
            win_b = pto.vmi.vbrc(win_s, size=VL)
            s0 = pto.vmi.vsel(pto.vmi.vcmp(i0, win_b, mask, "eq"), neg_inf, s0)
            s1 = pto.vmi.vsel(pto.vmi.vcmp(i1, win_b, mask, "eq"), neg_inf, s1)
            s2 = pto.vmi.vsel(pto.vmi.vcmp(i2, win_b, mask, "eq"), neg_inf, s2)
            s3 = pto.vmi.vsel(pto.vmi.vcmp(i3, win_b, mask, "eq"), neg_inf, s3)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def reference(scores):
    row = scores.copy()
    out = np.empty(K, dtype=np.int32)
    for k in range(K):
        mx = float(row.max())
        cand = np.where(row == mx)[0]
        w = int(cand.min())
        out[k] = w
        row[w] = NEG_INF
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
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def run_acl(kernel, tag: str) -> int:
    rng = np.random.RandomState(2026)
    scores = rng.uniform(-3, 3, E).astype(np.float32)
    ref = reference(scores)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    sd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(sd), SCORES_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "ms")
    _chk(lib.aclrtMalloc(ctypes.byref(od), OUT_BYTES, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            sd, SCORES_BYTES, np.ascontiguousarray(scores).ctypes.data_as(ctypes.c_void_p), SCORES_BYTES,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(sd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(K, dtype=np.int32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(sd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.array_equal(oh, ref):
        raise AssertionError(f"mismatch got={oh.tolist()} ref={ref.tolist()}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=E)
    ap.add_argument("K", nargs="?", type=int, default=K)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument(
        "arm", nargs="?", default="keep", choices=("keep", "remat_scores", "remat_idx")
    )
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    tag = f"sv9d_e{E}_k{K}_t{T}_{args.arm}"
    if args.arm == "keep":
        kernel = _kernel_keep()
    elif args.arm == "remat_scores":
        kernel = _kernel_remat_scores()
    else:
        kernel = _kernel_remat_idx()
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag)


if __name__ == "__main__":
    raise SystemExit(main())
