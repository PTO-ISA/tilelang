#!/usr/bin/env python3
"""ST-SV4d — PTO-DSL twin of Simt SV4 (index gather + partial-sum).

Simt: kernels/sv4_index_gather_psum.py (keep_idx / remat_idx — NOT keep2)
Tags: sv4d_e{E}_b{B}_t{T}_{keep_idx,remat_idx}

Gold: Out[i] = sum_b A[b,i] * W[Idx[i]],  Idx[i] = (i*7+3) % E
Mapping:
  keep_idx  (3): idxs + acc stay-alive across serial(B)  -> 2*NCH live VRs
  remat_idx (1+3): remat idxs from UB each b; KEEP acc   -> NCH live VRs
with NCH = E // VL vector registers per array (VL=64 f32 lanes).

2026-10-05: E/B are now real knobs (they used to be hard-wired to 256/8) so the
near-spill ladder E in {256,512,1024,2048} can be swept for both arms.
No AABBCC / vf_fuse.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

VL = 64

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _layout(E, B):
    a_bytes = B * E * 4
    w_bytes = E * 4
    idx_bytes = E * 4
    out_bytes = E * 4
    off_w = a_bytes
    off_idx = off_w + w_bytes
    off_out = off_idx + idx_bytes
    return a_bytes, w_bytes, idx_bytes, out_bytes, off_w, off_idx, off_out


def build_kernel(E: int, B: int, arm: str):
    """Build the SV4d kernel for (E, B, arm). NCH = E // VL chunks."""
    assert E % VL == 0, f"E={E} must be a multiple of VL={VL}"
    NCH = E // VL
    A_BYTES, W_BYTES, IDX_BYTES, OUT_BYTES, OFF_W, OFF_IDX, OFF_OUT = _layout(E, B)
    keep = arm == "keep_idx"

    @pto.jit(
        name=f"SV4d_e{E}_b{B}_{arm}",
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
        w_gm: pto.ptr(pto.f32, "gm"),
        idx_gm: pto.ptr(pto.i32, "gm"),
    ):
        a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        w_ub = pto.castptr(pto.const(OFF_W, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        idx_ub = pto.castptr(pto.const(OFF_IDX, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        out_ub = pto.castptr(pto.const(OFF_OUT, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(a_gm, a_ub, 0, A_BYTES, nburst=(1, A_BYTES, A_BYTES))
        pto.mte_load(w_gm, w_ub, 0, W_BYTES, nburst=(1, W_BYTES, W_BYTES))
        pto.mte_load(idx_gm, idx_ub, 0, IDX_BYTES, nburst=(1, IDX_BYTES, IDX_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        ix = None
        if keep:
            # KEEP idxs across B: NCH index VRs stay live
            ix = [pto.vmi.vload(idx_ub, c * VL, size=VL) for c in range(NCH)]
        acc = [pto.vmi.vbrc(pto.f32(0.0), size=VL) for _ in range(NCH)]

        for b in range(B):
            base = b * E
            if not keep:
                # remat: re-load the index VRs each b (dead after the gather)
                ix = [pto.vmi.vload(idx_ub, c * VL, size=VL) for c in range(NCH)]
            for c in range(NCH):
                a = pto.vmi.vload(a_ub, base + c * VL, size=VL)
                w = pto.vmi.vgather(w_ub, ix[c], mask)
                acc[c] = pto.vmi.vadd(acc[c], pto.vmi.vmul(a, w, mask), mask)

        for c in range(NCH):
            pto.vmi.vstore(acc[c], out_ub, c * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel


def reference(A, W, Idx, E, B):
    Out = np.zeros(E, dtype=np.float32)
    for b in range(B):
        Out += A[b] * W[Idx]
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


def run_acl(kernel, tag: str, E: int, B: int) -> int:
    A_BYTES, W_BYTES, IDX_BYTES, OUT_BYTES, _, _, _ = _layout(E, B)
    rng = np.random.RandomState(2026)
    A = rng.uniform(-1, 1, (B, E)).astype(np.float32)
    W = rng.uniform(-1, 1, E).astype(np.float32)
    Idx = ((np.arange(E, dtype=np.int64) * 7 + 3) % E).astype(np.int32)
    ref = reference(A, W, Idx, E, B)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    def malloc(n):
        p = ctypes.c_void_p()
        _chk(lib.aclrtMalloc(ctypes.byref(p), n, _ACL_MEM_MALLOC_HUGE_FIRST), "malloc")
        return p

    ad, wd, idd, od = malloc(A_BYTES), malloc(W_BYTES), malloc(IDX_BYTES), malloc(OUT_BYTES)

    def h2d(dst, arr, n):
        _chk(
            lib.aclrtMemcpy(
                dst, n, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), n, _ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "h2d",
        )

    h2d(ad, A, A_BYTES)
    h2d(wd, W, W_BYTES)
    h2d(idd, Idx, IDX_BYTES)
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(ad.value), int(wd.value), int(idd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros(E, dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    for p in (ad, wd, idd, od):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(oh - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("E", nargs="?", type=int, default=256)
    ap.add_argument("B", nargs="?", type=int, default=8)
    ap.add_argument("T", nargs="?", type=int, default=32)
    ap.add_argument("arm", nargs="?", default="keep_idx", choices=("keep_idx", "remat_idx"))
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    tag = f"sv4d_e{args.E}_b{args.B}_t{args.T}_{args.arm}"
    kernel = build_kernel(args.E, args.B, args.arm)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, args.E, args.B)


if __name__ == "__main__":
    raise SystemExit(main())
