#!/usr/bin/env python3
"""ST-SV3d — PTO-DSL Layer D twin of Simt SV3 (GEMV Acc KEEP vs split).

Simt: kernels/sv3_gemv_partial_keep.py
Tags: sv3d_m{M}_vl{VL}_k{K}_t{T}_{keep|split_cm{CM}}

Math: Acc[M, VL] += A[k, M, VL] * x[k]  (broadcast), then Out = Acc.
DSL I/O: f32 A/x/Out (Simt uses fp16 A/x, fp32 Acc/Out) — same Layer-D
convention as SV2d. Acc always fp32 vregs.

Arms:
  keep       (3): Acc[M] VL-vregs stay-alive across K; A/x reload each K
  split_cm16 (1+3): Acc[CM] live across K; flush to UB per chunk (O(NC) not O(K))

No AABBCC / vf_fuse. Mapping modes ∈ {(1),(3)}.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

VL_DEFAULT = 64
T_DEFAULT = 32

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _layout(M, VL, K):
    a_bytes = K * M * VL * 4
    x_bytes = K * 4
    out_bytes = M * VL * 4
    off_x = a_bytes
    off_out = a_bytes + x_bytes
    return a_bytes, x_bytes, out_bytes, off_x, off_out


def _kernel_keep(M: int, VL: int, K: int):
    a_bytes, x_bytes, out_bytes, off_x, off_out = _layout(M, VL, K)

    @pto.jit(
        name=f"SV3d_m{M}_vl{VL}_k{K}_keep",
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
        x_gm: pto.ptr(pto.f32, "gm"),
    ):
        a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        x_ub = pto.castptr(pto.const(off_x, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(a_gm, a_ub, 0, a_bytes, nburst=(1, a_bytes, a_bytes))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        # Acc KEEP across K — M VL-wide vregs
        acc = [pto.vmi.vbrc(pto.f32(0.0), size=VL) for _ in range(M)]

        for k in range(K):
            # x reload each K (scalar → broadcast)
            xk = pto.vmi.vbrc(pto.vmi.vload(x_ub, k, size=1), size=VL)
            base = k * M * VL
            for m in range(M):
                a = pto.vmi.vload(a_ub, base + m * VL, size=VL)  # A reload
                acc[m] = pto.vmi.vadd(acc[m], pto.vmi.vmul(a, xk, mask), mask)

        for m in range(M):
            pto.vmi.vstore(acc[m], out_ub, m * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel_split(M: int, VL: int, K: int, CM: int):
    a_bytes, x_bytes, out_bytes, off_x, off_out = _layout(M, VL, K)
    NC = (M + CM - 1) // CM

    @pto.jit(
        name=f"SV3d_m{M}_vl{VL}_k{K}_split_cm{CM}",
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
        x_gm: pto.ptr(pto.f32, "gm"),
    ):
        a_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        x_ub = pto.castptr(pto.const(off_x, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(a_gm, a_ub, 0, a_bytes, nburst=(1, a_bytes, a_bytes))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)

        for c in range(NC):
            # Acc tile CM×VL KEEP across K; flush once per chunk
            acc = [pto.vmi.vbrc(pto.f32(0.0), size=VL) for _ in range(CM)]
            for k in range(K):
                xk = pto.vmi.vbrc(pto.vmi.vload(x_ub, k, size=1), size=VL)
                for i in range(CM):
                    row = c * CM + i
                    if row < M:
                        a = pto.vmi.vload(a_ub, k * M * VL + row * VL, size=VL)
                        acc[i] = pto.vmi.vadd(acc[i], pto.vmi.vmul(a, xk, mask), mask)
            for i in range(CM):
                row = c * CM + i
                if row < M:
                    pto.vmi.vstore(acc[i], out_ub, row * VL, mask)
            # chunk boundary visibility (O(NC) membar tax)
            if c + 1 < NC:
                pto.mem_bar("VST_VLD")

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def reference(A, X, M, VL, K):
    Acc = np.zeros((M, VL), dtype=np.float32)
    for k in range(K):
        Acc += A[k] * float(X[k])
    return Acc


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


def run_acl(kernel, tag: str, M: int, VL: int, K: int) -> int:
    a_bytes, x_bytes, out_bytes, _, _ = _layout(M, VL, K)
    rng = np.random.RandomState(2026)
    A = rng.uniform(-1, 1, (K, M, VL)).astype(np.float32)
    X = rng.uniform(-1, 1, K).astype(np.float32)
    ref = reference(A, X, M, VL, K)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    def malloc(n):
        p = ctypes.c_void_p()
        _chk(lib.aclrtMalloc(ctypes.byref(p), n, _ACL_MEM_MALLOC_HUGE_FIRST), "malloc")
        return p

    ad, xd, od = malloc(a_bytes), malloc(x_bytes), malloc(out_bytes)

    def h2d(dst, arr, n):
        _chk(
            lib.aclrtMemcpy(
                dst, n, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), n,
                _ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "h2d",
        )

    h2d(ad, A, a_bytes)
    h2d(xd, X, x_bytes)
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(ad.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros((M, VL), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), out_bytes, od, out_bytes,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    for p in (ad, xd, od):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    maxabs = float(np.max(np.abs(oh - ref)))
    if not np.allclose(oh, ref, atol=1e-3, rtol=1e-3):
        raise AssertionError(f"mismatch max_diff={maxabs}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s maxabs={maxabs:.3e}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("M", nargs="?", type=int, default=32)
    ap.add_argument("VL", nargs="?", type=int, default=VL_DEFAULT)
    ap.add_argument("K", nargs="?", type=int, default=16)
    ap.add_argument("T", nargs="?", type=int, default=T_DEFAULT)
    ap.add_argument("arm", nargs="?", default="keep", choices=("keep", "split_cm16", "split"))
    ap.add_argument("CM", nargs="?", type=int, default=16)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    arm = args.arm
    CM = args.CM
    if arm == "split":
        arm = f"split_cm{CM}"
    if arm.startswith("split_cm"):
        CM = int(arm.split("split_cm", 1)[1])
        tag = f"sv3d_m{args.M}_vl{args.VL}_k{args.K}_t{args.T}_split_cm{CM}"
        kernel = _kernel_split(args.M, args.VL, args.K, CM)
    else:
        tag = f"sv3d_m{args.M}_vl{args.VL}_k{args.K}_t{args.T}_keep"
        kernel = _kernel_keep(args.M, args.VL, args.K)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, args.M, args.VL, args.K)


if __name__ == "__main__":
    raise SystemExit(main())
