#!/usr/bin/env python3
"""ST-SV6d — PTO-DSL Layer D twin of Simt SV5 (mid-G reduce RF ladder).

Tags: sv6d_r{R}_c{C}_g{G}_t{T}_{keep_in_warp|reload}
VMI: only size∈{1,64} for vload/vstore (size=8 hits ensure_layout).
keep_in_warp: G size-1 vregs force-live during absmax.
reload: vload size-1 each serial(G) step.
Y: VL=64 broadcast slot per (i,g) for 32B-aligned VST.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

EPS = 1e-6
T_DEFAULT = 32
VL = 64

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _layout(R, C, G):
    CG = C // G
    x_bytes = R * C * 4
    y_bytes = R * CG * VL * 4
    return CG, x_bytes, y_bytes, x_bytes


def _kernel_keep_in_warp(R: int, C: int, G: int):
    CG, x_bytes, y_bytes, off_y = _layout(R, C, G)

    @pto.jit(
        name=f"SV6d_r{R}_c{C}_g{G}_keep_in_warp",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(y_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        y_ub = pto.castptr(pto.const(off_y, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask1 = pto.vmi.create_mask(1, size=1)
        mask_vl = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=1)
        zero = pto.vmi.vbrc(pto.f32(0.0), size=1)

        with pto.for_(0, R, step=1) as i:
            for g in range(CG):
                # KEEP: force-live G size-1 vregs
                kept = []
                for t in range(G):
                    v = pto.vmi.vload(x_ub, i * C + g * G + t, size=1)
                    kept.append(pto.vmi.vadd(v, pto.vmi.vmul(v, zero, mask1), mask1))
                m = pto.vmi.vbrc(pto.f32(0.0), size=1)
                for t in range(G):
                    vv = kept[t]
                    absv = pto.vmi.vmax(vv, pto.vmi.vmul(vv, neg1, mask1), mask1)
                    m = pto.vmi.vmax(m, absv, mask1)
                m = pto.vmi.vmax(m, pto.vmi.vbrc(pto.f32(EPS), size=1), mask1)
                inv = pto.vmi.vdiv(pto.vmi.vbrc(pto.f32(1.0), size=1), m, mask1)
                pto.vmi.vstore(pto.vmi.vbrc(inv, size=VL), y_ub, (i * CG + g) * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(y_ub, y_gm, y_bytes, nburst=(1, y_bytes, y_bytes))

    return kernel


def _kernel_reload(R: int, C: int, G: int):
    CG, x_bytes, y_bytes, off_y = _layout(R, C, G)

    @pto.jit(
        name=f"SV6d_r{R}_c{C}_g{G}_reload",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(y_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        y_ub = pto.castptr(pto.const(off_y, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask1 = pto.vmi.create_mask(1, size=1)
        mask_vl = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=1)

        with pto.for_(0, R, step=1) as i:
            for g in range(CG):
                m = pto.vmi.vbrc(pto.f32(0.0), size=1)
                for t in range(G):
                    v = pto.vmi.vload(x_ub, i * C + g * G + t, size=1)
                    absv = pto.vmi.vmax(v, pto.vmi.vmul(v, neg1, mask1), mask1)
                    m = pto.vmi.vmax(m, absv, mask1)
                m = pto.vmi.vmax(m, pto.vmi.vbrc(pto.f32(EPS), size=1), mask1)
                inv = pto.vmi.vdiv(pto.vmi.vbrc(pto.f32(1.0), size=1), m, mask1)
                pto.vmi.vstore(pto.vmi.vbrc(inv, size=VL), y_ub, (i * CG + g) * VL, mask_vl)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(y_ub, y_gm, y_bytes, nburst=(1, y_bytes, y_bytes))

    return kernel


def reference(X, R, C, G):
    CG = C // G
    Y = np.empty((R, CG), dtype=np.float32)
    for i in range(R):
        for g in range(CG):
            chunk = X[i, g * G : (g + 1) * G].astype(np.float32)
            m = max(float(np.max(np.abs(chunk))), EPS)
            Y[i, g] = np.float32(1.0 / m)
    return Y


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


def run_acl(kernel, tag, R, C, G) -> int:
    CG, x_bytes, y_bytes, _ = _layout(R, C, G)
    rng = np.random.RandomState(6)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref = reference(X, R, C, G)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p()
    yd = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(yd), y_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "my")
    _chk(
        lib.aclrtMemcpy(
            xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes,
            _ACL_MEMCPY_HOST_TO_DEVICE,
        ),
        "h2d",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(yd.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    raw = np.zeros((R, CG, VL), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(raw.ctypes.data_as(ctypes.c_void_p), y_bytes, yd, y_bytes,
                        _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    yh = raw[:, :, 0].copy()
    lib.aclrtFree(xd)
    lib.aclrtFree(yd)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    maxabs = float(np.max(np.abs(yh - ref)))
    if not np.allclose(yh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={maxabs}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s maxabs={maxabs:.3e}")
    return 0


_ARM_BUILDERS = {
    "keep_in_warp": _kernel_keep_in_warp,
    "reload": _kernel_reload,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("R", nargs="?", type=int, default=64)
    ap.add_argument("C", nargs="?", type=int, default=128)
    ap.add_argument("G", nargs="?", type=int, default=32)
    ap.add_argument("T", nargs="?", type=int, default=T_DEFAULT)
    ap.add_argument("arm", nargs="?", default="keep_in_warp", choices=tuple(_ARM_BUILDERS))
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.C % args.G != 0:
        raise SystemExit(f"C={args.C} must be divisible by G={args.G}")
    tag = f"sv6d_r{args.R}_c{args.C}_g{args.G}_t{args.T}_{args.arm}"
    kernel = _ARM_BUILDERS[args.arm](args.R, args.C, args.G)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, args.R, args.C, args.G)


if __name__ == "__main__":
    raise SystemExit(main())
