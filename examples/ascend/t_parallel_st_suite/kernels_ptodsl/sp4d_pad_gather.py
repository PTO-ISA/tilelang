#!/usr/bin/env python3
"""ST-SP4d — PTO-DSL SIMD twin of Simt SP4 (pad gather, full-lane mask).

Simt twin: kernels/sp4_pad_gather.py
Primary tag: sp4d_e64_h128_t32_pad25

Gold (same as Simt / run_opsim_generic): ``Out[p,j] = Buf[p,j] if Expert[p] >= 0 else 0``.
Host synthesizes ~25% ``Expert[p] < 0`` holes (seed 104, same recipe as opsim).

Sensitivity: Simt may divergent-skip pad rows. This SIMD schedule does **not**
early-exit. Every ``Eexp × H`` lane is loaded and stored; the hole predicate
is a broadcast compare + ``vsel`` (live row vs zero).

Memory (Lok): Buf/Out rows are dense ``H=128`` (= 2×VL) and visited in loop
order, so continuous ``vload`` / ``vstore`` is aligned staging. ``Expert[p]``
is a size=1 scalar, not an address. No gather — the contrast is mask vs skip.
``Thr`` is a Simt tag mirror only (no thread launch in the explicit kernel).
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

VL = 64
EEXP = 64
H = 128
THR = 32
PAD_PCT = 25
SEED = 104

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _align64(n: int) -> int:
    return (n + 63) // 64 * 64


def synthesize(eexp: int, h: int, pad_pct: int):
    """Match run_opsim_generic sp4 host: ~pad_pct% of Expert slots set to -1."""
    rng = np.random.default_rng(SEED)
    buf = rng.standard_normal((eexp, h)).astype(np.float32)
    expert = np.arange(eexp, dtype=np.int32)
    n_pad = max(1, int(round(eexp * pad_pct / 100.0)))
    expert[rng.choice(eexp, size=n_pad, replace=False)] = np.int32(-1)
    gold = buf.copy()
    gold[expert < 0] = np.float32(0)
    return buf, expert, gold


def build_kernel(eexp: int, h: int, threads: int):
    buf_bytes = eexp * h * 4
    exp_bytes = eexp * 4
    out_bytes = eexp * h * 4
    off_exp = _align64(buf_bytes)
    off_out = _align64(off_exp + exp_bytes)
    name = f"SP4d_e{eexp}_h{h}_t{threads}"

    @pto.jit(
        name=name,
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(
        out_gm: pto.ptr(pto.f32, "gm"),
        buf_gm: pto.ptr(pto.f32, "gm"),
        exp_gm: pto.ptr(pto.i32, "gm"),
    ):
        buf_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        exp_ub = pto.castptr(pto.const(off_exp, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(buf_gm, buf_ub, 0, buf_bytes, nburst=(1, buf_bytes, buf_bytes))
        pto.mte_load(exp_gm, exp_ub, 0, exp_bytes, nburst=(1, exp_bytes, exp_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        zero = pto.vmi.vbrc(pto.f32(0.0), size=VL)
        # Expert >= 0  ⇔  Expert > -1 (integer). Predicate is lane-uniform.
        neg1 = pto.vmi.vbrc(pto.i32(-1), size=VL)

        # Full-lane mask: always walk every row. Do not branch on Expert.
        with pto.for_(0, eexp, step=1) as p:
            expert_b = pto.vmi.vbrc(pto.vmi.vload(exp_ub, p, size=1), size=VL)
            live = pto.vmi.vcmp(expert_b, neg1, mask, "gt")
            v0 = pto.vmi.vload(buf_ub, p * h + 0 * VL, size=VL)
            v1 = pto.vmi.vload(buf_ub, p * h + 1 * VL, size=VL)
            pto.vmi.vstore(pto.vmi.vsel(live, v0, zero), out_ub, p * h + 0 * VL, mask)
            pto.vmi.vstore(pto.vmi.vsel(live, v1, zero), out_ub, p * h + 1 * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


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
    lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def run_acl(eexp: int, h: int, threads: int, pad_pct: int, tag: str) -> int:
    buf, expert, gold = synthesize(eexp, h, pad_pct)
    buf_bytes = buf.nbytes
    exp_bytes = expert.nbytes
    out_bytes = gold.nbytes
    kernel = build_kernel(eexp, h, threads)

    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    def malloc(n):
        p = ctypes.c_void_p()
        _chk(lib.aclrtMalloc(ctypes.byref(p), n, _ACL_MEM_MALLOC_HUGE_FIRST), "malloc")
        return p

    def h2d(dst, arr, n):
        _chk(
            lib.aclrtMemcpy(
                dst,
                n,
                np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p),
                n,
                _ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "h2d",
        )

    bd, ed, od = malloc(buf_bytes), malloc(exp_bytes), malloc(out_bytes)
    h2d(bd, buf, buf_bytes)
    h2d(ed, expert, exp_bytes)
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(bd.value), int(ed.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    out = np.zeros((eexp, h), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(out.ctypes.data_as(ctypes.c_void_p), out_bytes, od, out_bytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    for p in (bd, ed, od):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    mx = float(np.max(np.abs(out - gold)))
    if not np.allclose(out, gold, rtol=1e-5, atol=1e-5):
        raise AssertionError(f"mismatch maxabs={mx}")
    print(f"PASS {tag} maxabs={mx:.6g} compile={cs:.3f}s launch={ls:.3f}s thr_mirror={threads}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="SP4d pad-gather SIMD mask twin")
    ap.add_argument("Eexp", nargs="?", type=int, default=EEXP)
    ap.add_argument("H", nargs="?", type=int, default=H)
    ap.add_argument("Thr", nargs="?", type=int, default=THR)
    ap.add_argument("pad", nargs="?", type=int, default=PAD_PCT)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.H != H:
        print(f"SP4d dense staging requires H={H} (2×VL); got H={args.H}", flush=True)
        return 2
    if args.Eexp <= 0 or args.pad < 0 or args.pad > 100:
        print(f"bad shape Eexp={args.Eexp} pad={args.pad}", flush=True)
        return 2
    tag = f"sp4d_e{args.Eexp}_h{args.H}_t{args.Thr}_pad{args.pad}"
    if args.emit_mlir:
        print(build_kernel(args.Eexp, args.H, args.Thr).compile().mlir_text())
        return 0
    return run_acl(args.Eexp, args.H, args.Thr, args.pad, tag)


if __name__ == "__main__":
    raise SystemExit(main())
