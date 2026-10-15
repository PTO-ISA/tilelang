#!/usr/bin/env python3
"""ST-SV2d — fold scale ALWAYS KEEP; input KEEP vs STREAM by R.

Ascend950PR_9599: VL=256B = 64 f32. C multiple of VL, NCH=C/VL, T=32.

Gold (fold along C): per-lane scale via vmax only:
  scale[lane] = max(EPS, max_ch abs(X[i, ch*VL + lane]))
  out = x / scale
Scale is a VL-wide vreg from the fold — ALWAYS KEEP in RF (never remat).

**Contract (Lok 2026-09-29):**
- Scale ALWAYS KEEP. Do NOT use scale remat as the primary turning point.
- Sensitivity is **input (rows) KEEP vs STREAM** by R:
  - R≈16 ``input_keep``: KEEP all row VL chunks in RF across fold+/scale.
  - R≥32 ``input_stream`` / ``fold_scale_keep``: STREAM rows (vload per use);
    scale still KEEP.
- ``fold_scale_reload`` DEMOTEd (legacy scale-remat foil; not primary).

RF math (input_keep all-rows live): R*NCH*VL_BYTES (+1 scale).
  R16×C64 NCH1 ≈ 4096+256 fits under 6144; R32×C64 ≈ 8192 spills.

I/O: f32. mode=explicit, insert_sync=False, ast_rewrite=False.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto

T = 32
VL = 64
EPS = 1e-6

VF_STACK_ASSUME_B = 6144
VL_BYTES = VL * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _rf_math(R: int, C: int):
    nch = C // VL
    stream_keep_approx_B = (1 + 2) * VL_BYTES  # 1 scale + ~2 transient chunks
    input_keep_vregs = R * nch  # all row chunks force-live
    input_keep_bytes = input_keep_vregs * VL_BYTES
    return {
        "R": R,
        "C": C,
        "NCH": nch,
        "STREAM_KEEP_APPROX_B": stream_keep_approx_B,
        "INPUT_KEEP_VREGS": input_keep_vregs,
        "INPUT_KEEP_BYTES": input_keep_bytes,
        "input_keep_blows": input_keep_bytes > VF_STACK_ASSUME_B,
    }


def _offsets(R: int, C: int):
    """UB layout: [x | out | scale_scratch(VL)]. scratch only for demoted reload."""
    x_bytes = R * C * 4
    out_bytes = R * C * 4
    off_out = x_bytes
    off_scale = x_bytes + out_bytes
    return x_bytes, out_bytes, off_out, off_scale


def _kernel_input_stream(R: int, C: int):
    """PRIMARY large-R: STREAM rows; KEEP scale VL in RF per row."""
    nch = C // VL
    x_bytes, out_bytes, off_out, _ = _offsets(R, C)

    @pto.jit(
        name=f"SV2d_r{R}_c{C}_input_stream",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
        eps_v = pto.vmi.vbrc(pto.f32(EPS), size=VL)

        with pto.for_(0, R, step=1) as i:
            scale = eps_v
            for ch in range(nch):
                c = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)
                absv = pto.vmi.vmax(c, pto.vmi.vmul(c, neg1, mask), mask)
                scale = pto.vmi.vmax(scale, absv, mask)  # KEEP scale
            scale = pto.vmi.vmax(scale, eps_v, mask)
            for ch in range(nch):
                c = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)  # STREAM reload
                pto.vmi.vstore(pto.vmi.vdiv(c, scale, mask), out_ub, i * C + ch * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel_input_keep(R: int, C: int):
    """Small-R: KEEP all row VL chunks in RF; scale ALWAYS KEEP per row.

    Force-live R*NCH row vregs across fold + consumer. Intended R≈16 C=64
    (fits); R≥32 expected spill / must use input_stream.
    """
    nch = C // VL
    x_bytes, out_bytes, off_out, _ = _offsets(R, C)

    @pto.jit(
        name=f"SV2d_r{R}_c{C}_input_keep",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
        eps_v = pto.vmi.vbrc(pto.f32(EPS), size=VL)

        # KEEP all row chunks force-live (R * NCH vregs)
        rows = []
        for i in range(R):
            for ch in range(nch):
                c = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)
                # cook so compiler cannot plain-remat past cliff
                rows.append(pto.vmi.vadd(c, pto.vmi.vmul(c, pto.vmi.vbrc(pto.f32(0.0), size=VL), mask), mask))

        for i in range(R):
            scale = eps_v
            for ch in range(nch):
                c = rows[i * nch + ch]
                absv = pto.vmi.vmax(c, pto.vmi.vmul(c, neg1, mask), mask)
                scale = pto.vmi.vmax(scale, absv, mask)  # KEEP scale
            scale = pto.vmi.vmax(scale, eps_v, mask)
            for ch in range(nch):
                c = rows[i * nch + ch]  # from KEEP, not UB reload
                pto.vmi.vstore(pto.vmi.vdiv(c, scale, mask), out_ub, i * C + ch * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


def _kernel_fold_scale_reload(R: int, C: int):
    """DEMOTEd foil: scale remat via UB+mem_bar. Not primary ST contrast."""
    nch = C // VL
    x_bytes, out_bytes, off_out, off_scale = _offsets(R, C)

    @pto.jit(
        name=f"SV2d_r{R}_c{C}_fold_scale_reload",
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel(out_gm: pto.ptr(pto.f32, "gm"), x_gm: pto.ptr(pto.f32, "gm")):
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        scale_ub = pto.castptr(pto.const(off_scale, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(x_gm, x_ub, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        neg1 = pto.vmi.vbrc(pto.f32(-1.0), size=VL)
        eps_v = pto.vmi.vbrc(pto.f32(EPS), size=VL)

        with pto.for_(0, R, step=1) as i:
            scale = eps_v
            for ch in range(nch):
                c = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)
                absv = pto.vmi.vmax(c, pto.vmi.vmul(c, neg1, mask), mask)
                scale = pto.vmi.vmax(scale, absv, mask)
            scale = pto.vmi.vmax(scale, eps_v, mask)
            pto.vmi.vstore(scale, scale_ub, 0, mask)
            pto.mem_bar("VST_VLD")
            scale = pto.vmi.vload(scale_ub, 0, size=VL)
            for ch in range(nch):
                c = pto.vmi.vload(x_ub, i * C + ch * VL, size=VL)
                pto.vmi.vstore(pto.vmi.vdiv(c, scale, mask), out_ub, i * C + ch * VL, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel


# Alias: fold_scale_keep == input_stream (scale KEEP + row STREAM)
def _kernel_fold_scale_keep(R: int, C: int):
    return _kernel_input_stream(R, C)


def reference(X, R: int, C: int, arm: str):
    """Per-lane vmax fold (NOT scalar row max / vcmax). Same gold all arms."""
    nch = C // VL
    Out = np.empty_like(X)
    for i in range(R):
        scale = np.full(VL, EPS, dtype=np.float32)
        for ch in range(nch):
            chunk = X[i, ch * VL : (ch + 1) * VL].astype(np.float32)
            scale = np.maximum(scale, np.abs(chunk))
        scale = np.maximum(scale, EPS)
        for ch in range(nch):
            chunk = X[i, ch * VL : (ch + 1) * VL].astype(np.float32)
            Out[i, ch * VL : (ch + 1) * VL] = (chunk / scale).astype(np.float32)
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


def run_acl(kernel, tag: str, R: int, C: int, arm: str) -> int:
    x_bytes, out_bytes, _, _ = _offsets(R, C)
    rng = np.random.RandomState(2026)
    X = rng.uniform(-3, 3, (R, C)).astype(np.float32)
    ref = reference(X, R, C, arm)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    xd = ctypes.c_void_p()
    od = ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), out_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(
        lib.aclrtMemcpy(
            xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes, _ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "h2d",
    )
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(xd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    oh = np.zeros((R, C), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(oh.ctypes.data_as(ctypes.c_void_p), out_bytes, od, out_bytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    lib.aclrtFree(xd)
    lib.aclrtFree(od)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(oh, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={np.max(np.abs(oh - ref))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


_ARM_BUILDERS = {
    "input_keep": _kernel_input_keep,
    "input_stream": _kernel_input_stream,
    "fold_scale_keep": _kernel_fold_scale_keep,  # alias → input_stream
    "fold_scale_reload": _kernel_fold_scale_reload,  # demoted
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("R", nargs="?", type=int, default=32)
    ap.add_argument("C", nargs="?", type=int, default=64)
    ap.add_argument("T", nargs="?", type=int, default=T)
    ap.add_argument(
        "arm",
        nargs="?",
        default="input_stream",
        choices=tuple(_ARM_BUILDERS.keys()),
    )
    ap.add_argument("--emit-mlir", action="store_true")
    ap.add_argument("--rf-math", action="store_true")
    args = ap.parse_args(argv)
    Rr, C = args.R, args.C
    if C % VL != 0:
        raise SystemExit(f"C must be multiple of VL={VL}, got {C}")
    m = _rf_math(Rr, C)
    if args.rf_math:
        print(
            f"RF_MATH VL={VL} f32 VL_BYTES={VL_BYTES} R={Rr} C={C} NCH={m['NCH']} "
            f"STREAM_KEEP_APPROX_B={m['STREAM_KEEP_APPROX_B']} "
            f"INPUT_KEEP_VREGS={m['INPUT_KEEP_VREGS']} INPUT_KEEP_BYTES={m['INPUT_KEEP_BYTES']} "
            f"VF_STACK_ASSUME_B={VF_STACK_ASSUME_B} "
            f"input_keep_blows={m['input_keep_blows']}"
        )
        return 0
    tag = f"sv2d_r{Rr}_c{C}_t{T}_{args.arm}"
    kernel = _ARM_BUILDERS[args.arm](Rr, C)
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(kernel, tag, Rr, C, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
