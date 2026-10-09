#!/usr/bin/env python3
"""ST-SP2d — PTO-DSL twin of Simt SP2 (indexed gather→reduce).

Simt twin: kernels/sp2_dual_gather_wreduce.py
Primary tags:
  sp2d_t32_k2_h128_g32_t32_{sf0|sf1}_{w0|w1}_{keep_acc|remat_acc}

Sensitivity (Lok 2026-10-07 / 10-08):
  1. Acc KEEP (VRF) vs remat (UB RMW) on indexed gather→reduce
  2. Under Acc KEEP: sf0_w0 vs sf1_w1 compute tax

Memory atoms (Lok 2026-10-08): **always gather / scatter** for indexed
Vexp[pos] / Out[t] accesses so camodel check_addr_aligned cannot fire on
Hs-stride continuous RV_VLD/RV_VSTS. Aligned VL continuous is OK only for
the Acc UB staging (H=128 = 2×VL) and the optional Sf expand scratch.

No AABBCC / vf_fuse. Legacy keep_pos/remat_pos not in *d (Simt-only history).
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto
from ptodsl import scalar

T_ = 32
K = 2
H = 128
G = 32
THR = 32
Hs = H // G
Eexp = T_ * K
VL = 64
NCH = H // VL  # 2

VEXP_BYTES = Eexp * H * 4
SF_BYTES = Eexp * Hs * 4
SF_FULL_BYTES = Eexp * H * 4
W_BYTES = T_ * K * 4
POS_BYTES = T_ * K * 4
OUT_BYTES = T_ * H * 4
ACC_BYTES = H * 4
POS_ROW_BYTES = K * VL * 4

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _layout(with_sf: bool):
    off = 0
    off_vexp = off
    off = off_vexp + VEXP_BYTES
    off_sf = off
    off = off_sf + SF_BYTES
    off_sf_full = off
    off = off_sf_full + (SF_FULL_BYTES if with_sf else 0)
    off_w = off
    off = off_w + W_BYTES
    off_pos = off
    off = off_pos + POS_BYTES
    off_out = off
    off = off_out + OUT_BYTES
    off_acc = off
    off = off_acc + ACC_BYTES
    off_pos_row = off
    return dict(
        off_vexp=off_vexp,
        off_sf=off_sf,
        off_sf_full=off_sf_full,
        off_w=off_w,
        off_pos=off_pos,
        off_out=off_out,
        off_acc=off_acc,
        off_pos_row=off_pos_row,
    )


def build_kernel(with_sf: bool, with_w: bool, remat_acc: bool):
    L = _layout(with_sf)
    arm = ("remat_acc" if remat_acc else "keep_acc")
    sf_tag = "sf1" if with_sf else "sf0"
    w_tag = "w1" if with_w else "w0"
    name = f"SP2d_t{T_}_k{K}_h{H}_g{G}_{sf_tag}_{w_tag}_{arm}"

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
        vexp_gm: pto.ptr(pto.f32, "gm"),
        sf_gm: pto.ptr(pto.f32, "gm"),
        w_gm: pto.ptr(pto.f32, "gm"),
        pos_gm: pto.ptr(pto.i32, "gm"),
    ):
        vexp_ub = pto.castptr(pto.const(L["off_vexp"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        sf_ub = pto.castptr(pto.const(L["off_sf"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        w_ub = pto.castptr(pto.const(L["off_w"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pos_ub = pto.castptr(pto.const(L["off_pos"], dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        out_ub = pto.castptr(pto.const(L["off_out"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        acc_ub = pto.castptr(pto.const(L["off_acc"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pos_row = pto.castptr(pto.const(L["off_pos_row"], dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        sf_full = pto.castptr(pto.const(L["off_sf_full"], dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(vexp_gm, vexp_ub, 0, VEXP_BYTES, nburst=(1, VEXP_BYTES, VEXP_BYTES))
        pto.mte_load(sf_gm, sf_ub, 0, SF_BYTES, nburst=(1, SF_BYTES, SF_BYTES))
        pto.mte_load(w_gm, w_ub, 0, W_BYTES, nburst=(1, W_BYTES, W_BYTES))
        pto.mte_load(pos_gm, pos_ub, 0, POS_BYTES, nburst=(1, POS_BYTES, POS_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        mask_hs = pto.vmi.create_mask(Hs, size=VL)
        mask_1 = pto.vmi.create_mask(1, size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)
        zero_f = pto.vmi.vbrc(pto.f32(0.0), size=VL)
        one_f = pto.vmi.vbrc(pto.f32(1.0), size=VL)

        # Optional: expand Sf[Eexp,Hs] → sf_full[Eexp,H] via gather+brc+scatter
        if with_sf:
            with pto.for_(0, Eexp, step=1) as p:
                p_i32 = scalar.index_cast(pto.i32, p)
                for g in range(Hs):
                    # size=1 load → 1-lane → vbrc (vgather returns VL lanes; vbrc rejects that)
                    s1 = pto.vmi.vload(sf_ub, p * Hs + g, size=1)
                    s_brc = pto.vmi.vbrc(s1, size=VL)
                    base_full = pto.vmi.vbrc(p_i32 * H + g * G, size=VL)
                    offs_full = pto.vmi.vadd(base_full, arange, mask)
                    mask_g = pto.vmi.create_mask(G, size=VL)
                    pto.vmi.vscatter(s_brc, sf_full, offs_full, mask_g)

        with pto.for_(0, T_, step=1) as t:
            t_i32 = scalar.index_cast(pto.i32, t)
            # freeze Pos[t,:] into VL-padded slots (aligned continuous)
            for k in range(K):
                pv = pto.vmi.vload(pos_ub, t * K + k, size=1)
                pto.vmi.vstore(pto.vmi.vbrc(pv, size=VL), pos_row, k * VL, mask_1)

            if remat_acc:
                # Acc remat: zero shared staging (aligned VL continuous OK for H)
                pto.vmi.vstore(zero_f, acc_ub, 0 * VL, mask)
                pto.vmi.vstore(zero_f, acc_ub, 1 * VL, mask)
            else:
                acc0 = zero_f
                acc1 = zero_f

            for k in range(K):
                pos_v = pto.vmi.vload(pos_row, k * VL, size=1)
                # wk scalar
                if with_w:
                    wk_v = pto.vmi.vload(w_ub, t * K + k, size=1)
                    wk = pto.vmi.vbrc(wk_v, size=VL)
                else:
                    wk = one_f

                # gather Vexp[pos, 0:64] and [64:128]
                # offs = pos*H + arange / +64
                pos_b = pto.vmi.vbrc(pos_v, size=VL)
                base0 = pto.vmi.vmul(pos_b, pto.vmi.vbrc(pto.i32(H), size=VL), mask)
                offs0 = pto.vmi.vadd(base0, arange, mask)
                offs1 = pto.vmi.vadd(offs0, pto.vmi.vbrc(pto.i32(VL), size=VL), mask)
                v0 = pto.vmi.vgather(vexp_ub, offs0, mask)
                v1 = pto.vmi.vgather(vexp_ub, offs1, mask)

                if with_sf:
                    s0 = pto.vmi.vgather(sf_full, offs0, mask)
                    s1 = pto.vmi.vgather(sf_full, offs1, mask)
                    term0 = pto.vmi.vmul(v0, pto.vmi.vmul(wk, s0, mask), mask)
                    term1 = pto.vmi.vmul(v1, pto.vmi.vmul(wk, s1, mask), mask)
                else:
                    term0 = pto.vmi.vmul(v0, wk, mask)
                    term1 = pto.vmi.vmul(v1, wk, mask)

                if remat_acc:
                    old0 = pto.vmi.vload(acc_ub, 0 * VL, size=VL)
                    old1 = pto.vmi.vload(acc_ub, 1 * VL, size=VL)
                    pto.vmi.vstore(pto.vmi.vadd(old0, term0, mask), acc_ub, 0 * VL, mask)
                    pto.vmi.vstore(pto.vmi.vadd(old1, term1, mask), acc_ub, 1 * VL, mask)
                else:
                    acc0 = pto.vmi.vadd(acc0, term0, mask)
                    acc1 = pto.vmi.vadd(acc1, term1, mask)

            # scatter Out[t,:] — indexed by t*H + arange (always scatter per Lok)
            base_out = pto.vmi.vbrc(t_i32 * H, size=VL)
            outs0 = pto.vmi.vadd(base_out, arange, mask)
            outs1 = pto.vmi.vadd(outs0, pto.vmi.vbrc(pto.i32(VL), size=VL), mask)
            if remat_acc:
                a0 = pto.vmi.vload(acc_ub, 0 * VL, size=VL)
                a1 = pto.vmi.vload(acc_ub, 1 * VL, size=VL)
                pto.vmi.vscatter(a0, out_ub, outs0, mask)
                pto.vmi.vscatter(a1, out_ub, outs1, mask)
            else:
                pto.vmi.vscatter(acc0, out_ub, outs0, mask)
                pto.vmi.vscatter(acc1, out_ub, outs1, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, OUT_BYTES, nburst=(1, OUT_BYTES, OUT_BYTES))

    return kernel, name


def reference(Vexp, Sf, W, Pos, with_sf, with_w):
    Out = np.zeros((T_, H), dtype=np.float32)
    if with_sf:
        sf_full = np.repeat(Sf, G, axis=1)
    for t in range(T_):
        acc = np.zeros(H, dtype=np.float32)
        for k in range(K):
            pos = int(Pos[t, k])
            if pos < 0:
                continue
            wk = float(W[t, k]) if with_w else 1.0
            if with_sf:
                acc += Vexp[pos] * (wk * sf_full[pos])
            else:
                acc += Vexp[pos] * wk
        Out[t] = acc
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


def run_acl(with_sf: bool, with_w: bool, remat_acc: bool, tag: str) -> int:
    kernel, _ = build_kernel(with_sf, with_w, remat_acc)
    rng = np.random.RandomState(2026)
    Vexp = rng.uniform(-1, 1, (Eexp, H)).astype(np.float32)
    Sf = rng.uniform(0.25, 4.0, (Eexp, Hs)).astype(np.float32)
    W = rng.uniform(0.5, 1.5, (T_, K)).astype(np.float32)
    Pos = np.arange(Eexp, dtype=np.int32).reshape(T_, K)
    Pos[0, 0] = -1
    ref = reference(Vexp, Sf, W, Pos, with_sf, with_w)

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
                dst, n, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), n, _ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "h2d",
        )

    vd, sfd, wd, pd, od = (
        malloc(VEXP_BYTES),
        malloc(SF_BYTES),
        malloc(W_BYTES),
        malloc(POS_BYTES),
        malloc(OUT_BYTES),
    )
    h2d(vd, Vexp, VEXP_BYTES)
    h2d(sfd, Sf, SF_BYTES)
    h2d(wd, W, W_BYTES)
    h2d(pd, Pos, POS_BYTES)

    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(od.value), int(vd.value), int(sfd.value), int(wd.value), int(pd.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    out = np.zeros((T_, H), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(out.ctypes.data_as(ctypes.c_void_p), OUT_BYTES, od, OUT_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    for p in (vd, sfd, wd, pd, od):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    if not np.allclose(out, ref, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"mismatch max_diff={float(np.max(np.abs(out - ref)))}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("T", nargs="?", type=int, default=T_)
    ap.add_argument("K", nargs="?", type=int, default=K)
    ap.add_argument("H", nargs="?", type=int, default=H)
    ap.add_argument("G", nargs="?", type=int, default=G)
    ap.add_argument("Thr", nargs="?", type=int, default=THR)
    ap.add_argument("--sf", default="sf1", choices=("sf0", "sf1"))
    ap.add_argument("--w", default="w1", choices=("w0", "w1"))
    ap.add_argument("--sched", default="keep_acc", choices=("keep_acc", "remat_acc"))
    # positional aliases after Thr: sf w sched (opsim harness)
    ap.add_argument("sf_pos", nargs="?", default=None, choices=("sf0", "sf1"))
    ap.add_argument("w_pos", nargs="?", default=None, choices=("w0", "w1"))
    ap.add_argument("sched_pos", nargs="?", default=None, choices=("keep_acc", "remat_acc"))
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    sf = args.sf_pos or args.sf
    w = args.w_pos or args.w
    sched = args.sched_pos or args.sched
    with_sf = sf == "sf1"
    with_w = w == "w1"
    remat_acc = sched == "remat_acc"
    tag = f"sp2d_t{T_}_k{K}_h{H}_g{G}_t{THR}_{sf}_{w}_{sched}"
    if args.emit_mlir:
        k, _ = build_kernel(with_sf, with_w, remat_acc)
        print(k.compile().mlir_text())
        return 0
    return run_acl(with_sf, with_w, remat_acc, tag)


if __name__ == "__main__":
    raise SystemExit(main())
