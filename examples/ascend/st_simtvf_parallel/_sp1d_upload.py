#!/usr/bin/env python3
"""ST-SP1d — PTO-DSL explicit twin of Simt SP1 keep_pos (dual scatter V,Sf).

Simt twin: kernels/sp1_dual_scatter_vsf.py arm=keep_pos
Primary tag: sp1d_t32_k2_h128_g32_t32_keep_pos

Fidelity:
  - mapping (3) stay-alive: V/Sf row loaded once per token, live across K
  - pos_row UB snapshot (shared keep)
  - scatter via pto.for_(Eexp) + vcmps(pos,p) + vsel (not Python-unrolled T×K×Eexp)
  - V: VL=64 chunks into vexp_ub
  - Sf: vgather→VL, update VL-padded sfexp_pad, pack via size=1 vload/vstore
  - keep_pos only; no AABBCC / vf_fuse
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

V_BYTES = T_ * H * 4
SF_BYTES = T_ * Hs * 4
POS_BYTES = T_ * K * 4
EXP_BYTES = Eexp * 4
VEXP_BYTES = Eexp * H * 4
SFEXP_BYTES = Eexp * Hs * 4
SFEXP_PAD_BYTES = Eexp * VL * 4
POS_ROW_BYTES = K * 4

OFF_SF = V_BYTES
OFF_POS = OFF_SF + SF_BYTES
OFF_EXP = OFF_POS + POS_BYTES
OFF_VEXP = OFF_EXP + EXP_BYTES
OFF_SFEXP = OFF_VEXP + VEXP_BYTES
# VL stores require aligned UB base; place pad before the tiny pos_row snapshot.
OFF_SFEXP_PAD = OFF_SFEXP + SFEXP_BYTES
OFF_POS_ROW = OFF_SFEXP_PAD + SFEXP_PAD_BYTES

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


@pto.jit(
    name="SP1d_t32_k2_h128_g32_keep_pos",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def kernel(
    vexp_gm: pto.ptr(pto.f32, "gm"),
    sfexp_gm: pto.ptr(pto.f32, "gm"),
    v_gm: pto.ptr(pto.f32, "gm"),
    sf_gm: pto.ptr(pto.f32, "gm"),
    pos_gm: pto.ptr(pto.i32, "gm"),
    exp_gm: pto.ptr(pto.i32, "gm"),
):
    v_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    sf_ub = pto.castptr(pto.const(OFF_SF, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pos_ub = pto.castptr(pto.const(OFF_POS, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
    exp_ub = pto.castptr(pto.const(OFF_EXP, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
    vexp_ub = pto.castptr(pto.const(OFF_VEXP, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    sfexp_ub = pto.castptr(pto.const(OFF_SFEXP, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pos_row = pto.castptr(pto.const(OFF_POS_ROW, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
    sfexp_pad = pto.castptr(pto.const(OFF_SFEXP_PAD, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

    pto.mte_load(v_gm, v_ub, 0, V_BYTES, nburst=(1, V_BYTES, V_BYTES))
    pto.mte_load(sf_gm, sf_ub, 0, SF_BYTES, nburst=(1, SF_BYTES, SF_BYTES))
    pto.mte_load(pos_gm, pos_ub, 0, POS_BYTES, nburst=(1, POS_BYTES, POS_BYTES))
    pto.mte_load(exp_gm, exp_ub, 0, EXP_BYTES, nburst=(1, EXP_BYTES, EXP_BYTES))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask_h = pto.vmi.create_mask(VL, size=VL)
    mask_hs_vl = pto.vmi.create_mask(Hs, size=VL)
    zero_f = pto.vmi.vbrc(pto.f32(0.0), size=VL)
    arange = pto.vmi.vci(pto.i32(0), size=VL)
    _ = exp_ub

    with pto.for_(0, Eexp, step=1) as p:
        base_v = p * H
        pto.vmi.vstore(zero_f, vexp_ub, base_v + 0 * VL, mask_h)
        pto.vmi.vstore(zero_f, vexp_ub, base_v + 1 * VL, mask_h)
        pto.vmi.vstore(zero_f, sfexp_pad, p * VL, mask_h)

    with pto.for_(0, T_, step=1) as t:
        v0 = pto.vmi.vload(v_ub, t * H + 0 * VL, size=VL)
        v1 = pto.vmi.vload(v_ub, t * H + 1 * VL, size=VL)
        t_i32 = scalar.index_cast(pto.i32, t)
        base_sf = pto.vmi.vbrc(t_i32 * Hs, size=VL)
        offs_sf = pto.vmi.vadd(base_sf, arange, mask_h)
        sf_frag = pto.vmi.vgather(sf_ub, offs_sf, mask_hs_vl)

        for k in range(K):
            pv = pto.vmi.vload(pos_ub, t * K + k, size=1)
            pto.vmi.vstore(pv, pos_row, k)

        for k in range(K):
            pos_v = pto.vmi.vload(pos_row, k, size=1)
            pos_b = pto.vmi.vbrc(pos_v, size=VL)
            with pto.for_(0, Eexp, step=1) as p:
                p_i32 = scalar.index_cast(pto.i32, p)
                hit = pto.vmi.vcmps(pos_b, p_i32, mask_h, "eq")
                hit_hs = pto.vmi.vcmps(pos_b, p_i32, mask_hs_vl, "eq")
                base_v = p * H
                old0 = pto.vmi.vload(vexp_ub, base_v + 0 * VL, size=VL)
                old1 = pto.vmi.vload(vexp_ub, base_v + 1 * VL, size=VL)
                pto.vmi.vstore(pto.vmi.vsel(hit, v0, old0), vexp_ub, base_v + 0 * VL, mask_h)
                pto.vmi.vstore(pto.vmi.vsel(hit, v1, old1), vexp_ub, base_v + 1 * VL, mask_h)
                old_s = pto.vmi.vload(sfexp_pad, p * VL, size=VL)
                pto.vmi.vstore(
                    pto.vmi.vsel(hit_hs, sf_frag, old_s), sfexp_pad, p * VL, mask_h
                )

    # Pack VL-padded Sf → compact Hs rows via size=1 moves
    # (vscatter/partial vstore at Hs-stride fails masked_store alignment)
    for p in range(Eexp):
        for j in range(Hs):
            val = pto.vmi.vload(sfexp_pad, p * VL + j, size=1)
            pto.vmi.vstore(val, sfexp_ub, p * Hs + j)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_store(vexp_ub, vexp_gm, VEXP_BYTES, nburst=(1, VEXP_BYTES, VEXP_BYTES))
    pto.mte_store(sfexp_ub, sfexp_gm, SFEXP_BYTES, nburst=(1, SFEXP_BYTES, SFEXP_BYTES))


def reference(V, Sf, Pos, Expert):
    Vexp = np.zeros((Eexp, H), dtype=np.float32)
    Sfexp = np.zeros((Eexp, Hs), dtype=np.float32)
    for p in range(Eexp):
        if Expert[p] < 0:
            Vexp[p] = 0
            Sfexp[p] = 0
    for t in range(T_):
        v_frag = V[t]
        sf_frag = Sf[t]
        pos_row = Pos[t].copy()
        for k in range(K):
            pos = int(pos_row[k])
            if pos >= 0:
                Vexp[pos] = v_frag
                Sfexp[pos] = sf_frag
    return Vexp, Sfexp


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
    V = rng.uniform(-1, 1, (T_, H)).astype(np.float32)
    Sf = rng.uniform(-1, 1, (T_, Hs)).astype(np.float32)
    Pos = np.arange(Eexp, dtype=np.int32).reshape(T_, K)
    Expert = (np.arange(Eexp, dtype=np.int32) % 8)
    Expert[0] = -1
    Expert[1] = -1
    Pos[0, 0] = -1
    ref_v, ref_s = reference(V, Sf, Pos, Expert)
    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    def malloc(n):
        p = ctypes.c_void_p()
        _chk(lib.aclrtMalloc(ctypes.byref(p), n, _ACL_MEM_MALLOC_HUGE_FIRST), "malloc")
        return p

    vd, sfd, pd, ed = malloc(V_BYTES), malloc(SF_BYTES), malloc(POS_BYTES), malloc(EXP_BYTES)
    vxd, sxd = malloc(VEXP_BYTES), malloc(SFEXP_BYTES)

    def h2d(dst, arr, n):
        _chk(
            lib.aclrtMemcpy(
                dst, n, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), n, _ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "h2d",
        )

    h2d(vd, V, V_BYTES)
    h2d(sfd, Sf, SF_BYTES)
    h2d(pd, Pos, POS_BYTES)
    h2d(ed, Expert, EXP_BYTES)
    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](int(vxd.value), int(sxd.value), int(vd.value), int(sfd.value), int(pd.value), int(ed.value))
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    vh = np.zeros((Eexp, H), dtype=np.float32)
    sh = np.zeros((Eexp, Hs), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(vh.ctypes.data_as(ctypes.c_void_p), VEXP_BYTES, vxd, VEXP_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h_v",
    )
    _chk(
        lib.aclrtMemcpy(sh.ctypes.data_as(ctypes.c_void_p), SFEXP_BYTES, sxd, SFEXP_BYTES, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h_s",
    )
    for p in (vd, sfd, pd, ed, vxd, sxd):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    dv = float(np.max(np.abs(vh - ref_v)))
    ds = float(np.max(np.abs(sh - ref_s)))
    if not (np.allclose(vh, ref_v, atol=1e-4) and np.allclose(sh, ref_s, atol=1e-4)):
        # Extra diagnostics for blockers
        print(
            f"DIAG v={dv} s={ds} sh_max={float(np.max(np.abs(sh)))} "
            f"sh_nnz={int(np.count_nonzero(sh))} ref_s_max={float(np.max(np.abs(ref_s)))} "
            f"sh0={sh[0].tolist()} sh1={sh[1].tolist()} ref1={ref_s[1].tolist()}"
        )
        raise AssertionError(f"mismatch v={dv} s={ds}")
    print(f"PASS {tag}  compile={cs:.3f}s launch={ls:.3f}s thr_mirror={THR}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("T", nargs="?", type=int, default=T_)
    ap.add_argument("K", nargs="?", type=int, default=K)
    ap.add_argument("H", nargs="?", type=int, default=H)
    ap.add_argument("G", nargs="?", type=int, default=G)
    ap.add_argument("Thr", nargs="?", type=int, default=THR)
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    tag = f"sp1d_t{T_}_k{K}_h{H}_g{G}_t{THR}_keep_pos"
    if args.emit_mlir:
        print(kernel.compile().mlir_text())
        return 0
    return run_acl(tag)


if __name__ == "__main__":
    raise SystemExit(main())
