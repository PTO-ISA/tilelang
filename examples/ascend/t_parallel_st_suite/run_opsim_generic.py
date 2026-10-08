#!/usr/bin/env python3
"""Generic opsim for SV1–SV9 + CF1–CF6 + SP1–SP6 Simt (+ VMI twins sv*v / cf*v).

Case-3 SV5/SV6/SV7 (+v): reduced Y[R,CG] fp32. Case-3 SV8/sv8v: full Out[R,C] fp16.
SV2 (ex-SV1B) row-scale eltwise; SV3 (ex-SV2G) GEMV Acc KEEP/split; SV4 index gather+psum.
SV9 topk e2e: use run_opsim_topk.py.
Legacy aliases: sv1b_→sv2_, sv2g_→sv3_; old sv7_n* block-reduce still gated.
"""
from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
import run_cf_mb_opsim as R  # noqa: E402

TAG = sys.argv[1]
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/t_parallel_st_suite")
SO = OUT / "so" / f"{TAG}.so"

acl = R.load_acl()
R.check_acl(acl.aclInit(None), "init")
R.check_acl(acl.aclrtSetDevice(0), "dev")
stream = ctypes.c_void_p()
R.check_acl(acl.aclrtCreateStream(ctypes.byref(stream)), "stream")
ctypes.CDLL("libruntime.so", mode=ctypes.RTLD_GLOBAL)

lib = ctypes.CDLL(str(SO))
fn = getattr(lib, "call")


def malloc(nbytes):
    p = ctypes.c_void_p()
    R.check_acl(acl.aclrtMalloc(ctypes.byref(p), nbytes, R.ACL_MEM_MALLOC_HUGE_FIRST), "m")
    return p


def h2d(d, arr):
    R.check_acl(
        acl.aclrtMemcpy(d, arr.nbytes, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, R.ACL_MEMCPY_HOST_TO_DEVICE),
        "h2d",
    )


def d2h(arr, d):
    R.check_acl(
        acl.aclrtMemcpy(arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, d, arr.nbytes, R.ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )


ok = False
detail = ""

if TAG.startswith("sv1_") or TAG.startswith("sv1v_"):
    m = re.search(r"_e(\d+)_t(\d+)", TAG)
    E = int(m.group(1))
    a = np.arange(E, dtype=np.float32) * 0.01
    b = np.arange(E, dtype=np.float32) * 0.02 + 1.0
    exp = a + b
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    da, db, dc = malloc(a.nbytes), malloc(b.nbytes), malloc(out.nbytes)
    h2d(da, a)
    h2d(db, b)
    print("call rc", fn(da, db, dc, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, dc)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out-exp))}"

elif TAG.startswith("sv5v_"):
    # VMI twin of Case-3 SV5: same gold as sv5_ (seed 5)
    # sv5v_r{R}_c{C}_g{G}_t{L}[_{keep_reg|ub_reload}] — arm suffix optional,
    # it only changes scale residency (RF vs UB round-trip), never the gold.
    m = re.search(r"_r(\d+)_c(\d+)_g(\d+)_t(\d+)(?:_(keep_in_rf|ub_stream|keep_reg|ub_reload))?$", TAG)
    if m is None:
        raise SystemExit(
            f"sv5v tag must match _r{{R}}_c{{C}}_g{{G}}_t{{T}}[_{{keep_in_rf|ub_stream|keep_reg|ub_reload}}], got {TAG}"
        )
    ARM = m.group(5) or "legacy_noarm"
    R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
    CG = C // G
    rng = np.random.default_rng(5)
    x = rng.standard_normal((R_, C)).astype(np.float16)
    x32 = x.astype(np.float32)
    amax = np.zeros((R_, CG), np.float32)
    for i in range(R_):
        for g in range(CG):
            amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
    exp = (1.0 / np.maximum(amax, 1e-6)).astype(np.float32)
    y = np.zeros((R_, CG), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 3
    dx, dy = malloc(x.nbytes), malloc(y.nbytes)
    h2d(dx, x)
    print("call rc", fn(dx, dy, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(y, dy)
    ok = bool(np.allclose(y, exp, rtol=1e-4, atol=1e-4))
    detail = f"vmi maxabs={np.max(np.abs(y-exp))} R={R_} C={C} G={G} CG={CG} arm={ARM}"

elif TAG.startswith("sv5_"):
    # Case-3: sv5_r{R}_c{C}_g{G}_t{T}[_{keep_reg|ub_reload}] → Y[R,CG] fp32
    #         = 1/max(group_absmax, 1e-6)
    # The arm suffix selects post-reduce inverse-scale residency
    # (keep_reg = live fragment/RF, ub_reload = store to UB + reload in a
    # later Parallel loop). Gold is IDENTICAL for both arms and for the
    # legacy no-arm tag sv5_r64_c128_g16_t32 (pre-arm baseline).
    # Legacy sv5_e{E}_t{T}_{frag,reload} (Y,Z vectors) dropped — use old kernel if needed.
    m = re.search(r"_r(\d+)_c(\d+)_g(\d+)_t(\d+)(?:_(keep_in_rf|ub_stream|keep_reg|ub_reload))?$", TAG)
    if m is None:
        raise SystemExit(
            f"sv5 tag must match _r{{R}}_c{{C}}_g{{G}}_t{{T}}[_{{keep_in_rf|ub_stream|keep_reg|ub_reload}}], got {TAG}"
        )
    ARM = m.group(5) or "legacy_noarm"
    R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
    CG = C // G
    rng = np.random.default_rng(5)
    x = rng.standard_normal((R_, C)).astype(np.float16)
    x32 = x.astype(np.float32)
    amax = np.zeros((R_, CG), np.float32)
    for i in range(R_):
        for g in range(CG):
            amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
    exp = (1.0 / np.maximum(amax, 1e-6)).astype(np.float32)
    y = np.zeros((R_, CG), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 3
    dx, dy = malloc(x.nbytes), malloc(y.nbytes)
    h2d(dx, x)
    print("call rc", fn(dx, dy, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(y, dy)
    ok = bool(np.allclose(y, exp, rtol=1e-4, atol=1e-4))
    detail = f"maxabs={np.max(np.abs(y-exp))} R={R_} C={C} G={G} CG={CG} arm={ARM}"

elif TAG.startswith("sv7_") or TAG.startswith("sv7v_"):
    # Case-3 large-G (ex old SV6): sv7_r{R}_c{C}_g{G}_t{T} → Y[R,CG] fp32
    # Legacy block-reduce: sv7_n{N}_t{T} still gated
    m = re.search(
        r"_r(\d+)_c(\d+)_g(\d+)_t(\d+)(?:_(multiwarp_ub|reload))?$", TAG
    )
    if m is not None:
        R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ARM7 = m.group(5) or "legacy_noarm"
        CG = C // G
        # seed 6 matches archived large-G Case-3 gold (ex-SV6); sv7v same math
        rng = np.random.default_rng(6)
        x = rng.standard_normal((R_, C)).astype(np.float16)
        x32 = x.astype(np.float32)
        amax = np.zeros((R_, CG), np.float32)
        for i in range(R_):
            for g in range(CG):
                amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
        exp = (1.0 / np.maximum(amax, 1e-6)).astype(np.float32)
        y = np.zeros((R_, CG), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 3
        dx, dy = malloc(x.nbytes), malloc(y.nbytes)
        h2d(dx, x)
        print("call rc", fn(dx, dy, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(y, dy)
        ok = bool(np.allclose(y, exp, rtol=1e-4, atol=1e-4))
        detail = f"maxabs={np.max(np.abs(y-exp))} R={R_} C={C} G={G} CG={CG} arm={ARM7}"
    else:
        m = re.search(r"_n(\d+)_t(\d+)", TAG)
        if m is None:
            raise SystemExit(f"sv7 tag must match _r.._c.._g.._t.. or legacy _n.._t.., got {TAG}")
        N = int(m.group(1))
        inp = np.linspace(-3.0, 4.0, N, dtype=np.float32)
        inp[41 % N] = 9.5
        exp = np.array([inp.max()], np.float32)
        out = np.zeros(1, np.float32)
        fn.argtypes = [ctypes.c_void_p] * 3
        di, do = malloc(inp.nbytes), malloc(out.nbytes)
        h2d(di, inp)
        print("call rc", fn(di, do, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(out, do)
        ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
        detail = f"legacy_n got={out.tolist()} exp={exp.tolist()}"

elif TAG.startswith("sv8v_"):
    # VMI twin of Case-3 SV8: same gold as sv8_ (seed 8)
    m = re.search(r"_r(\d+)_c(\d+)_g(\d+)_", TAG)
    if m is None:
        raise SystemExit(f"sv8v tag must match _r{{R}}_c{{C}}_g{{G}}_…, got {TAG}")
    R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
    CG = C // G
    rng = np.random.default_rng(8)
    x = rng.standard_normal((R_, C)).astype(np.float16)
    x32 = x.astype(np.float32)
    amax = np.zeros((R_, CG), np.float32)
    for i in range(R_):
        for g in range(CG):
            amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
    sf_inv = 1.0 / np.maximum(amax, 1e-6)
    scale_bc = np.repeat(sf_inv, G, axis=1)
    exp = (x32 * scale_bc).astype(np.float16)
    out = np.zeros((R_, C), np.float16)
    fn.argtypes = [ctypes.c_void_p] * 3
    di, do = malloc(x.nbytes), malloc(out.nbytes)
    h2d(di, x)
    print("call rc", fn(di, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out.astype(np.float32), exp.astype(np.float32), rtol=2e-2, atol=2e-2))
    detail = f"vmi maxabs={np.max(np.abs(out.astype(np.float32)-exp.astype(np.float32)))} R={R_} C={C} G={G}"

elif TAG.startswith("sv8_"):
    # Case-3: sv8_r{R}_c{C}_g{G}_t{T}_{live|spill_dist} → Out[R,C] fp16 = x * sf_inv
    # Optional legacy: sv8_32x32_t{T} → 32x32 fp32 normalize (old block reduce).
    m = re.search(r"_r(\d+)_c(\d+)_g(\d+)_", TAG)
    if m is not None:
        R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
        CG = C // G
        rng = np.random.default_rng(8)
        x = rng.standard_normal((R_, C)).astype(np.float16)
        x32 = x.astype(np.float32)
        amax = np.zeros((R_, CG), np.float32)
        for i in range(R_):
            for g in range(CG):
                amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
        sf_inv = 1.0 / np.maximum(amax, 1e-6)
        scale_bc = np.repeat(sf_inv, G, axis=1)
        exp = (x32 * scale_bc).astype(np.float16)
        out = np.zeros((R_, C), np.float16)
        fn.argtypes = [ctypes.c_void_p] * 3
        di, do = malloc(x.nbytes), malloc(out.nbytes)
        h2d(di, x)
        print("call rc", fn(di, do, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(out, do)
        ok = bool(np.allclose(out.astype(np.float32), exp.astype(np.float32), rtol=2e-2, atol=2e-2))
        detail = f"maxabs={np.max(np.abs(out.astype(np.float32)-exp.astype(np.float32)))} R={R_} C={C} G={G}"
    else:
        # legacy sv8_32x32_t*
        N = 32
        rng = np.random.default_rng(0)
        inp = rng.standard_normal((N, N)).astype(np.float32)
        inp[3, 7] = 12.0
        am = float(inp.max())
        exp = inp / am
        out = np.zeros((N, N), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 3
        di, do = malloc(inp.nbytes), malloc(out.nbytes)
        h2d(di, inp)
        print("call rc", fn(di, do, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(out, do)
        ok = bool(np.allclose(out, exp, rtol=1e-4, atol=1e-5))
        detail = f"legacy32x32 maxabs={np.max(np.abs(out-exp))}"

elif TAG.startswith("sv2_") or TAG.startswith("sv2v_") or TAG.startswith("sv1b_"):
    # SV2 fold gold: scale[i,lane]=max(EPS,max_ch abs(X[i,ch*LANES+lane])); LANES=T
    # Arms: input_keep|input_stream|frag_live|reload (same fold gold; scale always KEEP)
    m = re.search(r"_r(\d+)_c(\d+)_t(\d+)_", TAG)
    R_, C, LANES = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert C % LANES == 0, f"C={C} not divisible by LANES={LANES}"
    NCH = C // LANES
    rng = np.random.default_rng(2)
    x = rng.standard_normal((R_, C)).astype(np.float16)
    x32 = x.astype(np.float32)
    scale = np.full((R_, LANES), 1e-6, dtype=np.float32)
    for ch in range(NCH):
        chunk = x32[:, ch * LANES : (ch + 1) * LANES]
        scale = np.maximum(scale, np.abs(chunk))
    scale = np.maximum(scale, 1e-6)
    exp32 = np.empty_like(x32)
    for ch in range(NCH):
        exp32[:, ch * LANES : (ch + 1) * LANES] = (
            x32[:, ch * LANES : (ch + 1) * LANES] / scale
        )
    exp = exp32.astype(np.float16)
    out = np.zeros((R_, C), np.float16)
    fn.argtypes = [ctypes.c_void_p] * 3
    di, do = malloc(x.nbytes), malloc(out.nbytes)
    h2d(di, x)
    print("call rc", fn(di, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out.astype(np.float32), exp.astype(np.float32), rtol=2e-2, atol=2e-2))
    detail = f"fold_lanes={LANES} nch={NCH} maxabs={np.max(np.abs(out.astype(np.float32)-exp.astype(np.float32)))}"

elif TAG.startswith("sv3_") or TAG.startswith("sv3v_") or TAG.startswith("sv2g_"):
    # sv3_m{M}_vl{VL}_k{K}_t{T}_{keep|split_cm{CM}} (alias sv2g_*)
    m = re.search(r"_m(\d+)_vl(\d+)_k(\d+)_t(\d+)_", TAG)
    M, VL, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
    rng = np.random.default_rng(3)
    A = rng.standard_normal((K, M, VL)).astype(np.float16)
    X = rng.standard_normal((K,)).astype(np.float16)
    Acc = np.zeros((M, VL), np.float32)
    for k in range(K):
        Acc += A[k].astype(np.float32) * np.float32(X[k])
    exp = Acc
    out = np.zeros((M, VL), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    da, dx, do = malloc(A.nbytes), malloc(X.nbytes), malloc(out.nbytes)
    h2d(da, A)
    h2d(dx, X)
    print("call rc", fn(da, dx, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    # fp16→fp32 accum: modest atol/rtol
    ok = bool(np.allclose(out, exp, rtol=2e-2, atol=2e-2))
    detail = f"maxabs={np.max(np.abs(out-exp))} M={M} VL={VL} K={K}"

elif TAG.startswith("sv6v_"):
    # VMI twin of Case-3 SV6: same gold as sv6_ reduced (seed 6)
    m = re.search(r"_r(\d+)_c(\d+)_g(\d+)_t(\d+)(?:_(reload|keep_in_warp|ub_stream))?$", TAG)
    if m is None:
        raise SystemExit(
            f"sv6v tag must match _r{{R}}_c{{C}}_g{{G}}_t{{T}}[_{{reload|keep_in_warp|ub_stream}}], got {TAG}"
        )
    R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
    CG = C // G
    rng = np.random.default_rng(6)
    x = rng.standard_normal((R_, C)).astype(np.float16)
    x32 = x.astype(np.float32)
    amax = np.zeros((R_, CG), np.float32)
    for i in range(R_):
        for g in range(CG):
            amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
    exp = (1.0 / np.maximum(amax, 1e-6)).astype(np.float32)
    y = np.zeros((R_, CG), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 3
    dx, dy = malloc(x.nbytes), malloc(y.nbytes)
    h2d(dx, x)
    print("call rc", fn(dx, dy, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(y, dy)
    ok = bool(np.allclose(y, exp, rtol=1e-4, atol=1e-4))
    detail = f"vmi maxabs={np.max(np.abs(y-exp))} R={R_} C={C} G={G} CG={CG}"

elif TAG.startswith("sv6_"):
    # Case-3 mid-G SV6: sv6_r{R}_c{C}_g{G}_t{T} → Y[R,CG] fp32 (same gold family as sv5/sv7).
    # Legacy remat/reload tags (…_remat / …_reload) expected full Out[R,C] fp16 —
    # gated: only run reduced gold when tag has no arm suffix.
    # 2026-10-06: new RF-capacity arms keep_in_warp / reload BOTH use the reduced
    # gold; only the archived `remat` arm (ex large-G bcast kernel, moved to SV8)
    # expects the full Out[R,C] fp16 gold.
    m = re.search(
        r"_r(\d+)_c(\d+)_g(\d+)_t(\d+)(?:_(remat|keep_in_warp|reload))?$", TAG
    )
    if m is None:
        raise SystemExit(
            f"sv6 tag must match _r{{R}}_c{{C}}_g{{G}}_t{{T}}[_{{keep_in_warp|reload|remat}}], got {TAG}"
        )
    R_, C, G = int(m.group(1)), int(m.group(2)), int(m.group(3))
    arm = m.group(5)  # remat|reload or None
    CG = C // G
    rng = np.random.default_rng(6)
    x = rng.standard_normal((R_, C)).astype(np.float16)
    x32 = x.astype(np.float32)
    amax = np.zeros((R_, CG), np.float32)
    for i in range(R_):
        for g in range(CG):
            amax[i, g] = np.max(np.abs(x32[i, g * G : (g + 1) * G]))
    if arm == "remat":
        # LEGACY full Out[R,C] = x / amax_bc
        scale_bc = np.maximum(np.repeat(amax, G, axis=1), 1e-6)
        exp = (x32 / scale_bc).astype(np.float16)
        out = np.zeros((R_, C), np.float16)
        fn.argtypes = [ctypes.c_void_p] * 3
        di, do = malloc(x.nbytes), malloc(out.nbytes)
        h2d(di, x)
        print("call rc", fn(di, do, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(out, do)
        ok = bool(np.allclose(out.astype(np.float32), exp.astype(np.float32), rtol=2e-2, atol=2e-2))
        detail = f"legacy_{arm} maxabs={np.max(np.abs(out.astype(np.float32)-exp.astype(np.float32)))}"
    else:
        exp = (1.0 / np.maximum(amax, 1e-6)).astype(np.float32)
        y = np.zeros((R_, CG), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 3
        dx, dy = malloc(x.nbytes), malloc(y.nbytes)
        h2d(dx, x)
        print("call rc", fn(dx, dy, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(y, dy)
        ok = bool(np.allclose(y, exp, rtol=1e-4, atol=1e-4))
        detail = f"maxabs={np.max(np.abs(y-exp))} R={R_} C={C} G={G} CG={CG}"


elif TAG.startswith("sv4_") or TAG.startswith("sv4v_"):
    # sv4(v)_e{E}_b{B}_t{T}_{keep_idx,remat_idx} — gather + partial sum across B
    # Replaced keep2 (identity idxs, Out[B,E]). Gold: Out[i]=sum_b A[b,i]*W[Idx[i]]
    m = re.search(r"_e(\d+)_b(\d+)_t(\d+)_(keep_idx|remat_idx)", TAG)
    if m is None:
        raise SystemExit(
            f"sv4 tag must match _e{{E}}_b{{B}}_t{{T}}_(keep_idx|remat_idx), got {TAG}"
        )
    E, B = int(m.group(1)), int(m.group(2))
    arm = m.group(4)
    rng = np.random.default_rng(4)
    A = rng.standard_normal((B, E)).astype(np.float32)
    W = rng.standard_normal((E,)).astype(np.float32)
    # Fixed non-identity permutation so gather is real (not idxs[i]=i)
    Idx = ((np.arange(E, dtype=np.int64) * 7 + 3) % E).astype(np.int32)
    exp = (A * W[Idx][None, :]).sum(axis=0).astype(np.float32)
    out = np.zeros((E,), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 5
    da, dw, di, do = (
        malloc(A.nbytes),
        malloc(W.nbytes),
        malloc(Idx.nbytes),
        malloc(out.nbytes),
    )
    h2d(da, A)
    h2d(dw, W)
    h2d(di, Idx)
    print("call rc", fn(da, dw, di, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out-exp))} E={E} B={B} arm={arm}"

elif TAG.startswith("cf1_") or TAG.startswith("cf1v_"):
    # cf1_e{E}_k{K}_t{T}_keep — KEEP thresh kill
    m = re.search(r"_e(\d+)_k(\d+)_t(\d+)_keep", TAG)
    if m is None:
        raise SystemExit(f"cf1 tag must match _e{{E}}_k{{K}}_t{{T}}_keep, got {TAG}")
    E, K = int(m.group(1)), int(m.group(2))
    NEG = np.float32(-3.402823e38)
    rng = np.random.default_rng(11)
    a = rng.standard_normal((E,)).astype(np.float32)
    # descending thresholds so successive kills are meaningful
    thr = np.sort(a.copy())[::-1][:K].astype(np.float32)
    thr = thr * 0.99  # slightly below max scores so some lanes die each round
    scores = a.copy()
    for k in range(K):
        scores = np.where(scores > thr[k], NEG, scores)
    exp = scores.astype(np.float32)
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    da, dt, do = malloc(a.nbytes), malloc(thr.nbytes), malloc(out.nbytes)
    h2d(da, a)
    h2d(dt, thr)
    print("call rc", fn(da, dt, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out.astype(np.float64)-exp.astype(np.float64)))} E={E} K={K}"

elif TAG.startswith("cf2_") or TAG.startswith("cf2v_"):
    # cf2_e{E}_k{K}_t{T}_remat — remat + kill-in-shared (same gold as CF1)
    m = re.search(r"_e(\d+)_k(\d+)_t(\d+)_remat", TAG)
    if m is None:
        raise SystemExit(f"cf2 tag must match _e{{E}}_k{{K}}_t{{T}}_remat, got {TAG}")
    E, K = int(m.group(1)), int(m.group(2))
    NEG = np.float32(-3.402823e38)
    rng = np.random.default_rng(12)
    a = rng.standard_normal((E,)).astype(np.float32)
    thr = (np.sort(a.copy())[::-1][:K] * 0.99).astype(np.float32)
    scores = a.copy()
    for k in range(K):
        scores = np.where(scores > thr[k], NEG, scores)
    exp = scores.astype(np.float32)
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    da, dt, do = malloc(a.nbytes), malloc(thr.nbytes), malloc(out.nbytes)
    h2d(da, a)
    h2d(dt, thr)
    print("call rc", fn(da, dt, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out.astype(np.float64)-exp.astype(np.float64)))} E={E} K={K}"

elif TAG.startswith("cf3_") or TAG.startswith("cf3v_"):
    # cf3_e{E}_k{K}_t{T}_remat_idx — kill by rematted i == victim[k]
    m = re.search(r"_e(\d+)_k(\d+)_t(\d+)_remat_idx", TAG)
    if m is None:
        raise SystemExit(f"cf3 tag must match _e{{E}}_k{{K}}_t{{T}}_remat_idx, got {TAG}")
    E, K = int(m.group(1)), int(m.group(2))
    NEG = np.float32(-3.402823e38)
    rng = np.random.default_rng(13)
    a = rng.standard_normal((E,)).astype(np.float32)
    # distinct victims in [0, E)
    victim = rng.choice(E, size=K, replace=False).astype(np.int32)
    scores = a.copy()
    for k in range(K):
        scores[victim[k]] = NEG
    exp = scores.astype(np.float32)
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    da, dv, do = malloc(a.nbytes), malloc(victim.nbytes), malloc(out.nbytes)
    h2d(da, a)
    h2d(dv, victim)
    print("call rc", fn(da, dv, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out.astype(np.float64)-exp.astype(np.float64)))} E={E} K={K}"

elif TAG.startswith("cf4_") or TAG.startswith("cf4v_"):
    # cf4_e{E}_t{T}_pfat{P} — nested if; pfat = % fat path (skew in host data)
    m = re.search(r"_e(\d+)_t(\d+)_pfat(\d+)", TAG)
    if m is None:
        raise SystemExit(f"cf4 tag must match _e{{E}}_t{{T}}_pfat{{P}}, got {TAG}")
    E, pfat = int(m.group(1)), int(m.group(3))
    lo, hi = np.float32(-1.0), np.float32(1.0)
    rng = np.random.default_rng(14)
    scale = (rng.standard_normal((E,)).astype(np.float32) * 0.5 + 1.0)
    # skew: (100-pfat)% short path (outside [lo,hi]), pfat% fat path inside
    n_fat = max(1, int(round(E * pfat / 100.0)))
    a = np.empty(E, np.float32)
    # fat lanes in [lo, hi]
    a[:n_fat] = rng.uniform(-0.99, 0.99, size=n_fat).astype(np.float32)
    # short: half above hi, half below lo
    n_hi = (E - n_fat) // 2
    a[n_fat : n_fat + n_hi] = rng.uniform(1.01, 3.0, size=n_hi).astype(np.float32)
    a[n_fat + n_hi :] = rng.uniform(-3.0, -1.01, size=E - n_fat - n_hi).astype(np.float32)
    rng.shuffle(a)
    exp = np.empty(E, np.float32)
    for i in range(E):
        x = a[i]
        if x > hi:
            exp[i] = hi
        elif x < lo:
            exp[i] = lo
        else:
            exp[i] = x * scale[i]
    out = np.zeros(E, np.float32)
    lo_t = np.array([lo], np.float32)
    hi_t = np.array([hi], np.float32)
    fn.argtypes = [ctypes.c_void_p] * 6
    da, ds, dlo, dhi, do = (
        malloc(a.nbytes),
        malloc(scale.nbytes),
        malloc(lo_t.nbytes),
        malloc(hi_t.nbytes),
        malloc(out.nbytes),
    )
    h2d(da, a)
    h2d(ds, scale)
    h2d(dlo, lo_t)
    h2d(dhi, hi_t)
    print("call rc", fn(da, ds, dlo, dhi, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out-exp))} E={E} pfat={pfat}"

elif TAG.startswith("cf5_") or TAG.startswith("cf5v_"):
    # cf5_e{E}_t{T}_pnear{P} — div + near0 branch; eps=1e-4 fixed
    m = re.search(r"_e(\d+)_t(\d+)_pnear(\d+)", TAG)
    if m is None:
        raise SystemExit(f"cf5 tag must match _e{{E}}_t{{T}}_pnear{{P}}, got {TAG}")
    E, pnear = int(m.group(1)), int(m.group(3))
    eps = np.float32(1e-4)
    rng = np.random.default_rng(15)
    n_near = max(1, int(round(E * pnear / 100.0)))
    a = rng.standard_normal((E,)).astype(np.float32) * 2.0
    b = rng.standard_normal((E,)).astype(np.float32) * 2.0 + 0.5  # avoid tiny denom
    # force first n_near lanes near 0 in a
    a[:n_near] = rng.uniform(-eps * 0.5, eps * 0.5, size=n_near).astype(np.float32)
    # ensure remaining |a| >= eps
    mask = np.abs(a) < eps
    a = np.where(mask & (np.arange(E) >= n_near), np.sign(a) * (eps * 2.0 + 0.1), a)
    exp = np.where(np.abs(a) < eps, np.float32(0.0), (a / b).astype(np.float32))
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    da, db, do = malloc(a.nbytes), malloc(b.nbytes), malloc(out.nbytes)
    h2d(da, a)
    h2d(db, b)
    print("call rc", fn(da, db, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=2e-5, atol=1e-6))
    detail = f"maxabs={np.max(np.abs(out-exp))} E={E} pnear={pnear}"

elif TAG.startswith("cf6b_"):
    # cf6b_e{E}_t{T}_phard{P} — fat hard-band Newton+poly vs thin easy; seed=26
    m = re.search(r"_e(\d+)_t(\d+)_phard(\d+)", TAG)
    if m is None:
        raise SystemExit(f"cf6b tag must match _e{{E}}_t{{T}}_phard{{P}}, got {TAG}")
    E, phard = int(m.group(1)), int(m.group(3))
    eps = np.float32(1e-4)
    hard_hi = np.float32(1e-1)
    n_fast, n_hard = 2, 12
    poly_c = (0.0, 1e-6, -2e-6, 3e-6, -4e-6, 5e-6, -6e-6, 7e-6)
    rng = np.random.default_rng(26)
    n_h = max(1, int(round(E * phard / 100.0)))
    x = rng.uniform(0.5, 2.0, size=E).astype(np.float32)
    lo = np.float32(float(eps) * 2.0)
    hi = np.float32(float(hard_hi) * 0.9)
    x[:n_h] = rng.uniform(lo, hi, size=n_h).astype(np.float32)
    mask_bad = (np.abs(x) < eps) & (np.arange(E) >= n_h)
    x = np.where(mask_bad, np.float32(1.0), x).astype(np.float32)
    cs = [np.float32(c) for c in poly_c]
    exp = np.zeros(E, np.float32)
    for i in range(E):
        xi = np.float32(x[i])
        ax = abs(float(xi))
        if ax < float(eps):
            exp[i] = 0.0
        elif ax < float(hard_hi):
            y = np.float32(1.0) / xi
            for _ in range(n_hard):
                y = y * (np.float32(2.0) - xi * y)
            p_ = cs[7]
            for k in range(6, -1, -1):
                p_ = p_ * xi + cs[k]
            exp[i] = y + p_
        else:
            y = np.float32(1.0) / xi
            for _ in range(n_fast):
                y = y * (np.float32(2.0) - xi * y)
            exp[i] = y
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 3
    dx, do = malloc(x.nbytes), malloc(out.nbytes)
    h2d(dx, x)
    print("call rc", fn(dx, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=2e-5, atol=1e-6))
    detail = f"maxabs={np.max(np.abs(out-exp))} E={E} phard={phard}"


elif TAG.startswith("cf6_") or TAG.startswith("cf6v_"):
    # cf6_e{E}_t{T}_pnear{P} — Newton recip; eps=1e-4, N_FAST=3
    m = re.search(r"_e(\d+)_t(\d+)_pnear(\d+)", TAG)
    if m is None:
        raise SystemExit(f"cf6 tag must match _e{{E}}_t{{T}}_pnear{{P}}, got {TAG}")
    E, pnear = int(m.group(1)), int(m.group(3))
    eps = np.float32(1e-4)
    n_fast = 3
    rng = np.random.default_rng(16)
    n_near = max(1, int(round(E * pnear / 100.0)))
    x = rng.standard_normal((E,)).astype(np.float32) * 2.0 + 1.0  # mostly away from 0
    x[:n_near] = rng.uniform(-eps * 0.5, eps * 0.5, size=n_near).astype(np.float32)
    mask_far = (np.abs(x) < eps) & (np.arange(E) >= n_near)
    x = np.where(mask_far, np.sign(x) * (eps * 2.0 + 0.5), x)
    exp = np.zeros(E, np.float32)
    for i in range(E):
        ax = abs(float(x[i]))
        if ax < float(eps):
            exp[i] = 0.0
        else:
            y = np.float32(1.0) / x[i]
            for _ in range(n_fast):
                y = y * (np.float32(2.0) - x[i] * y)
            exp[i] = y
    out = np.zeros(E, np.float32)
    fn.argtypes = [ctypes.c_void_p] * 3
    dx, do = malloc(x.nbytes), malloc(out.nbytes)
    h2d(dx, x)
    print("call rc", fn(dx, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=2e-5, atol=1e-6))
    detail = f"maxabs={np.max(np.abs(out-exp))} E={E} pnear={pnear}"


elif TAG.startswith("sp1_"):
    # sp1_t{T}_k{K}_h{H}_g{G}_t{Thr}_{keep_pos|remat_pos}
    m = re.search(r"_t(\d+)_k(\d+)_h(\d+)_g(\d+)_t(\d+)_(keep_pos|remat_pos)", TAG)
    if m is None:
        raise SystemExit(f"sp1 tag parse fail: {TAG}")
    T_, K, H, G = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    Hs = H // G
    Eexp = T_ * K
    rng = np.random.default_rng(101)
    V = rng.standard_normal((T_, H)).astype(np.float32)
    Sf = rng.standard_normal((T_, Hs)).astype(np.float32)
    # injective Pos: token t,topk k → slot t*K+k; inject a few -1 unused
    Pos = np.arange(T_ * K, dtype=np.int32).reshape(T_, K)
    Pos[0, K - 1] = -1  # one unused topk → leaves slot 1 empty
    Expert = np.arange(Eexp, dtype=np.int32)
    # pad holes only on slots that are NOT live Pos targets (kernel: pad-zero then scatter)
    live = set(int(x) for x in Pos.ravel() if int(x) >= 0)
    for p in (1, Eexp // 2, Eexp - 1):
        if p not in live:
            Expert[p] = -1
    Vexp = np.zeros((Eexp, H), np.float32)
    Sfexp = np.zeros((Eexp, Hs), np.float32)
    # match kernel: pad zeros first (already zero), then scatter overwrites live Pos
    for t in range(T_):
        for k in range(K):
            pos = int(Pos[t, k])
            if pos >= 0:
                Vexp[pos] = V[t]
                Sfexp[pos] = Sf[t]
    out_v = np.zeros_like(Vexp)
    out_sf = np.zeros_like(Sfexp)
    fn.argtypes = [ctypes.c_void_p] * 7
    dv, dsf, dpos, dexp, dvo, dsfo = (
        malloc(V.nbytes), malloc(Sf.nbytes), malloc(Pos.nbytes),
        malloc(Expert.nbytes), malloc(out_v.nbytes), malloc(out_sf.nbytes),
    )
    h2d(dv, V); h2d(dsf, Sf); h2d(dpos, Pos); h2d(dexp, Expert)
    print("call rc", fn(dv, dsf, dpos, dexp, dvo, dsfo, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out_v, dvo); d2h(out_sf, dsfo)
    ok = bool(np.allclose(out_v, Vexp, rtol=1e-5, atol=1e-5) and np.allclose(out_sf, Sfexp, rtol=1e-5, atol=1e-5))
    detail = f"vmax={np.max(np.abs(out_v-Vexp))} sfmax={np.max(np.abs(out_sf-Sfexp))} T={T_} K={K} H={H}"

elif TAG.startswith("sp2_"):
    # sp2_t{T}_k{K}_h{H}_g{G}_t{Thr}_{sf0|sf1}_{w0|w1}_{keep_acc|remat_acc|keep_pos|remat_pos}
    m = re.search(
        r"_t(\d+)_k(\d+)_h(\d+)_g(\d+)_t(\d+)_(sf[01])_(w[01])_(keep_acc|remat_acc|keep_pos|remat_pos)",
        TAG,
    )
    if m is None:
        raise SystemExit(f"sp2 tag parse fail: {TAG}")
    T_, K, H, G = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    with_sf = m.group(6) == "sf1"
    with_w = m.group(7) == "w1"
    Hs = H // G
    Eexp = T_ * K
    rng = np.random.default_rng(102)
    Vexp = rng.standard_normal((Eexp, H)).astype(np.float32)
    Sf = rng.standard_normal((Eexp, Hs)).astype(np.float32)
    W = rng.standard_normal((T_, K)).astype(np.float32)
    Pos = np.arange(T_ * K, dtype=np.int32).reshape(T_, K)
    Pos[-1, -1] = -1
    exp = np.zeros((T_, H), np.float32)
    for t in range(T_):
        for k in range(K):
            pos = int(Pos[t, k])
            if pos < 0:
                continue
            wk = float(W[t, k]) if with_w else 1.0
            if with_sf:
                for g in range(Hs):
                    s = wk * float(Sf[pos, g])
                    exp[t, g * G : (g + 1) * G] += Vexp[pos, g * G : (g + 1) * G] * s
            else:
                exp[t] += Vexp[pos] * wk
    out = np.zeros((T_, H), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 6
    dv, dsf, dw, dpos, do = (
        malloc(Vexp.nbytes), malloc(Sf.nbytes), malloc(W.nbytes),
        malloc(Pos.nbytes), malloc(out.nbytes),
    )
    h2d(dv, Vexp); h2d(dsf, Sf); h2d(dw, W); h2d(dpos, Pos)
    print("call rc", fn(dv, dsf, dw, dpos, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-4, atol=1e-4))
    detail = f"maxabs={np.max(np.abs(out-exp))} T={T_} sf={with_sf} w={with_w}"

elif TAG.startswith("sp3_"):
    # sp3_m{M}_h{H}_g{G}_t{Thr}_{fp32|e8m0|ue8m0}
    # Soft Pow2 LUT RETIRED. e8m0 = A5 bit-reinterpret compose (host gold = e<<23 view).
    m = re.search(r"_m(\d+)_h(\d+)_g(\d+)_t(\d+)_(fp32|e8m0|ue8m0)", TAG)
    if m is None:
        raise SystemExit(f"sp3 tag parse fail: {TAG}")
    M, H, G, arm = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(5)
    if arm == "ue8m0":
        arm = "e8m0"
    Hs = H // G
    rng = np.random.default_rng(103)
    V = rng.standard_normal((M, H)).astype(np.float32)
    if arm == "fp32":
        Sf = rng.uniform(0.25, 4.0, size=(M, Hs)).astype(np.float32)
        scale = np.repeat(Sf, G, axis=1)
        exp = (V * scale).astype(np.float32)
        out = np.zeros((M, H), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 4
        dv, dsf, do = malloc(V.nbytes), malloc(Sf.nbytes), malloc(out.nbytes)
        h2d(dv, V); h2d(dsf, Sf)
        print("call rc", fn(dv, dsf, do, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(out, do)
    else:
        # A5 bit-reinterpret gold: bits = e<<23; view as f32. NOT a device Pow2 LUT.
        e = rng.integers(100, 127, size=(M, Hs), dtype=np.uint8)
        Sf_u8 = e
        bits = (e.astype(np.uint32) << np.uint32(23))
        Sf_f = bits.view(np.float32)
        scale = np.repeat(Sf_f, G, axis=1)
        exp = (V * scale).astype(np.float32)
        out = np.zeros((M, H), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 4
        dv, dsf, do = malloc(V.nbytes), malloc(Sf_u8.nbytes), malloc(out.nbytes)
        h2d(dv, V); h2d(dsf, Sf_u8)
        print("call rc", fn(dv, dsf, do, stream))
        R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
        d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-4, atol=1e-4))
    detail = f"maxabs={np.max(np.abs(out-exp))} M={M} arm={arm}"

elif TAG.startswith("sp4_"):
    # sp4_e{Eexp}_h{H}_t{Thr}_pad{Pct}
    m = re.search(r"_e(\d+)_h(\d+)_t(\d+)_pad(\d+)", TAG)
    if m is None:
        raise SystemExit(f"sp4 tag parse fail: {TAG}")
    Eexp, H, pad_pct = int(m.group(1)), int(m.group(2)), int(m.group(4))
    rng = np.random.default_rng(104)
    Buf = rng.standard_normal((Eexp, H)).astype(np.float32)
    Expert = np.arange(Eexp, dtype=np.int32)
    n_pad = max(1, int(round(Eexp * pad_pct / 100.0)))
    pad_idx = rng.choice(Eexp, size=n_pad, replace=False)
    Expert[pad_idx] = -1
    exp = Buf.copy()
    exp[Expert < 0] = 0
    out = np.zeros((Eexp, H), np.float32)
    fn.argtypes = [ctypes.c_void_p] * 4
    db, de, do = malloc(Buf.nbytes), malloc(Expert.nbytes), malloc(out.nbytes)
    h2d(db, Buf); h2d(de, Expert)
    print("call rc", fn(db, de, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out-exp))} Eexp={Eexp} pad={pad_pct}"

elif TAG.startswith("sp5_"):
    # sp5_n{N}_h{H}_g{G}_qg{Qg}_t{Thr}_{sideband|interleave}
    m = re.search(r"_n(\d+)_h(\d+)_g(\d+)_qg(\d+)_t(\d+)_(sideband|interleave)", TAG)
    if m is None:
        raise SystemExit(f"sp5 tag parse fail: {TAG}")
    N, H, G, Qg, arm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(6)
    Hs = H // G
    rng = np.random.default_rng(105)
    V = rng.standard_normal((N, H)).astype(np.float32)
    Sf = rng.uniform(0.5, 2.0, size=(N, Hs)).astype(np.float32)
    Idx = rng.integers(0, N, size=(Qg,), dtype=np.int32)
    exp = np.zeros((Qg, H), np.float32)
    for q in range(Qg):
        slot = int(Idx[q])
        scale = np.repeat(Sf[slot], G)
        exp[q] = V[slot] * scale
    out = np.zeros((Qg, H), np.float32)
    if arm == "sideband":
        fn.argtypes = [ctypes.c_void_p] * 5
        dv, dsf, di, do = malloc(V.nbytes), malloc(Sf.nbytes), malloc(Idx.nbytes), malloc(out.nbytes)
        h2d(dv, V); h2d(dsf, Sf); h2d(di, Idx)
        print("call rc", fn(dv, dsf, di, do, stream))
    else:
        Pack = np.concatenate([V, Sf], axis=1)
        fn.argtypes = [ctypes.c_void_p] * 4
        dp, di, do = malloc(Pack.nbytes), malloc(Idx.nbytes), malloc(out.nbytes)
        h2d(dp, Pack); h2d(di, Idx)
        print("call rc", fn(dp, di, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-4, atol=1e-4))
    detail = f"maxabs={np.max(np.abs(out-exp))} arm={arm} Qg={Qg}"

elif TAG.startswith("sp6_"):
    # sp6_n{N}_h{H}_g{G}_t{Thr}_{unpack|unpack_sf}  — soft LUT 4-bit e2m1 (appendix; not HW FP4)
    m = re.search(r"_n(\d+)_h(\d+)_g(\d+)_t(\d+)_(unpack(?:_sf)?)", TAG)
    if m is None:
        raise SystemExit(f"sp6 tag parse fail: {TAG}")
    N, H, G, arm = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(5)
    Hs = H // G
    Hbytes = H // 2
    e2m1 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    rng = np.random.default_rng(106)
    V_i8 = rng.integers(0, 256, size=(N, Hbytes), dtype=np.int16).astype(np.int8)
    Lut = e2m1.copy()
    decoded = np.zeros((N, H), np.float32)
    for n in range(N):
        for b in range(Hbytes):
            byte = int(V_i8[n, b]) & 0xFF
            lo, hi = byte & 0x0F, (byte >> 4) & 0x0F
            decoded[n, b * 2] = Lut[lo]
            decoded[n, b * 2 + 1] = Lut[hi]
    if arm == "unpack_sf":
        Sf = rng.uniform(0.5, 2.0, size=(N, Hs)).astype(np.float32)
        scale = np.repeat(Sf, G, axis=1)
        exp = (decoded * scale).astype(np.float32)
        out = np.zeros((N, H), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 5
        dv, dsf, dl, do = malloc(V_i8.nbytes), malloc(Sf.nbytes), malloc(Lut.nbytes), malloc(out.nbytes)
        h2d(dv, V_i8); h2d(dsf, Sf); h2d(dl, Lut)
        print("call rc", fn(dv, dsf, dl, do, stream))
    else:
        exp = decoded
        out = np.zeros((N, H), np.float32)
        fn.argtypes = [ctypes.c_void_p] * 4
        dv, dl, do = malloc(V_i8.nbytes), malloc(Lut.nbytes), malloc(out.nbytes)
        h2d(dv, V_i8); h2d(dl, Lut)
        print("call rc", fn(dv, dl, do, stream))
    R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
    d2h(out, do)
    ok = bool(np.allclose(out, exp, rtol=1e-5, atol=1e-5))
    detail = f"maxabs={np.max(np.abs(out-exp))} arm={arm} N={N} H={H}"


else:
    raise SystemExit(f"unknown tag family {TAG}")

print(("PASS" if ok else "FAIL"), detail)
(OUT / f"opsim_{TAG}_result.txt").write_text(f"ok={ok} {detail}\n")
raise SystemExit(0 if ok else 2)
