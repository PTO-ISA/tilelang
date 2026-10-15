#!/usr/bin/env python3
"""ST-SP5d — PTO-DSL SIMD twin of Simt SP5 (sideband vs interleave).

Simt twin: kernels/sp5_sideband_vs_interleave.py
Primary tags:
  sp5d_n64_h128_g32_qg32_t32_sideband
  sp5d_n64_h128_g32_qg32_t32_interleave

Gold (both arms): ``Out[q,j] = V[slot,j] * Sf[slot, j//G]`` with ``slot = Idx[q]``.
``Hs = H/G``, ``PackW = H + Hs``. Host seed 105 matches run_opsim_generic.

Arms:
  sideband   — two tensors: vgather ``V[slot,:]`` and vgather ``Sf[slot, j//G]``
  interleave — one ``Pack[slot] = concat(V row, Sf row)`` gather, then split the
               Hs scale tail and mul

Indexed / Hs-stride addresses use vgather / vscatter (Lok). Continuous VL is
not used on ``slot * Hs`` or on ``Pack`` rows (``PackW=132`` is not 32B-aligned
for odd slots). ``Thr`` is a Simt tag mirror only.
"""

import argparse
import ctypes
import time

import numpy as np
from ptodsl import pto
from ptodsl import scalar

VL = 64
N = 64
H = 128
G = 32
QG = 32
THR = 32
# lane // G for G=32
GROUP_SHIFT = 5
SEED = 105

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _align64(n: int) -> int:
    return (n + 63) // 64 * 64


def synthesize(n: int, h: int, g: int, qg: int):
    hs = h // g
    rng = np.random.default_rng(SEED)
    v = rng.standard_normal((n, h)).astype(np.float32)
    sf = rng.uniform(0.5, 2.0, size=(n, hs)).astype(np.float32)
    idx = rng.integers(0, n, size=(qg,), dtype=np.int32)
    gold = np.zeros((qg, h), np.float32)
    for q in range(qg):
        slot = int(idx[q])
        gold[q] = v[slot] * np.repeat(sf[slot], g)
    pack = np.concatenate([v, sf], axis=1)
    return v, sf, idx, pack, gold


def build_kernel(n: int, h: int, g: int, qg: int, threads: int, arm: str):
    hs = h // g
    pack_w = h + hs
    v_bytes = n * h * 4
    sf_bytes = n * hs * 4
    pack_bytes = n * pack_w * 4
    idx_bytes = qg * 4
    out_bytes = qg * h * 4
    name = f"SP5d_n{n}_h{h}_g{g}_qg{qg}_t{threads}_{arm}"

    if arm == "interleave":
        off_idx = _align64(pack_bytes)
        off_out = _align64(off_idx + idx_bytes)
        off_scratch = _align64(off_out + out_bytes)

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
            pack_gm: pto.ptr(pto.f32, "gm"),
            idx_gm: pto.ptr(pto.i32, "gm"),
        ):
            pack_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
            idx_ub = pto.castptr(pto.const(off_idx, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
            out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
            scratch = pto.castptr(pto.const(off_scratch, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

            pto.mte_load(pack_gm, pack_ub, 0, pack_bytes, nburst=(1, pack_bytes, pack_bytes))
            pto.mte_load(idx_gm, idx_ub, 0, idx_bytes, nburst=(1, idx_bytes, idx_bytes))
            pto.set_flag("MTE2", "V", event_id=0)
            pto.wait_flag("MTE2", "V", event_id=0)

            mask = pto.vmi.create_mask(VL, size=VL)
            mask_hs = pto.vmi.create_mask(hs, size=VL)
            mask_lo = pto.vmi.create_mask(g, size=VL)
            arange = pto.vmi.vci(pto.i32(0), size=VL)
            h_b = pto.vmi.vbrc(pto.i32(h), size=VL)
            pw_b = pto.vmi.vbrc(pto.i32(pack_w), size=VL)
            # Keep inactive gather lanes inside the Hs tail (Pack row ends at H+Hs).
            tail_i = pto.vmi.vmin(arange, pto.vmi.vbrc(pto.i32(hs - 1), size=VL), mask)

            with pto.for_(0, qg, step=1) as q:
                q_i32 = scalar.index_cast(pto.i32, q)
                slot_b = pto.vmi.vbrc(pto.vmi.vload(idx_ub, q, size=1), size=VL)
                pack_base = pto.vmi.vmul(slot_b, pw_b, mask)
                # One Pack[slot] stream: V prefix (2×VL) + Sf tail at column H.
                sf_offs = pto.vmi.vadd(pack_base, pto.vmi.vadd(h_b, tail_i, mask), mask)
                sf = pto.vmi.vgather(pack_ub, sf_offs, mask_hs)
                # Split tail into Hs scalars (vscatter; size=1 + vbrc). No VL load of Hs.
                pto.vmi.vscatter(sf, scratch, arange, mask_hs)
                scales = []
                for gi in range(hs):
                    scales.append(pto.vmi.vbrc(pto.vmi.vload(scratch, gi, size=1), size=VL))

                base_out = pto.vmi.vbrc(q_i32 * h, size=VL)
                for c in range(h // VL):
                    chunk = pto.vmi.vbrc(pto.i32(c * VL), size=VL)
                    v_offs = pto.vmi.vadd(pack_base, pto.vmi.vadd(arange, chunk, mask), mask)
                    v = pto.vmi.vgather(pack_ub, v_offs, mask)
                    # VL holds two groups: low G lanes = scale[2c], high = scale[2c+1]
                    s = pto.vmi.vsel(mask_lo, scales[c * 2], scales[c * 2 + 1])
                    y = pto.vmi.vmul(v, s, mask)
                    outs = pto.vmi.vadd(base_out, pto.vmi.vadd(arange, chunk, mask), mask)
                    pto.vmi.vscatter(y, out_ub, outs, mask)

            pto.set_flag("V", "MTE3", event_id=0)
            pto.wait_flag("V", "MTE3", event_id=0)
            pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

        return kernel

    # arm == "sideband"
    off_sf = _align64(v_bytes)
    off_idx = _align64(off_sf + sf_bytes)
    off_out = _align64(off_idx + idx_bytes)

    @pto.jit(
        name=name,
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
        ast_rewrite=False,
    )
    def kernel_sideband(
        out_gm: pto.ptr(pto.f32, "gm"),
        v_gm: pto.ptr(pto.f32, "gm"),
        sf_gm: pto.ptr(pto.f32, "gm"),
        idx_gm: pto.ptr(pto.i32, "gm"),
    ):
        v_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        sf_ub = pto.castptr(pto.const(off_sf, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        idx_ub = pto.castptr(pto.const(off_idx, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
        out_ub = pto.castptr(pto.const(off_out, dtype=pto.i64), pto.ptr(pto.f32, "ub"))

        pto.mte_load(v_gm, v_ub, 0, v_bytes, nburst=(1, v_bytes, v_bytes))
        pto.mte_load(sf_gm, sf_ub, 0, sf_bytes, nburst=(1, sf_bytes, sf_bytes))
        pto.mte_load(idx_gm, idx_ub, 0, idx_bytes, nburst=(1, idx_bytes, idx_bytes))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        mask = pto.vmi.create_mask(VL, size=VL)
        arange = pto.vmi.vci(pto.i32(0), size=VL)
        h_b = pto.vmi.vbrc(pto.i32(h), size=VL)
        hs_b = pto.vmi.vbrc(pto.i32(hs), size=VL)

        with pto.for_(0, qg, step=1) as q:
            q_i32 = scalar.index_cast(pto.i32, q)
            slot_b = pto.vmi.vbrc(pto.vmi.vload(idx_ub, q, size=1), size=VL)
            v_base = pto.vmi.vmul(slot_b, h_b, mask)
            sf_base = pto.vmi.vmul(slot_b, hs_b, mask)
            base_out = pto.vmi.vbrc(q_i32 * h, size=VL)
            for c in range(h // VL):
                chunk = pto.vmi.vbrc(pto.i32(c * VL), size=VL)
                # Gather 1: V[slot, c*VL : c*VL+VL]
                v_offs = pto.vmi.vadd(v_base, pto.vmi.vadd(arange, chunk, mask), mask)
                v = pto.vmi.vgather(v_ub, v_offs, mask)
                # Gather 2: Sf[slot, j//G]. Repeated index broadcasts the group scale.
                # g_id = c*(VL/G) + (lane >> log2(G)); Hs-stride, so gather not VL load.
                g_id = pto.vmi.vadd(
                    pto.vmi.vbrc(pto.i32(c * (VL // g)), size=VL),
                    pto.vmi.vshr(arange, GROUP_SHIFT, mask),
                    mask,
                )
                sf_offs = pto.vmi.vadd(sf_base, g_id, mask)
                s = pto.vmi.vgather(sf_ub, sf_offs, mask)
                y = pto.vmi.vmul(v, s, mask)
                outs = pto.vmi.vadd(base_out, pto.vmi.vadd(arange, chunk, mask), mask)
                pto.vmi.vscatter(y, out_ub, outs, mask)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_store(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))

    return kernel_sideband


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


def run_acl(n: int, h: int, g: int, qg: int, threads: int, arm: str, tag: str) -> int:
    v, sf, idx, pack, gold = synthesize(n, h, g, qg)
    kernel = build_kernel(n, h, g, qg, threads, arm)
    out_bytes = gold.nbytes
    idx_bytes = idx.nbytes

    lib = _load_acl()
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")

    def malloc(nbytes):
        p = ctypes.c_void_p()
        _chk(lib.aclrtMalloc(ctypes.byref(p), nbytes, _ACL_MEM_MALLOC_HUGE_FIRST), "malloc")
        return p

    def h2d(dst, arr, nbytes):
        _chk(
            lib.aclrtMemcpy(
                dst,
                nbytes,
                np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p),
                nbytes,
                _ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "h2d",
        )

    od = malloc(out_bytes)
    id_ = malloc(idx_bytes)
    h2d(id_, idx, idx_bytes)
    if arm == "interleave":
        pd = malloc(pack.nbytes)
        h2d(pd, pack, pack.nbytes)
        launch = (int(od.value), int(pd.value), int(id_.value))
        extra = (pd,)
    else:
        vd = malloc(v.nbytes)
        sfd = malloc(sf.nbytes)
        h2d(vd, v, v.nbytes)
        h2d(sfd, sf, sf.nbytes)
        launch = (int(od.value), int(vd.value), int(sfd.value), int(id_.value))
        extra = (vd, sfd)

    t0 = time.perf_counter()
    comp = kernel.compile()
    cs = time.perf_counter() - t0
    t0 = time.perf_counter()
    comp[1, st](*launch)
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    ls = time.perf_counter() - t0
    out = np.zeros((qg, h), dtype=np.float32)
    _chk(
        lib.aclrtMemcpy(out.ctypes.data_as(ctypes.c_void_p), out_bytes, od, out_bytes, _ACL_MEMCPY_DEVICE_TO_HOST),
        "d2h",
    )
    for p in (od, id_, *extra):
        lib.aclrtFree(p)
    lib.aclrtDestroyStream(st)
    lib.aclrtResetDevice(0)
    lib.aclFinalize()
    mx = float(np.max(np.abs(out - gold)))
    if not np.allclose(out, gold, rtol=1e-4, atol=1e-4):
        raise AssertionError(f"mismatch maxabs={mx}")
    print(f"PASS {tag} maxabs={mx:.6g} compile={cs:.3f}s launch={ls:.3f}s thr_mirror={threads}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="SP5d sideband vs interleave SIMD twin")
    ap.add_argument("N", nargs="?", type=int, default=N)
    ap.add_argument("H", nargs="?", type=int, default=H)
    ap.add_argument("G", nargs="?", type=int, default=G)
    ap.add_argument("Qg", nargs="?", type=int, default=QG)
    ap.add_argument("Thr", nargs="?", type=int, default=THR)
    ap.add_argument("arm", nargs="?", default="sideband", choices=("sideband", "interleave"))
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args(argv)
    if args.H != H or args.G != G:
        print(
            f"SP5d primary geometry is H={H} G={G} (VL={VL}, two groups per chunk); got H={args.H} G={args.G}",
            flush=True,
        )
        return 2
    if args.N <= 0 or args.Qg <= 0:
        print(f"bad shape N={args.N} Qg={args.Qg}", flush=True)
        return 2
    tag = f"sp5d_n{args.N}_h{args.H}_g{args.G}_qg{args.Qg}_t{args.Thr}_{args.arm}"
    if args.emit_mlir:
        print(build_kernel(args.N, args.H, args.G, args.Qg, args.Thr, args.arm).compile().mlir_text())
        return 0
    return run_acl(args.N, args.H, args.G, args.Qg, args.Thr, args.arm, tag)


if __name__ == "__main__":
    raise SystemExit(main())
