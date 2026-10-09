#!/usr/bin/env python3
"""ST-SP2 — Acc KEEP/remat **and** sf0_w0/sf1_w1 tax (one case).

**Schedule arms (same ST; ex-SP2x folded — Lok 2026-10-07):**
1. Acc **KEEP vs remat** on indexed gather→reduce (Pos frozen via shared
   ``pos_row``). Tags: ``keep_acc`` / ``remat_acc``.
2. Under fixed Acc KEEP, ``sf0_w0`` vs ``sf1_w1`` compute tax.

**Legacy Pos arms** (Acc KEEP): ``keep_pos`` / ``remat_pos`` — historical PASS.

**Workload remark:** MoE ``reduce_fused`` (gather by fused slot Pos).

Simplified: expand Sf → full-H sideband once, then ``Parallel(H)`` gather.

Tags:
  ``sp2_t{T}_k{K}_h{H}_g{G}_t{Thr}_{sf0|sf1}_{w0|w1}_{keep_acc|remat_acc|keep_pos|remat_pos}``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402
from vf_mode import vf_region  # noqa: E402

_SCHED = ("keep_acc", "remat_acc", "keep_pos", "remat_pos")


def build_sp2(
    T_: int,
    K: int,
    H: int,
    G: int,
    threads: int,
    with_sf: bool,
    with_w: bool,
    sched: str,
    Eexp: int | None = None,
):
    """sched: keep_acc | remat_acc | keep_pos | remat_pos."""
    import tilelang.ascend.language as T

    Hs = H // G
    if Eexp is None:
        Eexp = T_ * K

    remat_pos = sched == "remat_pos"
    # Acc remat: accumulate in shared (reload each Parallel) vs fragment KEEP
    remat_acc = sched == "remat_acc"
    # Pos frozen (keep snapshot) for Acc arms; remat_pos reloads each k
    freeze_pos = sched in ("keep_acc", "remat_acc", "keep_pos")

    @T.prim_func
    def main(
        Vexp: T.Tensor((Eexp, H), "float32"),
        Sf: T.Tensor((Eexp, Hs), "float32"),
        W: T.Tensor((T_, K), "float32"),
        Pos: T.Tensor((T_, K), "int32"),
        Out: T.Tensor((T_, H), "float32"),
    ):
        with T.Kernel(1):
            v_ub = T.alloc_shared((Eexp, H), "float32")
            sf_ub = T.alloc_shared((Eexp, Hs), "float32")
            sf_full = T.alloc_shared((Eexp, H), "float32")
            w_ub = T.alloc_shared((T_, K), "float32")
            pos_ub = T.alloc_shared((T_, K), "int32")
            pos_row = T.alloc_shared((K,), "int32")
            # Acc remat staging (shared); unused path still allocated for layout stability
            acc_ub = T.alloc_shared((H,), "float32")
            out_ub = T.alloc_shared((T_, H), "float32")
            T.copy(Vexp, v_ub)
            T.copy(Sf, sf_ub)
            T.copy(W, w_ub)
            T.copy(Pos, pos_ub)
            with vf_region(threads):
                acc = T.alloc_fragment((H,), "float32")
                if with_sf:
                    for p, g in T.Parallel(Eexp, Hs):
                        s = sf_ub[p, g]
                        for tt in T.serial(G):
                            sf_full[p, g * G + tt] = s
                for t in T.serial(T_):
                    if remat_acc:
                        for j in T.Parallel(H):
                            acc_ub[j] = T.float32(0.0)
                    else:
                        for j in T.Parallel(H):
                            acc[j] = T.float32(0.0)
                    if freeze_pos:
                        for k in T.serial(K):
                            pos_row[k] = pos_ub[t, k]
                    for k in T.serial(K):
                        if remat_pos:
                            pos = pos_ub[t, k]
                        else:
                            pos = pos_row[k]
                        if pos >= 0:
                            wk = T.alloc_var("float32")
                            wk = T.float32(1.0)
                            if with_w:
                                wk = w_ub[t, k]
                            if remat_acc:
                                # Acc remat: RMW through shared (no fragment KEEP across K)
                                for j in T.Parallel(H):
                                    s = wk
                                    if with_sf:
                                        s = wk * sf_full[pos, j]
                                    acc_ub[j] = acc_ub[j] + v_ub[pos, j] * s
                            else:
                                for j in T.Parallel(H):
                                    s = wk
                                    if with_sf:
                                        s = wk * sf_full[pos, j]
                                    acc[j] = acc[j] + v_ub[pos, j] * s
                    if remat_acc:
                        for j in T.Parallel(H):
                            out_ub[t, j] = acc_ub[j]
                    else:
                        for j in T.Parallel(H):
                            out_ub[t, j] = acc[j]
            T.copy(out_ub, Out)

    return main


def main():
    import os

    set_out(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
    boot()
    T_ = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    G = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    sf_arm = sys.argv[6] if len(sys.argv) > 6 else "sf1"
    w_arm = sys.argv[7] if len(sys.argv) > 7 else "w1"
    sched = sys.argv[8] if len(sys.argv) > 8 else "keep_acc"
    if sf_arm not in ("sf0", "sf1") or w_arm not in ("w0", "w1"):
        print(f"bad sf/w arm {sf_arm}/{w_arm}", flush=True)
        return 2
    if sched not in _SCHED:
        print(f"unknown sched {sched!r}; use {'|'.join(_SCHED)}", flush=True)
        return 2
    tag = f"sp2_t{T_}_k{K}_h{H}_g{G}_t{threads}_{sf_arm}_{w_arm}_{sched}"
    so = compile_prim(
        build_sp2(T_, K, H, G, threads, sf_arm == "sf1", w_arm == "w1", sched),
        tag,
        target="ascend",
    )
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
