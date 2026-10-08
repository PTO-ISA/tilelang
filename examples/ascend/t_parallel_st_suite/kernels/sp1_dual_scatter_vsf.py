#!/usr/bin/env python3
"""ST-SP1 — Pos KEEP vs remat under dual live (V, Sf) co-scatter.

**Single sensitivity:** index residency (KEEP vs remat) while two payloads
stay live and co-store to the same fused slot. SV4-cousin + dual write.

**Workload remark:** MoE ``expand_to_fused_with_sf`` / ``get_fused_mapping``.
Expert id only marks pad holes — not the measured axis.

Arms:
- ``keep_pos``  — Pos[t,:] snapshotted once into shared ``pos_row[K]`` across K.
  **Gap:** fragment / true VRF Pos KEEP failed Ascend layout at H=128; shared
  ``pos_row`` is a stand-in. VMI / *d twin should try real VRF KEEP for the cliff.
- ``remat_pos`` — reload ``pos_ub[t,k]`` each scatter (SV4-style index remat)

Do not bolt UE8M0 / FP4 / col-major onto this ST (those = SP3 / SP6 / SP5).

Tags: ``sp1_t{T}_k{K}_h{H}_g{G}_t{Thr}_{keep_pos,remat_pos}``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_asc_harness import boot, compile_prim, set_out  # noqa: E402


def build_sp1(T_: int, K: int, H: int, G: int, threads: int, arm: str, Eexp: int | None = None):
    import tilelang.ascend.language as T

    Hs = H // G
    if Eexp is None:
        Eexp = T_ * K

    if arm == "remat_pos":

        @T.prim_func
        def main(
            V: T.Tensor((T_, H), "float32"),
            Sf: T.Tensor((T_, Hs), "float32"),
            Pos: T.Tensor((T_, K), "int32"),
            Expert: T.Tensor((Eexp,), "int32"),
            Vexp: T.Tensor((Eexp, H), "float32"),
            Sfexp: T.Tensor((Eexp, Hs), "float32"),
        ):
            with T.Kernel(1):
                v_ub = T.alloc_shared((T_, H), "float32")
                sf_ub = T.alloc_shared((T_, Hs), "float32")
                pos_ub = T.alloc_shared((T_, K), "int32")
                exp_ub = T.alloc_shared((Eexp,), "int32")
                vexp_ub = T.alloc_shared((Eexp, H), "float32")
                sfexp_ub = T.alloc_shared((Eexp, Hs), "float32")
                T.copy(V, v_ub)
                T.copy(Sf, sf_ub)
                T.copy(Pos, pos_ub)
                T.copy(Expert, exp_ub)
                with T.SimtVF(threads=threads):
                    v_frag = T.alloc_fragment((H,), "float32")
                    sf_frag = T.alloc_fragment((Hs,), "float32")
                    for p, j in T.Parallel(Eexp, H):
                        if exp_ub[p] < 0:
                            vexp_ub[p, j] = T.float32(0.0)
                    for p, j in T.Parallel(Eexp, Hs):
                        if exp_ub[p] < 0:
                            sfexp_ub[p, j] = T.float32(0.0)
                    for t in T.serial(T_):
                        for j in T.Parallel(H):
                            v_frag[j] = v_ub[t, j]
                        for j in T.Parallel(Hs):
                            sf_frag[j] = sf_ub[t, j]
                        for k in T.serial(K):
                            pos = pos_ub[t, k]  # remat each k
                            if pos >= 0:
                                for j in T.Parallel(H):
                                    vexp_ub[pos, j] = v_frag[j]
                                for j in T.Parallel(Hs):
                                    sfexp_ub[pos, j] = sf_frag[j]
                T.copy(vexp_ub, Vexp)
                T.copy(sfexp_ub, Sfexp)

        return main

    # keep_pos: row snapshot in shared (fragment Pos KEEP failed layout @ H=128)
    @T.prim_func
    def main(
        V: T.Tensor((T_, H), "float32"),
        Sf: T.Tensor((T_, Hs), "float32"),
        Pos: T.Tensor((T_, K), "int32"),
        Expert: T.Tensor((Eexp,), "int32"),
        Vexp: T.Tensor((Eexp, H), "float32"),
        Sfexp: T.Tensor((Eexp, Hs), "float32"),
    ):
        with T.Kernel(1):
            v_ub = T.alloc_shared((T_, H), "float32")
            sf_ub = T.alloc_shared((T_, Hs), "float32")
            pos_ub = T.alloc_shared((T_, K), "int32")
            exp_ub = T.alloc_shared((Eexp,), "int32")
            vexp_ub = T.alloc_shared((Eexp, H), "float32")
            sfexp_ub = T.alloc_shared((Eexp, Hs), "float32")
            pos_row = T.alloc_shared((K,), "int32")
            T.copy(V, v_ub)
            T.copy(Sf, sf_ub)
            T.copy(Pos, pos_ub)
            T.copy(Expert, exp_ub)
            with T.SimtVF(threads=threads):
                v_frag = T.alloc_fragment((H,), "float32")
                sf_frag = T.alloc_fragment((Hs,), "float32")
                for p, j in T.Parallel(Eexp, H):
                    if exp_ub[p] < 0:
                        vexp_ub[p, j] = T.float32(0.0)
                for p, j in T.Parallel(Eexp, Hs):
                    if exp_ub[p] < 0:
                        sfexp_ub[p, j] = T.float32(0.0)
                for t in T.serial(T_):
                    for j in T.Parallel(H):
                        v_frag[j] = v_ub[t, j]
                    for j in T.Parallel(Hs):
                        sf_frag[j] = sf_ub[t, j]
                    for k in T.serial(K):
                        pos_row[k] = pos_ub[t, k]  # snapshot once
                    for k in T.serial(K):
                        pos = pos_row[k]
                        if pos >= 0:
                            for j in T.Parallel(H):
                                vexp_ub[pos, j] = v_frag[j]
                            for j in T.Parallel(Hs):
                                sfexp_ub[pos, j] = sf_frag[j]
            T.copy(vexp_ub, Vexp)
            T.copy(sfexp_ub, Sfexp)

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
    arm = sys.argv[6] if len(sys.argv) > 6 else "keep_pos"
    if arm not in ("keep_pos", "remat_pos"):
        print(f"unknown arm {arm!r}; use keep_pos|remat_pos", flush=True)
        return 2
    tag = f"sp1_t{T_}_k{K}_h{H}_g{G}_t{threads}_{arm}"
    so = compile_prim(build_sp1(T_, K, H, G, threads, arm), tag, target="ascend")
    print("SO", so) if so else None
    return 0 if so else 1


if __name__ == "__main__":
    raise SystemExit(main())
