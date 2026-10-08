# ST-SV5 — Small-G reduce (group inputs fit in RF)

**Date:** 2026-09-23 · **reframe 2026-10-06** (RF-capacity ladder)  
**Suite:** `/workspace/st_simtvf_parallel`  
**Kernel:** `kernels/sv5_reduce_small_eltwise.py`  
**Umbrella:** [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md)

---

## Sensitivity (primary)

**Small group width G≈16** (nvfp4-ish): whole group **inputs** fit in **SIMD ~32 arch VL-regs** and in **one SIMT thread** RF. Probe pure reduce+inv with **keep-in-RF** (no UB reload of inputs). Contrast optional UB-reload foil.

Not the primary story: post-reduce `sf_inv` keep vs ub (that foil still exists from 2026-10-05).

## Math / IO

`Y[i,g] = 1 / max(absmax(X[i, g*G:(g+1)*G]), 1e-6)` — **no bcast**.  
Defaults: **R=64, C=128, G=16 → CG=8, T=32**. Y fp32 `[R,CG]`.

## Schedule / arms

| arm | meaning | VMI mode |
|-----|---------|----------|
| **`keep_in_rf`** (planned) | Load group into RF / VL-regs; absmax in place; write `sf_inv` | **2/3** fused stay |
| **`ub_reload`** (foil) | Stream from UB each serial step | **1** |
| historical `keep_reg` / `ub_reload` | Post-reduce **scale** residency + subsequent serial(G) | keep→3, ub→1 |

## Turning knobs

| knob | values | cliff |
|------|--------|-------|
| G | **16** primary; optional 8 | still ≤32 SIMD regs / one-thread RF |
| R,C,T | 64,128,32 | elems/thr for Parallel fold |

## Baseline µs (Simt, Ascend950PR_9599)

| tag | PASS | µs | note |
|-----|------|---:|------|
| sv5_r64_c128_g16_t32 | PASS | 9.34 | pre-arm reduce+write |
| …_g16_t32_ub_reload | PASS | **7.74** | post-reduce scale foil winner |
| …_g16_t32_keep_reg | PASS | 10.82 | post-reduce; spills LDK/STK |

`keep_in_rf` input-residency arm: **TBD** (kernel rewrite).

## VMI twin

`sv5v_*` — same G=16; prefer shared working set; lanes∈{64,128,256}. Opsim pending (prior CG/lanes ABI issues on Case-3).

---

## RESULT (2026-10-06, pto-b10, Ascend950PR_9599) — primary arms landed

`kernels/sv5_reduce_small_eltwise.py` now ships four arms:
`keep_in_rf | ub_stream` (primary, group-input RF capacity) and the historical
`keep_reg | ub_reload` (post-reduce scale residency, rows below still valid).

| tag | PASS | wall µs | VF µs | IPC_meas | maxabs |
|-----|------|--------:|------:|---------:|--------|
| `sv5_r64_c128_g16_t32_keep_in_rf` | PASS | 7.04 | 5.91 | 0.110 | 1.19e-07 |
| `sv5_r64_c128_g16_t32_ub_stream` | PASS | **4.18** | 3.06 | 0.220 | 1.19e-07 |

**Winner: `ub_stream`, −2.86 µs (−40.6%).** The "keep" rung does **not** pay even
at G=16 where the group nominally fits one thread's RF:

- the fragment preload spills — histogram shows `SIMT_STK:194` + `SIMT_LDK:194`
  (local stack store/load) and only `SIMT_LDS:128` / `SIMT_F2F:128`, i.e. the
  inputs land in the stack, not in registers;
- it also costs a whole extra element-wise pass (RF store + RF load) that the
  streaming arm does not have, while `SIMT_LDS` out of UB is cheap and both
  16-step loops unroll fully.

Implementation note (compile blocker fixed): preload and reduce must be **two
separate `T.Parallel` loops**. Writing and reading `xf` inside one Parallel body
fails LayoutInference with
`RecordBufferAccess: xf: (i,g,t) and (i,g,t)` — one `ParallelOp` records only one
index pattern per buffer, and the two `serial(G)` loop vars are different Vars.

Theory check (`reports/ST_VF_THEORY.md`): `ub_stream` C_pred 5660 vs C_meas 5506
(ratio 0.97); `keep_in_rf` C_pred 7190 vs C_meas 10641 (ratio 1.48 — the unmodelled
spill).

### Historical rows (post-reduce scale residency, 2026-10-05) — unchanged
`sv5_*_keep_reg` / `sv5_*_ub_reload` rows in `PASS_TABLE_sv1_sv9.txt` remain
valid for the *other* question (where `sf_inv` lives for the subsequent compute),
including the R≥128 `keep_reg` RF-clobber FAILs.
