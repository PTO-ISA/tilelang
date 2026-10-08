# Case-3 design — SV5 / SV6 / SV7 / SV8 (group-reduce RF ladder → optional bcast)

> **Renumber (2026-09-23):** Case-3 ladder is **SV5 (G=16) / SV6 (G=32 mid) / SV7 (G=64/128) / SV8 (quant e2e)**.
> **Reframe (2026-10-06, Lok):** SV5–SV7 probe **reduce-group input RF capacity** (SIMD ~32 arch VL-regs vs SIMT per-thread / warp / multi-warp), not only post-reduce `sf_inv` residency. SV8 still owns bcast + live vs layout-spill.

**Date:** 2026-09-23 (Simt GREEN) · **updated 2026-10-06** (RF-capacity ladder)  
**Suite:** `/workspace/st_simtvf_parallel`

---

## Design principle

Quant / group-norm pipelines **reduce along group width G**, emit `sf_inv[R, CG]` (`CG=C/G`), then optionally **broadcast-multiply** back to activations. Case-3 splits that so we isolate:

1. **SV5 / SV6 / SV7** — reduce → `Y = sf_inv` on `[R,CG]` only (**no bcast**). The rung is **how the group inputs fit in RF** during absmax.
2. **SV8** — e2e bcast mul, with **live** `sf_inv` vs **layout-spill** expand `[R,CG]→[R,C]`.

Math (SV5–SV8 reduce stage):

```
sf_inv[i,g] = 1 / max( absmax(X[i, g*G : (g+1)*G]) , 1e-6 )
```

- SV5/SV6/SV7 IO: `Y = sf_inv` fp32 `[R,CG]`.
- SV8 IO: `Out[i,j] = X[i,j] * sf_inv[i, j//G]` fp16 `[R,C]`.

---

## SV5–SV7 — reduce-group RF capacity ladder (primary sensitivity)

Mental model (fp32 group elems; SIMD ≈ **32** arch VL-regs; SIMT warp = **32** threads):

| ST | G (primary) | CG @ C=128 | SIMD (~32 VL-regs) | SIMT | Expected schedule |
|----|------------:|-----------:|--------------------|------|-------------------|
| **SV5 small** | **16** (nvfp4-ish) | 8 | Whole group **fits** in arch regs | Whole group **fits in one thread** RF | Keep inputs in RF; pure reduce+inv; **no UB reload of inputs** |
| **SV6 mid** | **32** | 4 | **Does not fit** → slice / **reload** | One **warp** holds group (≈1 elem/thr) → **warp-level reduce only**, **no UB allreduce** of partials across warps | SIMD reload tax vs SIMT in-warp keep |
| **SV7 large** | **128** (+ try 64) | 1 (or 2) | Must **reload** / multi-slice | Needs **~4×32** threads to keep group → multi-warp + **UB** (or cross-warp) partial reduce | Multi-warp / UB path vs SIMD reload |

**Compare:** same math + IO; wall / IPC / instr (SIMT_LDS/STS vs RV_V*) across Simt vs SimdVF twins. Winner story = **reduce performance under RF cliff**, not bcast.

### Arms (planned under this reframe)

| ST | Primary arms (input residency during reduce) | Secondary / historical |
|----|----------------------------------------------|------------------------|
| SV5 | `keep_in_rf` (group in RF) vs optional `ub_reload` foil | Prior `keep_reg`/`ub_reload` on **post-reduce** `sf_inv` (2026-10-05) — keep as foil, not the rung definition |
| SV6 | Simt: `keep_in_warp` (no UB allreduce); VMI: `reload` / slice | Single-body baseline `sv6_r64_c128_g32_t32` (12.25 µs) until arms land |
| SV7 | Simt: `multiwarp_ub` partials; VMI: `reload` | Baseline G64/G128 already green (14.44 / 14.69 µs) |

Defaults still **R=64, C=128, T=32** (Simt) / **lanes=64** (VMI ABI) unless a shape forces T=128 for the 4-warp story.

---

## GPU vs NPU vs VMI

| layer | role |
|-------|------|
| **GPU** | MRF often holds mid-G; large G → shared partials |
| **NPU SimtVF** | Folded RF; prefer `Parallel(R,CG)+serial(G)`; abs via `T.max(v,-v)`; warp / multi-warp stories via `threads` + UB partials |
| **VMI SimdVF** | Same math/arms; working set prefer `alloc_shared`; **mode 1** when reloading, **mode 2/3** when group/scale stay live. Twin must **not** add AABBCC / vf_fuse / token_tile |

---

## Tags (oneshot primary)

| ST | tag | IO |
|----|-----|-----|
| SV5 | `sv5_r64_c128_g16_t32_*` | X fp16 `[64,128]` → Y fp32 `[64,8]` |
| SV6 | `sv6_r64_c128_g32_t32_*` | → Y `[64,4]` |
| SV7 | `sv7_r64_c128_g128_t32_*` (+ `g64`) | → Y `[64,1]` (or `[64,2]`) |
| SV8 live / spill_dist | `sv8_r64_c128_g16_t32_{live,spill_dist}` | → Out fp16 `[64,128]` |
| SV5V–SV8V | matching `*v_*` | same IO; SimdVF |

---

## Historical Simt µs (pre-reframe baselines — Ascend950PR_9599)

| tag | PASS | µs | note |
|-----|------|---:|------|
| sv5_r64_c128_g16_t32 (no-arm) | PASS | 9.34 | reduce+write only |
| sv5 …_ub_reload / keep_reg | PASS/FAIL | see [`ST_SV5_REDUCE_SMALL.md`](ST_SV5_REDUCE_SMALL.md) | post-reduce scale foil |
| sv6_r64_c128_g32_t32 | PASS | 12.25 | mid-G baseline |
| sv7_r64_c128_g64_t32 | PASS | 14.44 | |
| sv7_r64_c128_g128_t32 | PASS | 14.69 | CG=1 OK |
| sv8 …_live / spill_dist | PASS | 17.04 / 18.23 | bcast e2e |

New input-residency arms: **opsim TBD** after kernel rewrite.

---

## Reports

- [`ST_SV5_REDUCE_SMALL.md`](ST_SV5_REDUCE_SMALL.md)
- [`ST_SV6_REDUCE_MID.md`](ST_SV6_REDUCE_MID.md)
- [`ST_SV7_REDUCE_LARGE.md`](ST_SV7_REDUCE_LARGE.md)
- [`ST_SV8_CASE3_BCAST.md`](ST_SV8_CASE3_BCAST.md)
- [`ST_CASE3_VMI.md`](ST_CASE3_VMI.md)
- [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md)

## Run

```powershell
powershell -File .\run_via_laptop_sv5_sv6_sv8.ps1
powershell -File .\run_via_laptop_sv5v_sv6v_sv8v.ps1
```

Legacy retained: `sv5_bcast_multiconsumer.py`, `sv6_group_scale_remat.py`, `sv8_block_reduce_32x32.py`.

---

## IMPLEMENTED + MEASURED (2026-10-06, pto-b10, Ascend950PR_9599)

Work dir on pto-b10: `/tmp/pr272_sv567_rfladder_20261006/` (RESULT.md, VF_THEORY.md).
Simt arms all compile + opsim PASS; gold unchanged across arms (seed 5 for sv5,
seed 6 for sv6/sv7).

| rung | arm | tag | PASS | wall µs | VF µs | IPC_meas | C_meas | C_pred | ratio |
|------|-----|-----|------|--------:|------:|----:|-------:|-------:|------:|
| SV5 G=16 | `keep_in_rf` | `sv5_r64_c128_g16_t32_keep_in_rf` | PASS | 7.04 | 5.91 | 0.109 | 10641 | 7190 | 1.48 |
| SV5 G=16 | `ub_stream` | `sv5_r64_c128_g16_t32_ub_stream` | PASS | **4.18** | 3.06 | 0.218 | 5506 | 5660 | 0.97 |
| SV6 G=32 | `keep_in_warp` | `sv6_r64_c128_g32_t32_keep_in_warp` | PASS | 14.83 | 13.72 | 0.092 | 24697 | 34070 | 0.72 |
| SV6 G=32 | `reload` | `sv6_r64_c128_g32_t32_reload` | PASS | **12.58** | 11.53 | 0.106 | 20748 | 21020 | 0.99 |
| SV7 G=64 | `multiwarp_ub` | `sv7_r64_c128_g64_t32_multiwarp_ub` | PASS | **7.32** | 6.21 | 0.114 | 11182 | 6370 | 1.76 |
| SV7 G=64 | `reload` | `sv7_r64_c128_g64_t32_reload` | PASS | 14.54 | 13.52 | 0.090 | 24339 | 20900 | 1.16 |
| SV7 G=128 | `multiwarp_ub` | `sv7_r64_c128_g128_t32_multiwarp_ub` | PASS | 15.42 | 14.42 | 0.089 | 25948 | 21730 | 1.19 |
| SV7 G=128 | `reload` | `sv7_r64_c128_g128_t32_reload` | PASS | **14.58** | 13.57 | 0.089 | 24423 | 20840 | 1.17 |

Winner per rung (Simt):

- **SV5 (G=16): `ub_stream` wins, −40.6%** (4.18 vs 7.04 µs). Preloading the group
  into a fragment does *not* pay even at the rung where the group "fits": the
  fragment spills (`SIMT_LDK:194` / `SIMT_STK:194` in the histogram) and the
  extra RF store/load pass costs more than re-reading UB (`SIMT_LDS` is cheap
  and the 16-step loop is fully unrolled either way).
- **SV6 (G=32): `reload` wins, −15.2%** (12.58 vs 14.83 µs). The one-warp keep is
  real (no UB partials are written) but at G=32 the preload pass spills
  (`LDK/STK:256`) and both bodies lose their unroll, so the keep only adds a
  second branchy pass.
- **SV7: the winner flips with chunk width.** At **G=64** `multiwarp_ub` wins
  massively (**7.32 vs 14.54 µs, −50%**) because NW=4 makes the inner chunk 16
  steps → the chunk loop is **unrolled and branch-free** (no `SIMT_BRANCH` in its
  histogram) while the 64-step single stream is a branch loop. At **G=128** the
  chunk is 32 steps, the unroll is lost, and the UB partial traffic becomes pure
  overhead: `reload` wins by 5.4% (14.58 vs 15.42 µs). **So the UB allreduce at
  G=128 costs ≈ +0.84 µs (+5.8%) over a plain stream.**

The cross-cutting lesson is that on SimtVF the dominant knob at these shapes is
the **serial-loop unroll budget (≈16 steps)**, not RF capacity: every body with
an inner trip > 16 drops from IPC ≈0.22 to ≈0.09. Chunking a large G into
16-step pieces (`multiwarp_ub` at G=64) is worth far more than keeping inputs in
RF. See `reports/ST_VF_THEORY.md` for the model that predicts this (`W_loop`
term + branchy IPC) and the per-arm C_pred/C_meas ratios.
