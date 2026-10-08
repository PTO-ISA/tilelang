# ST PTO-DSL Phase 2 — RF ladder compare (KEEP vs STREAM)

**Date:** 2026-09-29 ~13:55 HKT  
**SOC:** Ascend950PR_9599  
**Sources:** `kernels_ptodsl/` · **Audit:** [`ST_RF_KEEP_STREAM_AUDIT.md`](ST_RF_KEEP_STREAM_AUDIT.md)

## Contract (Lok)

VMI/ptodsl must make **keep vs stream/reload** source-visible. Turning point: **KEEP scales** in RF; **STREAM rows**. Remat of scales must pay **UB store + mem_bar** when modeling real UB visibility (TileKernels-vmi `VST_VLD`) — not a free RF-hot reload.

## RF budget (Ascend950PR_9599)

| Constant | Value | Source |
|----------|------:|--------|
| VL | 256 B = **64 f32** | ISA |
| **VF stack** | **6144 B** | ptoas `exceeded vf stack size (6144)` |
| C=128 row KEEP | 64×256 = **16384 B** | illegal |
| C=256 row KEEP | 128×256 = **32768 B** | illegal |

I/O: DSL **f32** (Simt A f16). C=32 **skipped** (not VL-aligned for f32 VL=64); small geom = **C=64**.

## Why scale KEEP did not beat `full_reload`

`full_reload` remats with `vload(scale_s,i,size=1)` **without** `mem_bar`. The producer VST→consumer VLD of scales is not forced through UB visibility — remat stays cheap (often RF-hot). Meanwhile `scale_keep_stream` pays **pack lane-select** (`vcmp`/`vsel`/`vcmax`/`vbrc` per i) so **RVECEX is higher** on KEEP than on bare reload. Wall: KEEP 1.51 µs vs reload **1.47** µs (C=128).

## Arm `scale_ub_membar` — remat = UB + mem_bar

After publishing all scales to UB:
1. phase-boundary `pto.mem_bar("VST_VLD")` + `pto.pipe_barrier("V")` (TileKernels-vmi / pto-vmi style)
2. each consumer i: **`mem_bar("VST_VLD")` again** then `vload(scale,size=1)` + STREAM row

Opsim shows **`RV_SMEM_BAR` ×33** (1 phase + 32 per-use) — real barrier tax.

### VF-time comparison (measured; no invented metrics)

**Recipes**
- **wall µs** = `core0.veccore0` `duration_time(us)` from msprof / opsim log
- **RVECEX cyc** = Σ `cycles` where `pipe==RVECEX` in latest `core0.veccore0_instr_exe.csv` (VF compute/EX-pipe work)
- **RVECLD cyc** = Σ `pipe==RVECLD` (includes `RV_SMEM_BAR` + vector loads)
- **RV_SMEM_BAR** = `call_count` / `cycles` for instr `RV_SMEM_BAR` (absent → —)
- **VF PUSHQ cyc** = instr `VF` on pipe `PUSHQ` (VF **scope launch** framing only — **not** body time)
- **EXIPC** = native `EXIPC`/`IFU` row if present; else **—** (ptodsl dumps here have none)

| Tag | wall µs | RVECEX | RVECLD | RV_SMEM_BAR cc/cyc | VF PUSHQ | EXIPC |
|-----|--------:|-------:|-------:|-------------------:|---------:|------:|
| `sv2d_r32_c64_t32_scale_keep_stream` | **1.18** | 4223 | 585 | — | 429 | — |
| `sv2d_r32_c64_t32_full_reload` | **1.10** | 3128 | 882 | — | 350 | — |
| `sv2d_r32_c64_t32_scale_ub_membar` | **1.79** | 3128 | 3511 | **33 / 1442** | 1588 | — |
| `sv2d_r32_c128_t32_scale_keep_stream` | **1.51** | 5407 | 1161 | — | 525 | — |
| `sv2d_r32_c128_t32_full_reload` | **1.47** | 4312 | 1458 | — | 464 | — |
| `sv2d_r32_c128_t32_scale_ub_membar` | **2.16** | 4312 | 4107 | **33 / 1446** | 1685 | — |
| `sv2d_r32_c256_t32_scale_keep_stream` | **2.25** | 7775 | 2313 | — | 772 | — |
| `sv2d_r32_c256_t32_full_reload` | **2.22** | 6680 | 2600 | — | 716 | — |
| `sv2d_r32_c256_t32_scale_ub_membar` | **2.90** | 6680 | 5459 | **33 / 1521** | 1935 | — |

Raw harvest: [`sv2d_membar_harvest.json`](sv2d_membar_harvest.json).

### Reading the table

| Comparison | C=128 wall | Takeaway |
|------------|-----------:|----------|
| KEEP vs **full_reload** | 1.51 vs **1.47** | Reload wins — remat **under-taxed** (no mem_bar); KEEP pays pack-select RVECEX |
| KEEP vs **scale_ub_membar** | **1.51** vs 2.16 | **KEEP wins** once remat = UB + VST_VLD (±0.65 µs) |
| full_reload vs scale_ub_membar | 1.47 vs 2.16 | mem_bar tax ≈ **+0.69 µs** / +1446 SMEM_BAR cyc |

Same pattern at C=64 (−0.61 µs KEEP vs membar) and C=256 (−0.65 µs).

**Why RF KEEP should win when remat = UB+membar:** scales reused across C/VL row chunks stay in `scale_pack` with **no** `RV_SMEM_BAR` on the consumer path; remat must re-observe UB (`RV_SMEM_BAR` + `RV_VLDI`) every i. Pack-select still costs RVECEX on KEEP, but wall and RVECLD are dominated by the barrier tax on the membar arm.

## row_keep cliff (unchanged)

| Tag | Result |
|-----|--------|
| `sv2d_r32_c128_t32_row_keep_full` | **FAIL** spill 14368 > 6144 |
| `sv2d_r32_c256_t32_row_keep_full` | **FAIL** spill 36128 > 6144 |

## Other Phase2 tags (unchanged)

| Tag | Opsim | µs |
|-----|-------|---:|
| `sv4d_e256_b8_t32_keep_idx` | PASS | 1.02 |
| `sv4d_e256_b8_t32_remat_idx` | PASS | 1.02 |
| `sv9d_e256_k8_t32_keep` | PASS | 1.09 |
| `sv9d_e256_k8_t32_remat_scores` | PASS | 1.0 |
| `sv9d_e256_k8_t32_remat_idx` | PASS | 1.09 |

## Push

See git log — `scale_ub_membar` + VF-time table.


## SV2d fold rewrite (2026-09-29 ~14:30 HKT)

**Semantics change:** gold was scalar row max (`vcmax`); now **per-lane vmax fold** across channel chunks. Scale stays a VL register — KEEP has no pack/lane-select tax, so KEEP wins naturally vs VL remat.

| Tag | Opsim | wall µs | RVECEX | RVECLD | notes |
|-----|-------|--------:|-------:|-------:|-------|
| `sv2d_r32_c64_t32_fold_scale_keep` | **PASS** | **1.01** | 1394 | 288 | optional |
| `sv2d_r32_c128_t32_fold_scale_keep` | **PASS** | **1.34** | 2578 | 864 | **primary** |
| `sv2d_r32_c128_t32_fold_scale_reload` | **PASS** | **1.87** | 2578 | 3224 | contrast (UB+membar) |
| `sv2d_r32_c256_t32_fold_scale_keep` | not run | — | — | — | optional |
| `sv2d_r32_c256_t32_fold_scale_reload` | not run | — | — | — | optional |

**Reading:** C=128 KEEP **1.34** vs reload **1.87** (−0.53 µs). Same RVECEX (2578) — fold compute identical; reload pays RVECLD (3224 vs 864) from `mem_bar(VST_VLD)` + VL scale reload. KEEP wins naturally: scale already VL in RF, no lane-select tax.

## SV2 A ↔ D — scale ALWAYS KEEP; input KEEP vs STREAM (2026-09-29)

| A arm | D arm | Scale | Input | Intended R |
|-------|-------|-------|-------|------------|
| `input_keep` | `input_keep` | KEEP fragment / VL | KEEP | ≈16 |
| `input_stream` / `frag_live` | `input_stream` / `fold_scale_keep` | KEEP | STREAM | ≥32 |
| `reload` (demoted) | `fold_scale_reload` (demoted) | remat foil | — | — |

Gold: per-lane fold, not scalar row-max. Opsim: R16 keep vs R32/64 stream.


### Opsim input KEEP/STREAM (2026-09-29 ~14:48 HKT)

| Tag | Opsim | µs |
|-----|-------|---:|
| `sv2d_r16_c64_t32_input_keep` | PASS | 0.91 |
| `sv2d_r32_c64_t32_input_stream` | PASS | 1.01 |
| `sv2d_r64_c64_t32_input_stream` | PASS | 1.38 |
| `sv2d_r32_c128_t32_input_stream` | PASS | 1.34 |
| `sv2d_r32_c64_t32_input_keep` | PASS | 1.05 |
