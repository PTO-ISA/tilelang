# ST SV2 shape alignment — why R32×C32 diverged from *d C64/C128

**Date:** 2026-10-07 HKT  
**Owner:** REGRESSION-2 / Lok Chan  
**Scope:** Align Simt A SV2 tags to the same R/C/arm cases as PTO-DSL Layer-D `*d`. No git push / no PR #272.

## Why shapes looked different

`PASS_TABLE_sv1_sv9.txt` still carries **historical** `sv2_r32_c32_t32_{frag_live,reload}` rows from before the 2026-09-29 KEEP/STREAM contract. Those used C=32 (NCH=1 at T=32) because the early Simt baseline folded along a single LANES-wide chunk.

After Lok’s contract, **both** Simt A and *d use **scale ALWAYS KEEP** + **input KEEP@R≈16 vs STREAM@R≥32**. Layer-D is f32 VL=64, so **C must be a multiple of 64** (C=64/128). That alone forces *d primaries onto `R16×C64`, `R32×C64`, `R32×C128` — not R32×C32.

`ST_RF_KEEP_STREAM_AUDIT.md` already recorded partial aligned walls (Simt `r16_c32` keep 1.16; `r32_c64` stream 3.60 vs *d `r16_c64` keep ~0.92; `r32_c64` stream 1.01) but not a full same-shape matrix, and Simt never got a primary `R32×C128` row in the PASS table.

## Matched shape plan (this run)

| Role | Simt A tag | *d tag | Notes |
|------|------------|--------|-------|
| input KEEP (small R) | `sv2_r16_c64_t32_input_keep` | `sv2d_r16_c64_t32_input_keep` | If Simt C64 keep **COMPILE_FAIL** (layout: x frag R16×C64 + scale), fall back to `sv2_r16_c32_t32_input_keep` and document the ABI gap |
| input STREAM | `sv2_r32_c64_t32_input_stream` | `sv2d_r32_c64_t32_input_stream` | Primary stream |
| input STREAM wide-C | `sv2_r32_c128_t32_input_stream` | `sv2d_r32_c128_t32_input_stream` | Match *d C128 |
| optional foil | `sv2_r32_c64_t32_reload` | `sv2d_r32_c64_t32_fold_scale_reload` | Scale-remat foil — not primary |

Historical `sv2_r32_c32_t32_{frag_live,reload}` stay in the PASS table labeled **HISTORICAL**; they are not matched-shape rows for the e2e Simt vs Simd table.

## Environment

- Camodel: Ascend950PR_9599 via `sim_dsl.sh`, `.venv-npu` + cann_91b3, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- Work dirs on pto-b10: `/tmp/st_simtvf_parallel` (Simt), `/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d/` (*d)

## Opsim results (2026-10-07 HKT, Ascend950PR_9599)

| Tag | Result | wall µs | C_meas | body_instr | IPC |
|-----|--------|--------:|-------:|-----------:|----:|
| `sv2_r16_c64_t32_input_keep` | **COMPILE_FAIL** | — | — | — | — |
| `sv2_r16_c32_t32_input_keep` | **PASS** (fallback) | **1.16** | 530 | 192 | 0.362 |
| `sv2_r32_c64_t32_input_stream` | **PASS** | **3.60** | 4941 | 735 | 0.149 |
| `sv2_r32_c128_t32_input_stream` | **PASS** | **8.49** | 13500 | 1579 | 0.117 |
| `sv2_r32_c64_t32_reload` | **PASS** (foil) | **4.42** | 6411 | 833 | 0.130 |

**COMPILE_FAIL detail (r16_c64 keep):** `layout_inference.cc` — `no available layout found` (x fragment R16×C64 + scale fragment). ABI gap vs *d which keeps rows in VL=64 vregs at C64.

**Matched *d walls (from ptodsl_sv1_sv9d harvest):** `sv2d_r16_c64_t32_input_keep` 0.92 µs; `sv2d_r32_c64_t32_input_stream` 1.01 µs; `sv2d_r32_c128_t32_input_stream` 1.34 µs; `sv2d_r32_c64_t32_fold_scale_reload` 1.45 µs.

**E2e table:** [`SIMT_SIMD_E2E_COMPARE_20261007.md`](SIMT_SIMD_E2E_COMPARE_20261007.md) / [`simt_simd_e2e_20261007.tsv`](simt_simd_e2e_20261007.tsv)
