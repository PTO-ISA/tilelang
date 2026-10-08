# VF_total — Simt `T.Parallel` vs SIMD PTO-DSL `*d` (best schedule)

**Date:** 2026-10-08 HKT · **SOC:** Ascend950PR_9599 opsim (pto-b10) · **No git push / no PR #272.**

## Metric

- **VF_total** = Σ `cycles` over **every** VF launch row in `core0.veccore0_instr_exe.csv` (all addrs × call_count). Single-core kernels (1 core dir each).
  - **Simt:** Σ `VF_SIMT` rows. Every Simt CSV inspected (25 tags on pto-b10 + SV8 live box copy) has exactly **1 VF_SIMT row / 1 launch**, so VF_total = the old `C_meas`.
  - **`*d`:** Σ `VF`/`VF_SIMD` rows; fallback Σ RVEC pipes only if no VF row (stated per row in TSV `d_metric`).
- **`*d` best schedule** = lowest VF_total among the `*d` schedule arms at the **same shape** (arm named). Simt shows every sensitivity arm; ★ = Simt own best at that shape.
- **ratio** = Simt VF_total ÷ `*d` best VF_total (>1 ⇒ `*d` uses fewer VF cycles).
- wall µs (`core0.veccore0`) kept as a secondary column only.

## Sources

- **Simt SP1–SP6, SV4–SV7:** re-harvested 2026-10-08 from the 2026-10-07 opsim CSVs on pto-b10 (`/tmp/st_simtvf_parallel/opsim_sp*`, `/tmp/pr272_sv567_rfladder_20261006/`, `/tmp/pr272_sv4_cliff_20261005/out/`). SV4–SV7 values match the 10-07 TSV exactly.
- **Simt SV1/SV2/SV3/SV8/SV9:** reused `simt_simd_e2e_20261007.tsv` (`VF_SIMT.cycles`; Simt is single-launch, verified on 26/26 surviving CSVs; SV8 live box copy = 28,402 = TSV).
- **Simt CF1–CF6:** reused `simt_simd_e2e_cf_20261007.tsv` (from `harvest_cf_summary.py` → `harvest_simt_vmi_compare.analyze`, which sums all `VF_SIMT` rows by name ⇒ full sum). Raw CF Simt CSVs no longer on pto-b10.
- **`*d` (all SV/CF/SP):** **re-run 2026-10-08** from box `kernels_ptodsl/` (synced to `/tmp/vfc_20261008/suite`), out `/tmp/vfc_20261008/out_d/`; harvester `vf_total.py`. Old SV `*d` opsim dirs were gone from `/tmp/st_simtvf_parallel` (wiped by the SP run), so SV `*d` was re-run rather than reused. Result: **all 24 SV `*d` VF values reproduce the 10-07 TSV `*d C` exactly** and every `*d` kernel has exactly 1 VF row / 1 launch (no RVEC fallback needed anywhere). Run with 4-way parallel opsim; SP `*d` retried serially in `/tmp/vfc_20261008/out_d2/`.
- **SP `*d` alignment fix re-run 2026-10-08 (later):** `/tmp/sp_align_20261008/out/` — SP1d vscatter PASS (31424 VF), new SP2d×4 PASS (837/2301), SP3d e8m0 PASS (311); fp32 confirmed 349. Harvest: `reports/vfc_20261008/vf_total_sp_align.jsonl`.

## Caveats for slides

- VF_total counts only cycles inside VF launches on both sides; MTE (GM↔UB) and scalar work outside the VF are excluded on purpose (vector-part focus). Example: SV1 Simt 727 vs `*d` 59 VF cyc, while walls are 1.21 vs 0.89 µs.
- `*d` DSL I/O is f32; several Simt kernels use fp16 A/X (see `ST_PTODSL_PLAN.md`). Same math, not always same dtype.
- CF `*d` has one schedule per case; `pnear`/`pfat` are data skew, so the `*d` column is the matched-data `*d`, not a schedule pick.
- SP3 `*d` arms are data formats (fp32/e8m0), matched per arm.

## SV — vector / RF sensitivity

| case | shape | Simt arm | Simt VF_total | `*d` best arm | `*d` VF_total | Simt/`*d` | Simt µs | `*d` µs | `*d` arms (VF_total) |
|------|-------|----------|--------------:|---------------|--------------:|----------:|--------:|--------:|----------------------|
| SV1 | E256 | stream | 727 | stream | 59 | 12.32 | 1.21 | 0.89 | stream=59 |
| SV1 | E2048 | stream | 2,625 | — (no *d at E2048) | — | — | 2.58 | — | — |
| SV2 | R16×C64 | input_keep | — (COMPILE_FAIL (layout_inference: no available layout found)) | input_keep | 136 | — | — | 0.92 | input_keep=136 |
| SV2 | R16×C32 | input_keep (ABI fallback) | 530 | — (no *d at C32) | — | — | 1.16 | — | — |
| SV2 | R32×C64 | input_stream ★ | 4,941 | input_stream | 172 | 28.73 | 3.60 | 1.01 | input_stream=172, fold_scale_keep=172, fold_scale_reload=976 |
| SV2 | R32×C64 | fold_scale_reload | 6,411 | input_stream | 172 | 37.27 | 4.42 | 1.01 | input_stream=172, fold_scale_keep=172, fold_scale_reload=976 |
| SV2 | R32×C128 | input_stream | 13,500 | input_stream | 253 | 53.36 | 8.49 | 1.34 | input_stream=253 |
| SV3 | M24 VL64 K16 | keep | 9,105 | keep | 824 | 11.05 | 6.79 | 3.14 | keep=824 |
| SV3 | M32 VL64 K16 | keep | 11,874 | split_cm16 | 6,469 | 1.84 | 8.63 | 6.88 | keep=16,230, split_cm16=6,469 |
| SV3 | M32 VL64 K16 | split_cm16 ★ | 10,384 | split_cm16 | 6,469 | 1.61 | 7.79 | 6.88 | keep=16,230, split_cm16=6,469 |
| SV4 | E256 B8 | keep_idx | 1,372 | keep_idx | 120 | 11.43 | 1.67 | 1.02 | keep_idx=120, remat_idx=120 |
| SV4 | E256 B8 | remat_idx_calc ★ | 1,156 | keep_idx | 120 | 9.63 | 1.62 | 1.02 | keep_idx=120, remat_idx=120 |
| SV5 | R64 C128 G16 | keep_in_rf | 10,641 | ub_stream | 16,846 | 0.63 | 7.04 | 13.06 | keep_in_rf=27,057, ub_stream=16,846 |
| SV5 | R64 C128 G16 | ub_stream ★ | 5,506 | ub_stream | 16,846 | 0.33 | 4.18 | 13.06 | keep_in_rf=27,057, ub_stream=16,846 |
| SV6 | R64 C128 G32 | keep_in_warp | 24,697 | reload | 16,522 | 1.49 | 14.83 | 11.66 | keep_in_warp=65,444, reload=16,522 |
| SV6 | R64 C128 G32 | reload ★ | 20,748 | reload | 16,522 | 1.26 | 12.58 | 11.66 | keep_in_warp=65,444, reload=16,522 |
| SV7 | R64 C128 G64 | multiwarp_ub ★ | 11,182 | reload | 16,592 | 0.67 | 7.32 | 11.1 | multiwarp_ub=19,700, reload=16,592 |
| SV7 | R64 C128 G64 | reload | 24,339 | reload | 16,592 | 1.47 | 14.54 | 11.1 | multiwarp_ub=19,700, reload=16,592 |
| SV7 | R64 C128 G128 | multiwarp_ub | 25,948 | reload | 16,570 | 1.57 | 15.42 | 10.79 | multiwarp_ub=19,837, reload=16,570 |
| SV7 | R64 C128 G128 | reload ★ | 24,423 | reload | 16,570 | 1.47 | 14.58 | 10.79 | multiwarp_ub=19,837, reload=16,570 |
| SV8 | R64 C128 G16 | live ★ | 28,402 | live | 31,843 | 0.89 | 17.04 | 19.56 | live=31,843, spill_dist=33,145 |
| SV8 | R64 C128 G16 | spill_dist | 30,545 | live | 31,843 | 0.96 | 18.23 | 19.56 | live=31,843, spill_dist=33,145 |
| SV9 | E256 K8 | keep ★ | 1,122 | remat_scores | 271 | 4.14 | 1.48 | 1.00 | keep=521, remat_scores=271, remat_idx=521 |
| SV9 | E256 K8 | remat_scores | 2,094 | remat_scores | 271 | 7.73 | 2.02 | 1.00 | keep=521, remat_scores=271, remat_idx=521 |
| SV9 | E256 K8 | remat_idx | 1,122 | remat_scores | 271 | 4.14 | 1.48 | 1.00 | keep=521, remat_scores=271, remat_idx=521 |

## CF — control-flow sensitivity (primary CF1–CF3, CF6)

| case | shape | Simt arm | Simt VF_total | `*d` best arm | `*d` VF_total | Simt/`*d` | Simt µs | `*d` µs | `*d` arms (VF_total) |
|------|-------|----------|--------------:|---------------|--------------:|----------:|--------:|--------:|----------------------|
| CF1 | E256 K1 | keep | 576 | — (no CF1d K1) | — | — | 1.15 | — | — |
| CF1 | E256 K8 | keep | 309 | keep | 105 | 2.94 | 0.97 | 0.9 | keep=105 |
| CF2 | E256 K8 | remat+shared-kill | 2,041 | remat | 113 | 18.06 | 1.95 | 0.9 | remat=113 |
| CF3 | E256 K8 | remat_idx | 851 | remat_idx | 105 | 8.10 | 1.33 | 0.83 | remat_idx=105 |
| CF6 | E256 pnear5 | newton | 961 | newton | 129 | 7.45 | 1.37 | 0.93 | newton=129 |

### CF appendix (CF4, CF5, CF6 pnear25)

| case | shape | Simt arm | Simt VF_total | `*d` best arm | `*d` VF_total | Simt/`*d` | Simt µs | `*d` µs | `*d` arms (VF_total) |
|------|-------|----------|--------------:|---------------|--------------:|----------:|--------:|--------:|----------------------|
| CF6 | E256 pnear25 | newton | 961 | newton | 129 | 7.45 | 1.37 | 0.93 | newton=129 |
| CF4 | E256 pfat5 | nested_if | 758 | nested_if | 70 | 10.83 | 1.23 | 0.89 | nested_if=70 |
| CF4 | E256 pfat25 | nested_if | 781 | nested_if | 70 | 11.16 | 1.25 | 0.89 | nested_if=70 |
| CF5 | E256 pnear5 | div_ulp | 657 | div_ulp | 74 | 8.88 | 1.18 | 0.89 | div_ulp=74 |
| CF5 | E256 pnear25 | div_ulp | 656 | div_ulp | 74 | 8.86 | 1.18 | 0.89 | div_ulp=74 |

## SP — sparse / packing sensitivity

| case | shape | Simt arm | Simt VF_total | `*d` best arm | `*d` VF_total | Simt/`*d` | Simt µs | `*d` µs | `*d` arms (VF_total) |
|------|-------|----------|--------------:|---------------|--------------:|----------:|--------:|--------:|----------------------|
| SP1 | T32 K2 H128 G32 | keep_pos | 19,881 | keep_pos (vscatter) | 31,424 | 0.63 | 12.63 | 19.04 | keep_pos=31424 |
| SP1 | T32 K2 H128 G32 | remat_pos ★ | 17,290 | keep_pos (vscatter) | 31,424 | 0.55 | 11.19 | 19.04 | keep_pos=31424 |
| SP2 | T32 K2 H128 G32 | sf0_w0+keep_acc ★ | 13,738 | sf0_w0+keep_acc | 837 | 16.41 | 9.19 | 2.02 | sf0_w0+keep_acc=837, sf0_w0+remat_acc=837, sf1_w1+keep_acc=2301, sf1_w1+remat_acc=2301 |
| SP2 | T32 K2 H128 G32 | sf1_w1+keep_acc | 23,958 | sf0_w0+keep_acc | 837 | 28.62 | 14.88 | 2.02 | sf0_w0+keep_acc=837, sf0_w0+remat_acc=837, sf1_w1+keep_acc=2301, sf1_w1+remat_acc=2301 |
| SP2 | T32 K2 H128 G32 | sf1_w1+remat_acc | 37,255 | sf0_w0+keep_acc | 837 | 44.51 | 22.27 | 2.02 | sf0_w0+keep_acc=837, sf0_w0+remat_acc=837, sf1_w1+keep_acc=2301, sf1_w1+remat_acc=2301 |
| SP2 | T32 K2 H128 G32 | sf0_w0+remat_acc | 18,944 | sf0_w0+keep_acc | 837 | 22.63 | 12.08 | 2.02 | sf0_w0+keep_acc=837, sf0_w0+remat_acc=837, sf1_w1+keep_acc=2301, sf1_w1+remat_acc=2301 |
| SP2 | T32 K2 H128 G32 | sf1_w1+keep_pos (legacy) | 23,805 | sf0_w0+keep_acc | 837 | 28.44 | 14.79 | 2.02 | sf0_w0+keep_acc=837, sf0_w0+remat_acc=837, sf1_w1+keep_acc=2301, sf1_w1+remat_acc=2301 |
| SP2 | T32 K2 H128 G32 | sf1_w1+remat_pos (legacy) | 21,171 | sf0_w0+keep_acc | 837 | 25.29 | 13.35 | 2.02 | sf0_w0+keep_acc=837, sf0_w0+remat_acc=837, sf1_w1+keep_acc=2301, sf1_w1+remat_acc=2301 |
| SP3 | M32 H128 G32 fp32 | fp32 | 11,158 | fp32 | 349 | 31.97 | 7.45 | 1.44 | fp32=349 |
| SP3 | M32 H128 G32 e8m0 | e8m0 (A5 bit-reinterpret) | 11,171 | e8m0 (VL-aligned continuous) | 311 | 35.92 | 7.45 | 1.39 | e8m0=311 |
| SP4 | E64 H128 pad25 | pad25 | 13,968 | — (no *d twin) | — | — | 9.63 | — | — |
| SP5 | N64 H128 G32 QG32 | sideband ★ | 52,001 | — (no *d twin) | — | — | 30.44 | — | — |
| SP5 | N64 H128 G32 QG32 | interleave | 54,655 | — (no *d twin) | — | — | 31.92 | — | — |
| SP6 | N32 H128 G32 | unpack ★ | 7,813 | unpack+vselr | 393 | 19.88 | 5.35 | 1.22 | gather=534, vselr=393 |
| SP6 | N32 H128 G32 | unpack_sf | 18,727 | unpack_sf+vselr | 707 | 26.49 | 11.42 | 1.40 | gather=826, vselr=707 |

**SP6d (2026-10-08):** SIMD soft e2m1 LUT twin — gather + vselr both PASS; ★ = vselr (393 / 707 VF). Still appendix (not SP1–SP5 peer). vselr needs VL-padded table (`vselr` index lanes == source VL).

**SP alignment fix (2026-10-08, this pass):** SP1d store→`vscatter`; new SP2d gather/scatter twin (4 Acc×sf/w arms); SP3d e8m0→AntiMx VL-aligned continuous decode (VMI has no `vmi.vldu`; `pto.vldas`/`pto.vldus` return non-VMI vregs). Note: SP2d keep_acc≡remat_acc on VF_total (837 / 2301) — DSL Acc remat via aligned VL UB RMW does not cliff; Simt Acc remat cliff remains. SP2d sf0→sf1 tax survives (837→2301).

## Blanks and why

- SV1 E2048 stream: *d: no *d at E2048
- SV2 R16×C64 input_keep: Simt: COMPILE_FAIL (layout_inference: no available layout found)
- SV2 R16×C32 input_keep (ABI fallback): *d: no *d at C32
- CF1 E256 K1 keep: *d: no CF1d K1
- SP2 T32 K2 H128 G32 sf1_w1+keep_pos / remat_pos (legacy): *d: no legacy Pos arms in SP2d (Acc×sf/w only)
- SP4 E64 H128 pad25: *d: no *d twin
- SP5 N64 H128 G32 QG32 sideband / interleave: *d: no *d twin
- CF6b / CF6bd (appendix): not in this pass — Simt CF6b has no VF harvest (wall-only, 10-07) and CF6bd was not re-run.

## `*d` metric per tag

| `*d` tag | st | VF_total | metric | Σ RVEC (ref) | wall µs |
|---|---|---:|---|---:|---:|
| `cf1d_e256_k8_t32_keep` | PASS | 105 | Σ VF (1 row/1 launch) | 689 | 0.9 |
| `cf2d_e256_k8_t32_remat` | PASS | 113 | Σ VF (1 row/1 launch) | 1,255 | 0.9 |
| `cf3d_e256_k8_t32_remat_idx` | PASS | 105 | Σ VF (1 row/1 launch) | 747 | 0.83 |
| `cf4d_e256_t32_pfat25` | PASS | 70 | Σ VF (1 row/1 launch) | 332 | 0.89 |
| `cf4d_e256_t32_pfat5` | PASS | 70 | Σ VF (1 row/1 launch) | 332 | 0.89 |
| `cf5d_e256_t32_pnear25` | PASS | 74 | Σ VF (1 row/1 launch) | 357 | 0.89 |
| `cf5d_e256_t32_pnear5` | PASS | 74 | Σ VF (1 row/1 launch) | 357 | 0.89 |
| `cf6d_e256_t32_pnear25` | PASS | 129 | Σ VF (1 row/1 launch) | 675 | 0.93 |
| `cf6d_e256_t32_pnear5` | PASS | 129 | Σ VF (1 row/1 launch) | 675 | 0.93 |
| `sp1d_t32_k2_h128_g32_t32_keep_pos` | PASS | 31,424 | Σ VF (1 row/1 launch); 2026-10-08 align fix: vscatter Sf + VL-padded pos_row | 778,651 | 19.04 |
| `sp2d_t32_k2_h128_g32_t32_sf0_w0_keep_acc` | PASS | 837 | Σ VF (1 row/1 launch); new gather/scatter twin | 12,496 | 2.02 |
| `sp2d_t32_k2_h128_g32_t32_sf0_w0_remat_acc` | PASS | 837 | Σ VF (1 row/1 launch); =keep_acc on VF | 12,496 | 2.02 |
| `sp2d_t32_k2_h128_g32_t32_sf1_w1_keep_acc` | PASS | 2,301 | Σ VF (1 row/1 launch) | 38,005 | 2.84 |
| `sp2d_t32_k2_h128_g32_t32_sf1_w1_remat_acc` | PASS | 2,301 | Σ VF (1 row/1 launch); =keep_acc on VF | 38,005 | 2.84 |
| `sp3d_m32_h128_g32_t32_e8m0` | PASS | 311 | Σ VF (1 row/1 launch); VL-aligned continuous e8m0 decode (AntiMx); stripped `from __future__ import annotations` | 5,051 | 1.39 |
| `sp3d_m32_h128_g32_t32_fp32` | PASS | 349 | Σ VF (1 row/1 launch); confirmed still PASS after e8m0 fix | 6,468 | 1.44 |
| `sv1d_e256_t32` | PASS | 59 | Σ VF (1 row/1 launch) | 179 | 0.89 |
| `sv2d_r16_c64_t32_input_keep` | PASS | 136 | Σ VF (1 row/1 launch) | 1,300 | 0.92 |
| `sv2d_r32_c128_t32_input_stream` | PASS | 253 | Σ VF (1 row/1 launch) | 3,806 | 1.34 |
| `sv2d_r32_c64_t32_fold_scale_keep` | PASS | 172 | Σ VF (1 row/1 launch) | 2,038 | 1.01 |
| `sv2d_r32_c64_t32_fold_scale_reload` | PASS | 976 | Σ VF (1 row/1 launch) | 4,173 | 1.45 |
| `sv2d_r32_c64_t32_input_stream` | PASS | 172 | Σ VF (1 row/1 launch) | 2,038 | 1.01 |
| `sv3d_m24_vl64_k16_t32_keep` | PASS | 824 | Σ VF (1 row/1 launch) | 9,946 | 3.14 |
| `sv3d_m32_vl64_k16_t32_keep` | PASS | 16,230 | Σ VF (1 row/1 launch) | 23,556 | 12.29 |
| `sv3d_m32_vl64_k16_t32_split_cm16` | PASS | 6,469 | Σ VF (1 row/1 launch) | 13,556 | 6.88 |
| `sv4d_e256_b8_t32_keep_idx` | PASS | 120 | Σ VF (1 row/1 launch) | 1,045 | 1.02 |
| `sv4d_e256_b8_t32_remat_idx` | PASS | 120 | Σ VF (1 row/1 launch) | 1,045 | 1.02 |
| `sv5d_r64_c128_g16_t32_keep_in_rf` | PASS | 27,057 | Σ VF (1 row/1 launch) | 383,419 | 18.72 |
| `sv5d_r64_c128_g16_t32_ub_stream` | PASS | 16,846 | Σ VF (1 row/1 launch) | 259,854 | 13.06 |
| `sv6d_r64_c128_g32_t32_keep_in_warp` | PASS | 65,444 | Σ VF (1 row/1 launch) | 567,687 | 38.83 |
| `sv6d_r64_c128_g32_t32_reload` | PASS | 16,522 | Σ VF (1 row/1 launch) | 248,845 | 11.66 |
| `sv7d_r64_c128_g128_t32_multiwarp_ub` | PASS | 19,837 | Σ VF (1 row/1 launch) | 284,707 | 12.59 |
| `sv7d_r64_c128_g128_t32_reload` | PASS | 16,570 | Σ VF (1 row/1 launch) | 240,497 | 10.79 |
| `sv7d_r64_c128_g64_t32_multiwarp_ub` | PASS | 19,700 | Σ VF (1 row/1 launch) | 287,268 | 12.83 |
| `sv7d_r64_c128_g64_t32_reload` | PASS | 16,592 | Σ VF (1 row/1 launch) | 243,371 | 11.1 |
| `sv8d_r64_c128_g16_t32_live` | PASS | 31,843 | Σ VF (1 row/1 launch) | 314,000 | 19.56 |
| `sv8d_r64_c128_g16_t32_spill_dist` | PASS | 33,145 | Σ VF (1 row/1 launch) | 317,532 | 20.28 |
| `sv9d_e256_k8_t32_keep` | PASS | 521 | Σ VF (1 row/1 launch) | 1,781 | 1.09 |
| `sv9d_e256_k8_t32_remat_idx` | PASS | 521 | Σ VF (1 row/1 launch) | 1,781 | 1.09 |
| `sv9d_e256_k8_t32_remat_scores` | PASS | 271 | Σ VF (1 row/1 launch) | 2,317 | 1.00 |
| `sp6d_n32_h128_g32_t32_unpack_gather` | PASS | 534 | Σ VF (1 row/1 launch); soft e2m1 LUT gather | 6,939 | 1.30 |
| `sp6d_n32_h128_g32_t32_unpack_vselr` | PASS | 393 | Σ VF (1 row/1 launch); ★ best unpack; VL-padded vselr table | 5,188 | 1.22 |
| `sp6d_n32_h128_g32_t32_unpack_sf_gather` | PASS | 826 | Σ VF (1 row/1 launch) | 16,434 | 1.46 |
| `sp6d_n32_h128_g32_t32_unpack_sf_vselr` | PASS | 707 | Σ VF (1 row/1 launch); ★ best unpack_sf | 14,254 | 1.40 |

## Artifacts

- TSV: `reports/VF_CYCLES_SIMT_VS_DSL_BEST_20261008.tsv` · MD: this file
- Box copies: `reports/vfc_20261008/` (vf_total.py harvester, build_report.py, run_d*.sh, jobs*.txt, existing.jsonl = Simt re-harvest, vf_total_d.jsonl = `*d` re-run)
- Remote: `/tmp/vfc_20261008/` (`suite/`, `out_d/`, `out_d2/`, `existing.jsonl`). Note remote `vf_total.py` predates the integer-wall regex fix (sv9d remat_scores wall printed as `1` → 1.00, patched in jsonl).
- SP align re-run remote: `/tmp/sp_align_20261008/` (kernels + `out/opsim_sp*d*` + `vf_total_sp.jsonl`).
- SP6d remote: `/tmp/sp6d_20261008/` (4 arms PASS; `vf_total_sp6d.jsonl`).
