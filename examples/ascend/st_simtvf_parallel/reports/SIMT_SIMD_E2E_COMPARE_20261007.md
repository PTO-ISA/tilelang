# SimtVF vs Simd/VMI (*d / *v) e2e compare — matched shapes

**Date:** 2026-10-07 HKT  
**SOC:** Ascend950PR_9599  
**Sources:** `/tmp/st_simtvf_parallel`, `/tmp/pr272_sv567_rfladder_20261006`, `/tmp/pr272_sv4_cliff_20261005/out`, `/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d/`  
**No git push / no PR #272.**

## Metric recipes

- **wall_us** = `core0.veccore0` duration_time(us)
- **Simt:** `C_meas` = `VF_SIMT.cycles`; `body_instr` = Σ call_count excluding `VF_*` / set_flag / wait_flag / end / nop / push_pb / dcci; `IPC` = body_instr / C_meas
- **Simd/VMI (*d,*v):** `C_meas` = `VF.cycles` / `VF_SIMD.cycles` if present, else Σ RVEC pipe cycles; `body_instr` = Σ call_count on RVEC pipes only; `IPC` = body_instr / C_meas; EXIPC noted when present. Case-3 `*v` also notes `C_RVEC` (Σ RVEC) because wall_us is scalar-transpose dominated.
- Layer-D `*d` is the primary VMI column (24/24 PASS). Layer-B `*v` included where PASS; Case-3 `*v` walls are **scalar-transpose dominated** (see ST_CASE3_VMI.md).
- Do not invent numbers for COMPILE_FAIL / MISSING.

## Matched-shape main table

| Family | Shape / arm | Simt st | Simt µs | Simt C | Simt instr | Simt IPC | *d st | *d µs | *d C | *d instr | *d IPC | *v st | *v µs | *v C | *v instr | *v IPC | Note |
|--------|-------------|---------|--------:|-------:|-----------:|---------:|-------|------:|-----:|---------:|-------:|-------|------:|-----:|---------:|-------:|------|
| SV1 | E256 T32/T64 | PASS | 1.21 | 727 | 131 | 0.180 | PASS | 0.89 | 59 | 20 | 0.339 | PASS | 0.90 | 91 | 62 | 0.681 | stream baseline |
| SV1 | E2048 T32/T64 | PASS | 2.58 | 2625 | 515 | 0.196 | — | — | — | — | — | PASS | 1.25 | 343 | 426 | 1.242 | stream large; *d no E2048 primary |
| SV2 | R16×C64 input_keep | COMPILE_FAIL | — | — | — | — | PASS | 0.92 | 136 | 151 | 1.110 | — | — | — | — | — | KEEP; Simt may COMPILE_FAIL |
| SV2 | R16×C32 input_keep (fallback) | PASS | 1.16 | 530 | 192 | 0.362 | — | — | — | — | — | — | — | — | — | — | ABI fallback when C64 keep fails |
| SV2 | R32×C64 input_stream | PASS | 3.60 | 4941 | 735 | 0.149 | PASS | 1.01 | 172 | 230 | 1.337 | — | — | — | — | — | STREAM primary |
| SV2 | R32×C128 input_stream | PASS | 8.49 | 13500 | 1579 | 0.117 | PASS | 1.34 | 253 | 422 | 1.668 | — | — | — | — | — | STREAM wide-C |
| SV2 | R32×C64 fold_scale_reload foil | PASS | 4.42 | 6411 | 833 | 0.130 | PASS | 1.45 | 976 | 294 | 0.301 | — | — | — | — | — | optional scale-remat foil |
| SV3 | M24 VL64 K16 keep | PASS | 6.79 | 9105 | 2941 | 0.323 | PASS | 3.14 | 824 | 1694 | 2.056 | PASS | 4.57 | 5181 | 3621 | 0.699 | Acc KEEP |
| SV3 | M32 VL64 K16 keep | PASS | 8.63 | 11874 | 4032 | 0.340 | PASS | 12.29 | 16230 | 2761 | 0.170 | PASS | 5.81 | 6861 | 4800 | 0.700 | Acc KEEP |
| SV3 | M32 VL64 K16 split_cm16 | PASS | 7.79 | 10384 | 3680 | 0.354 | PASS | 6.88 | 6469 | 2366 | 0.366 | PASS | 5.84 | 6950 | 4882 | 0.702 | chunk flush |
| SV4 | E256 B8 keep_idx | PASS | 1.67 | 1372 | 494 | 0.360 | PASS | 1.02 | 120 | 118 | 0.983 | — | — | — | — | — | idxs+acc KEEP; *v COMPILE_FAIL |
| SV4 | E256 B8 remat_idx(_calc) | PASS | 1.62 | 1156 | 472 | 0.408 | PASS | 1.02 | 120 | 118 | 0.983 | — | — | — | — | — | Simt remat_idx_calc ↔ *d remat_idx |
| SV5 | R64 C128 G16 keep_in_rf | PASS | 7.04 | 10641 | 1159 | 0.109 | PASS | 18.72 | 27057 | 56613 | 2.092 | — | — | — | — | — | *v keep COMPILE_FAIL |
| SV5 | R64 C128 G16 ub_stream | PASS | 4.18 | 5506 | 1201 | 0.218 | PASS | 13.06 | 16846 | 40229 | 2.388 | PASS | 109.4 | 958 | 1455 | 1.519 | Case-3; *v wall scalar-transpose dominated; C_RVEC=11817 |
| SV6 | R64 C128 G32 keep_in_warp | PASS | 14.83 | 24697 | 2270 | 0.092 | PASS | 38.83 | 65444 | 61228 | 0.936 | — | — | — | — | — | *v keep COMPILE_FAIL |
| SV6 | R64 C128 G32 reload | PASS | 12.58 | 20748 | 2209 | 0.106 | PASS | 11.66 | 16522 | 38949 | 2.357 | PASS | 101.2 | 1099 | 1483 | 1.349 | Case-3; *v wall scalar-transpose dominated; C_RVEC=11397 |
| SV7 | R64 C128 G64 multiwarp_ub | PASS | 7.32 | 11182 | 1270 | 0.114 | PASS | 12.83 | 19700 | 43430 | 2.205 | PASS | 97.16 | 985 | 1469 | 1.491 | Case-3; *v wall scalar-transpose dominated; C_RVEC=13570 |
| SV7 | R64 C128 G64 reload | PASS | 14.54 | 24339 | 2177 | 0.089 | PASS | 11.10 | 16592 | 38245 | 2.305 | PASS | 97.16 | 1002 | 1417 | 1.414 | Case-3; *v wall scalar-transpose dominated; C_RVEC=13457 |
| SV7 | R64 C128 G128 multiwarp_ub | PASS | 15.42 | 25948 | 2307 | 0.089 | PASS | 12.59 | 19837 | 43173 | 2.176 | — | — | — | — | — |  |
| SV7 | R64 C128 G128 reload | PASS | 14.58 | 24423 | 2161 | 0.088 | PASS | 10.79 | 16570 | 37989 | 2.293 | PASS | 95.43 | 950 | 1418 | 1.493 | Case-3; *v wall scalar-transpose dominated; C_RVEC=12782 |
| SV8 | R64 C128 G16 live | PASS | 17.04 | 28402 | 3986 | 0.140 | PASS | 19.56 | 31843 | 43846 | 1.377 | — | — | — | — | — | *v COMPILE_FAIL historically |
| SV8 | R64 C128 G16 spill_dist | PASS | 18.23 | 30545 | 4210 | 0.138 | PASS | 20.28 | 33145 | 44168 | 1.333 | — | — | — | — | — |  |
| SV9 | E256 K8 keep | PASS | 1.48 | 1122 | 557 | 0.496 | PASS | 1.09 | 521 | 227 | 0.436 | — | — | — | — | — | *v COMPILE_FAIL |
| SV9 | E256 K8 remat_scores | PASS | 2.02 | 2094 | 621 | 0.297 | PASS | 1.00 | 271 | 296 | 1.092 | — | — | — | — | — |  |
| SV9 | E256 K8 remat_idx | PASS | 1.48 | 1122 | 557 | 0.496 | PASS | 1.09 | 521 | 227 | 0.436 | — | — | — | — | — |  |

## Historical SV2 R32×C32 appendix (not matched-shape)

| Tag | st | µs | C | instr | IPC | note |
|-----|----|---:|--:|------:|----:|------|
| `sv2_r32_c32_t32_frag_live` | PASS | 4.93 | 7379 | 743 | 0.101 | HISTORICAL pre-align |
| `sv2_r32_c32_t32_reload` | PASS | 4.24 | 6077 | 585 | 0.096 | HISTORICAL pre-align |

## Coverage notes

- Analyzed tags with any status: 60
- Simt SV2 `r16_c64_input_keep`: expect COMPILE_FAIL (layout); fallback `r16_c32` used for KEEP column.
- *v Case-3 (sv5v–sv7v): PASS arms have walls dominated by scalar transpose outside VF; C_meas uses VF row; C_RVEC (Σ RVEC) annotated for reference. wall_us is scalar-transpose dominated — do not compare walls to Simt/*d.
- *v SV2/SV4/SV8/SV9: historically COMPILE_FAIL — left blank, not invented.

## Artifact paths (remote)

- TSV: `/tmp/st_simtvf_parallel/reports/simt_simd_e2e_20261007.tsv`
- MD: `/tmp/st_simtvf_parallel/reports/SIMT_SIMD_E2E_COMPARE_20261007.md`
