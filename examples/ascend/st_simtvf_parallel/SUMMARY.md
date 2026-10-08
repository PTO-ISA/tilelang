# SUMMARY — SimtVF Parallel ST renumber to SV1–SV9

**Date:** 2026-09-23 ~18:40 HKT  
**Suite:** `/workspace/st_simtvf_parallel`  
**Deliverable tarball:** `/workspace/st_simtvf_parallel_sv1_sv9_src.tgz`

## File renames

| Old | New |
|-----|-----|
| `kernels/sv1b_eltwise_bcast_rf.py` | `kernels/sv2_eltwise_bcast_rf.py` (`sv1b_*`→`sv2_*`) |
| `kernels/sv2_gemv_partial_keep.py` | `kernels/sv3_gemv_partial_keep.py` (`sv2g_*`→`sv3_*`) |
| `kernels/sv6_reduce_large_eltwise.py` | `kernels/sv7_reduce_large_eltwise.py` (`sv6_g64/128`→`sv7_*`) |
| `kernels/sv6v_reduce_large_eltwise.py` | `kernels/sv7v_reduce_large_eltwise.py` |
| *(new)* | `kernels/sv6_reduce_mid_eltwise.py` + `sv6v_*` (G=32) |
| *(new)* | `kernels/sv9_topk_e2e.py` (keep / remat_scores / remat_idx) |
| `reports/ST_SV1_AND_SV1B.md` | `ST_SV1_AND_SV2.md` |
| `reports/ST_SV2_GEMV.md` | `ST_SV3_GEMV.md` |
| `reports/ST_SV6_REDUCE_LARGE.md` | `ST_SV7_REDUCE_LARGE.md` |
| *(new)* | `ST_SV6_REDUCE_MID.md`, `ST_SV9_TOPK_E2E.md` |
| `reports/ST_LIST_DESIGN.md` | rewritten locked SV1–SV9 table |

Kept: SV1 stream, SV4 gather_psum (keep2 replaced), SV5 small, SV8 quant e2e, SV5V. Legacy micros stay in `kernels/_removed_legacy/` (not wired).

## Oneshot (all Simt SV1–SV9)
```bash
bash oneshot_sv1_sv9.sh
# laptop relay:
powershell -File .\run_via_laptop_sv1_sv9.ps1
```

Per-family: `oneshot_sv2.sh`, `oneshot_sv3.sh`, `oneshot_sv4.sh`, `oneshot_sv5_sv6_sv7_sv8.sh`, `oneshot_sv9.sh`.

## SV9
Extracted tournament topk from `_removed_legacy/sv2_topk_keep.py` (+ remat forks) into
`sv9_topk_e2e.py`. Gold = stable min-index-on-ties top-K. Arms: KEEP (NEG in RF),
`remat_scores` (kill in shared), `remat_idx` (no idxs fragment).

## Success check
- Contiguous SV1–SV9 in `reports/ST_LIST_DESIGN.md`
- Tarball sources+md only at `/workspace/st_simtvf_parallel_sv1_sv9_src.tgz`

## CF1–CF6 + CF1V–CF6V (control-flow slices)

**Product cut (2026-10-07):** primary = **CF1–CF3 + CF6** only. CF4 / CF5 / CF6b = appendix / historical (files kept). Compare: `reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md`.

**Date:** 2026-09-28 ~10:45 HKT · **Oneshots:** `oneshot_cf1_cf6.sh` / `oneshot_cf1v_cf6v.sh`  
**Detail:** [`reports/ST_CF_IMPLEMENT.md`](reports/ST_CF_IMPLEMENT.md) · [`reports/ST_CF1_CF6_SLICES.md`](reports/ST_CF1_CF6_SLICES.md)

| case | tag | status | e2e µs |
|------|-----|--------|-------:|
| CF1 | `cf1_e256_k1_t32_keep` | PASS | 1.15 |
| CF1 | `cf1_e256_k8_t32_keep` | PASS | 0.97 |
| CF2 | `cf2_e256_k8_t32_remat` | PASS | 1.95 |
| CF3 | `cf3_e256_k8_t32_remat_idx` | PASS | 1.33 |
| CF4 | `cf4_e256_t32_pfat5` / `_pfat25` | PASS | 1.23 / 1.25 |
| CF5 | `cf5_e256_t32_pnear5` / `_pnear25` | PASS | 1.18 / 1.18 |
| CF6 | `cf6_e256_t32_pnear5` / `_pnear25` | PASS | 1.37 / 1.37 |
| CF1V | `cf1v_e256_k1_t64_keep` / `_k8_` | PASS | 1.17 / 1.01 |
| CF2V | `cf2v_e256_k8_t64_remat` | PASS | 1.04 |
| CF3V | `cf3v_e256_k8_t64_remat_idx` | PASS | 1.02 |
| CF4V | `cf4v_e256_t64_pfat5` / `_pfat25` | PASS | 0.93 / 0.93 |
| CF5V | `cf5v_e256_t64_pnear5` / `_pnear25` | PASS | 6.23 / 6.23 |
| CF6V | `cf6v_e256_t64_pnear5` / `_pnear25` | PASS | 12.35 / 12.35 |

VMI uses `T.Select` mask (not Simt divergent if). CF5V/CF6V Div/Newton outside SimdVF (VerifyParallelToPTO + AIV ABI). Deps lib restored after VMI oneshot.

## SP1–SP6 single-axis redesign (2026-10-07) + opsim

**OpSim:** 2026-10-07 ~18:40 HKT · Ascend950PR_9599 · pto-b10 · `oneshot_sp1_sp6.sh`  
**Doc:** [`reports/ST_SP1_TO_SP6_ACCESS.md`](reports/ST_SP1_TO_SP6_ACCESS.md) · PASS: [`PASS_TABLE_sp1_sp6.txt`](PASS_TABLE_sp1_sp6.txt) · raw: [`SUMMARY_sp1_sp6_raw.txt`](SUMMARY_sp1_sp6_raw.txt)
**SP6d (2026-10-08):** SIMD (PTO-DSL) soft e2m1 LUT via **gather** + **vselr** — [`PASS_TABLE_sp6d.txt`](reports/PASS_TABLE_sp6d.txt); still appendix, not SP1–SP5 peer.

| case | sensitivity / arm | status | e2e µs |
|------|-------------------|--------|------:|
| SP1 | Pos keep / remat | PASS | 12.63 / 11.19 |
| SP2 | Acc KEEP×sf0/sf1 + Acc remat×sf0/sf1 (+ legacy Pos) | PASS | 9.19 / 14.88 / 22.27 / 12.08 (+ 14.79 / 13.35) |
| SP3 | fp32 vs A5 bit-reinterpret e8m0 | PASS | 7.45 / 7.45 |
| SP4 | pad25 | PASS | 9.63 |
| SP5 | sideband / interleave | PASS | 30.44 / 31.92 |
| SP6 | soft LUT unpack / unpack_sf | PASS | 5.35 / 11.42 |
| SP6d | SIMD gather/vselr unpack (±sf) | PASS | ★vselr 1.22 / 1.40 µs (VF 393 / 707); gather 1.30 / 1.46 |

**Notes:** SP2x folded into SP2. Soft Pow2 LUT retired from SP3. All 15 oneshot tags PASS. No push / no #272.


## Case-3 RF-ladder (SV5–SV7 + VMI twins)

**Date:** 2026-10-06 ~17:48 HKT · **Host:** pto-b10 · **OUT:** `/tmp/pr272_sv567_rfladder_20261006`
**Reports:** [`reports/case3_rfladder_20261006/RESULT.md`](reports/case3_rfladder_20261006/RESULT.md) ·
[`VF_THEORY.md`](reports/case3_rfladder_20261006/VF_THEORY.md) ·
PASS tables [`PASS_TABLE_sv1_sv9.txt`](PASS_TABLE_sv1_sv9.txt) /
[`PASS_TABLE_sv1v_sv9v.txt`](PASS_TABLE_sv1v_sv9v.txt). **No push.**

- **Simt:** all 8 ladder arms PASS (sv5 keep/ub_stream, sv6 keep/reload, sv7 g64/g128 multiwarp/reload).
- **VMI:** 6/8 PASS — all reload/stream/multiwarp_ub arms; keep_in_rf + keep_in_warp still
  COMPILE_FAIL (fragment xf layout). Wall ~95–109 µs dominated by scalar transpose;
  theory `C_meas` uses RVEC pipe sum.
