# PTO-DSL Layer D — SV1d–SV9d VMI rooftop / sensitivity suite

**Date:** 2026-10-07 HKT  
**SOC:** Ascend950PR_9599 (opsim login, pto-b10)  
**Work dir (remote):** `/tmp/pr272_sv1_9d_vmi_20261007/`  
**Out (remote):** `/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d/`  
**Box sources:** `/workspace/st_simtvf_parallel/kernels_ptodsl/`  
**Oneshot:** `oneshot_ptodsl_sv1_sv9d.sh`  
**Harvest:** log-truth (`PASS <tag>` + `core0.veccore0`), not SUMMARY.tsv rc

## Intent

Implement **all** SV1–SV9 VMI rooftop twins as **PTO-DSL Layer D** (pure `pto.vmi` ld/st/compute + `pto.mte_*`), matching the expected sensitivity schedules. **Not** TileLang `T.SimdVF`+`T.Parallel` (`*v`). Layer-B `*v` (T.Parallel goal) is **deferred**.

Forbidden: AABBCC, vf_fuse, token tiling. Mapping modes ∈ {(1) ld+st+compute, (2) fused inline, (3) stay-alive}.

DSL I/O convention (same as Phase1/2): **f32** tensors (Simt often uses fp16 A/X). Gold math matches Simt.

## Harvest caveat (SUMMARY.tsv wrong)

Original `SUMMARY.tsv` labeled Case-3 tags `COMPILE_FAIL` whenever `sim_dsl`/pipe `rc≠0`, even when the log ends with `PASS <tag>` and a valid `core0.veccore0` wall. SV1–SV4/SV9 also showed bogus `wall_us=0.002`. Corrected TSV: `SUMMARY_corrected.tsv` (this dir + remote OUT). Spot-check: `sv5d_…_ub_stream` PASS maxabs=2.98e-08 wall **13.06 µs**; `sv8d_…_live` PASS maxabs=1.192e-07 wall **19.56 µs**.

## Tag table (filled)

| ST | shape | arm | schedule | status | µs | maxabs |
|----|-------|-----|----------|--------|---:|--------|
| SV1d | E=256 | stream | ld+add+st per VL | PASS | 0.89 | NA |
| SV2d | R16×C64 | input_keep | KEEP rows | PASS | 0.92 | NA |
| SV2d | R32×C64 | input_stream | KEEP scale + STREAM rows | PASS | 1.01 | NA |
| SV2d | R32×C64 | fold_scale_keep | KEEP scale + STREAM rows | PASS | 1.01 | NA |
| SV2d | R32×C128 | input_stream | KEEP scale + STREAM rows | PASS | 1.34 | NA |
| SV2d | R32×C64 | fold_scale_reload | demoted membar foil | PASS | 1.45 | NA |
| SV3d | M24 VL64 K16 | keep | Acc KEEP across K | PASS | 3.14 | 0.000e+00 |
| SV3d | M32 VL64 K16 | keep | Acc KEEP across K | PASS | 12.29 | 0.000e+00 |
| SV3d | M32 VL64 K16 | split_cm16 | chunk flush | PASS | 6.88 | 0.000e+00 |
| SV4d | E256 B8 | keep_idx | idxs+acc KEEP | PASS | 1.02 | NA |
| SV4d | E256 B8 | remat_idx | remat idxs | PASS | 1.02 | NA |
| SV5d | R64 C128 G16 | keep_in_rf | group in vregs | PASS | 18.72 | 2.980e-08 |
| SV5d | R64 C128 G16 | ub_stream | vload each step | PASS | **13.06** | 2.980e-08 |
| SV6d | R64 C128 G32 | keep_in_warp | VL-pack keep | PASS | 38.83 | 2.980e-08 |
| SV6d | R64 C128 G32 | reload | stream | PASS | **11.66** | 2.980e-08 |
| SV7d | R64 C128 G64 | multiwarp_ub | UB partials | PASS | 12.83 | 2.980e-08 |
| SV7d | R64 C128 G64 | reload | stream | PASS | **11.10** | 2.980e-08 |
| SV7d | R64 C128 G128 | multiwarp_ub | UB partials | PASS | 12.59 | 2.980e-08 |
| SV7d | R64 C128 G128 | reload | stream | PASS | **10.79** | 2.980e-08 |
| SV8d | R64 C128 G16 | live | sf_inv stay-alive | PASS | **19.56** | 1.192e-07 |
| SV8d | R64 C128 G16 | spill_dist | expand to UB | PASS | 20.28 | 1.192e-07 |
| SV9d | E256 K8 | keep | scores+idxs KEEP | PASS | 1.09 | NA |
| SV9d | E256 K8 | remat_scores | reload scores | PASS | **1.00** | NA |
| SV9d | E256 K8 | remat_idx | remat index | PASS | 1.09 | NA |

**Bold µs** = winner within each sensitivity pair/group. NA maxabs = oneshot PASS line omitted `maxabs=` (still numeric PASS).

## Winner per sensitivity

| Sensitivity | Winner | µs | Foil | µs | Δ |
|-------------|--------|---:|------|---:|---|
| SV2d R32×C64 stream vs fold_reload | input_stream / fold_scale_keep (tie) | 1.01 | fold_scale_reload | 1.45 | −30.3% |
| SV2d R32×C128 input_stream | (solo / phase2 known) | 1.34 | — | — | — |
| SV3d M32 keep vs split | split_cm16 | 6.88 | keep | 12.29 | −44.0% |
| SV4d keep_idx vs remat_idx | tie | 1.02 | — | 1.02 | 0% |
| **SV5d** keep_in_rf vs ub_stream | **ub_stream** | **13.06** | keep_in_rf | 18.72 | **−30.2%** |
| **SV6d** keep_in_warp vs reload | **reload** | **11.66** | keep_in_warp | 38.83 | **−70.0%** |
| **SV7d G64** multiwarp_ub vs reload | **reload** | **11.10** | multiwarp_ub | 12.83 | **−13.5%** |
| **SV7d G128** multiwarp_ub vs reload | **reload** | **10.79** | multiwarp_ub | 12.59 | **−14.3%** |
| **SV8d** live vs spill_dist | **live** | **19.56** | spill_dist | 20.28 | **−3.5%** |
| SV9d keep / remat_idx / remat_scores | remat_scores | 1.00 | keep / remat_idx | 1.09 | −8.3% |

### Case-3 winners (SV5d–SV8d)

1. **SV5d G16:** `ub_stream` **13.06 µs** beats `keep_in_rf` 18.72 µs (−30.2%) — streaming vload cheaper than holding group in RF.
2. **SV6d G32:** `reload` **11.66 µs** beats `keep_in_warp` 38.83 µs (−70.0%) — keep-in-warp is catastrophic at G32.
3. **SV7d G64/G128:** `reload` wins both (11.10 / 10.79 µs) over `multiwarp_ub` (12.83 / 12.59 µs).
4. **SV8d G16:** `live` **19.56 µs** beats `spill_dist` 20.28 µs (−3.5%) — stay-alive preferred.

## Remaining blockers

**None** for this SV1d–SV9d tag set: **24/24 PASS** after log-truth harvest. No Case-3 kernel re-fix or re-sim required. CF*d left untouched. No git push / no PR #272.

## Artifacts

- `SUMMARY_corrected.tsv` — this dir + copied to remote OUT
- `/workspace/st_simtvf_parallel/PASS_TABLE_sv1d_sv9d.txt`
- Remote logs: `/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d/sv*.log`
- Harvest helper: `_harvest_ptodsl_walls.sh` (PASS-line + veccore0)

## Notes

- No git push. No PR #272 remote commits.
- Layer-B `*v` deferred (T.Parallel long-term goal).
- Bogus SUMMARY `0.002` walls replaced by veccore0 (SV2d C128 input_stream = 1.34 µs matches phase2).
