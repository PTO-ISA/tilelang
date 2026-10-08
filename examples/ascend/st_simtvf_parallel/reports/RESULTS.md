# SimtVF Parallel ST suite — RESULTS (opsim)

**Date:** 2026-09-23 ~18:30 HKT (renumber to contiguous SV1–SV9)  
**Host:** pto-b10 camodel Ascend950PR_9599  
**Suite:** `/workspace/st_simtvf_parallel`

> **Tag renumber:** `sv1b_*`→`sv2_*`, `sv2g_*`→`sv3_*`, old large-G `sv6_*`→`sv7_*`.  
> New mid-G `sv6_r*_g32*` and SV9 topk e2e tags are **sources ready; µs TBD**.  
> Historical numbers below keep measured µs; **Archived** section lists old tag names.

## Report series

- **Report 1:** [`ST_SV1_AND_SV2.md`](ST_SV1_AND_SV2.md) — stream Parallel eltwise (SV1, measured) + live RF eltwise/bcast quant-style (SV2).
- **SV3:** [`ST_SV3_GEMV.md`](ST_SV3_GEMV.md) — GEMV/thin-GEMM loop-carried fp32 partial-sum RF stress (tags `sv2g_*`). **Opsim GREEN** (M24/M32 keep + M32 split).

- **SV4 gather+psum:** [`ST_SV4_INDEX_GATHER_PSUM.md`](ST_SV4_INDEX_GATHER_PSUM.md) — non-identity Idx gather + `Out[E]` psum (`keep_idx` / `remat_idx`). **keep2 replaced** (historical keep2 µs below; do not invent new numbers). Legacy topk remat archived under `_removed_legacy/`.
- **Case-3 SV5/SV6/SV8:** [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md) — reduce → reduced eltwise (+ SV8 bcast). Tags `sv5_r*`, `sv6_r*` (no arm), `sv8_r*_{live,spill_dist}`. **Opsim GREEN.** SV7 unchanged. Legacy kernels retained.

## SV2 live RF vs reload (32×32) — both PASS

| tag | µs | IPC | notes |
|-----|---:|----:|-------|
| sv2_r32_c32_t32_frag_live | **4.93** | 0.102 | scale live in `scale[1]`; `x[32]` |
| sv2_r32_c32_t32_reload | **4.24** | 0.098 | shared scale reload; **faster** at this size |

Full write-up: [`ST_SV1_AND_SV2.md`](ST_SV1_AND_SV2.md).

## SV3 GEMV partial-sum (NEW SV2) — all PASS

| tag | µs | IPC_proxy | Acc RF (CCE) |
|-----|---:|----------:|--------------|
| sv3_m24_vl64_k16_t32_keep | **6.79** | 0.324 | `acc[48]` |
| sv3_m32_vl64_k16_t32_keep | **8.63** | 0.340 | `acc[64]` |
| sv3_m32_vl64_k16_t32_split_cm16 | **7.79** | 0.355 | `acc_c[32]` |

Split beats M32 keep wall (−0.84 µs) with half Acc RF. HardEvent bars = 2 (entry/exit) for keep and split — no per-K flush tax. Dumps: `reports/sources/sv2g_*_{tir,source,lowered}.txt`. Details: [`ST_SV3_GEMV.md`](ST_SV3_GEMV.md).

## SV2 legacy topk priority matrix (loop-carried KEEP) — all PASS

> **Note:** NEW SV2 sensitivity is **SV3** GEMV partial-sum (`ST_SV3_GEMV.md`). Numbers below are legacy topk CF (`sv2_e*`).

| tag | µs | prior AB | layout (from CCE) |
|-----|----:|---------:|-------------------|
| sv2_e256_k1_t32 | **1.17** | 1.17 | `launch_bounds=32`, `scores[8]`, `idxs[8]` |
| sv2_e256_k8_t32 | **1.48** | 1.48 | same; 1-warp style reduce |
| sv2_e256_k1_t128 | **1.18** | 1.18 | `launch_bounds=128`, `scores[2]` |
| sv2_e256_k8_t128 | **3.03** | 3.03 | `scores[2]`; cross-warp AllReduce tax |

**Sensitivity:** at K=8, T128 is **2.05×** T32 (3.03/1.48). Matches REGRESSION-2 A/B.

## Remat forks vs SV2 @ T=32

| tag | µs | vs SV2 same cell |
|-----|----:|------------------|
| sv2_e256_k8_t32 (KEEP) | 1.48 | baseline |
| sv3_e256_k8_t32 (reload remat) | **2.02** | **+36%** (reload tax) |
| sv4_e256_k8_t32 (legacy index remat) | **1.48** | ≈ KEEP (index remat cheap) |
| sv4_e256_k8_t128 (legacy) | 3.03 | ≈ SV2 T128 |

> **Note:** Active ST-SV4 is now **gather+psum** (`keep_idx`/`remat_idx`, see `ST_SV4_INDEX_GATHER_PSUM.md`); keep2 and topk CF remat rows are historical.

SV3 K1 was 1.15 vs SV2 K1 1.17 (noise). Reload hurts more as K grows.

## Other PASS

| tag | µs | notes |
|-----|----:|-------|
| sv1_e256_t32 | 1.21 | stream; `t1[8]` |
| sv1_e2048_t32 | 2.58 | stream; `t1[64]`; VF IPC_proxy~0.20 |
| sv5_e256_t32_frag | 1.17 | **LEGACY** bcast multi-consumer (fragment) |
| sv5_e256_t32_reload | 1.15 | **LEGACY** bcast multi-consumer (reload arm) |
| sv7_n128_t32 | 1.14 | block reduce 128; `v[4]` |
| sv7_n128_t128 | 1.22 | block reduce 128; `v[1]`-ish |

## Case-3 SV5 / SV6 / SV7 / SV8 — GREEN (2026-09-23 ~17:20 HKT)

Reframe: reduce → `sf_inv[R,CG]` reduced eltwise; SV8 adds bcast mul with `live` vs `spill_dist` (layout-transform spill). Details: [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md), [`ST_SV5_REDUCE_SMALL.md`](ST_SV5_REDUCE_SMALL.md), [`ST_SV7_REDUCE_LARGE.md`](ST_SV7_REDUCE_LARGE.md), [`ST_SV8_CASE3_BCAST.md`](ST_SV8_CASE3_BCAST.md).

| tag | PASS | wall µs | IPC_proxy | RF / notes |
|-----|------|--------:|----------:|------------|
| sv5_r64_c128_g16_t32 | **PASS** | **9.34** | **0.104** | sf_inv ≈ 16 fp32/thr; Y`[64,8]` fp32 |
| sv7_r64_c128_g64_t32 | **PASS** | **14.44** | **0.090** | sf_inv ≈ 4 fp32/thr; Y`[64,2]` |
| sv7_r64_c128_g128_t32 | **PASS** | **14.69** | **0.088** | sf_inv ≈ 2 fp32/thr; CG=1 OK |
| sv8_r64_c128_g16_t32_live | **PASS** | **17.04** | **0.141** | live frag consumer; Out`[64,128]` fp16 |
| sv8_r64_c128_g16_t32_spill_dist | **PASS** | **18.23** | **0.138** | layout expand scale_ub`[R,C]`; +1.19 µs vs live |

Dumps: `/tmp/st_simtvf_parallel/sources/*sv{5,6,8}_r64*_{tir,source,lowered}.txt`. Reports tgz: laptop + `/tmp/st_simtvf_sv5_sv6_sv8_reports.tgz`.

### Legacy SV6 / SV8 (pre Case-3) — GREEN archive

| tag | status | µs | IPC |
|-----|--------|---:|----:|
| sv8_32x32_t32 | **PASS** | **1.65** | **0.261** |
| sv8_32x32_t128 | **PASS** | **1.48** | **0.496** |
| sv6_r64_c128_g16_t32_remat | **PASS** | **21.59** | **0.116** |
| sv6_r64_c128_g16_t32_reload | **PASS** | **19.00** | **0.126** |

Frontend lessons carried into Case-3: `T.alloc_var` + `T.max(v,-v)`; Parallel(R,CG)+serial(G); separate `@T.prim_func` arms.

## Takeaways for fragment-RA discussion

1. Loop-carried KEEP (SV2) reproduces prior green; **thread width dominates K=8**.
2. Reload-as-remat (SV3) costs **~+0.5µs @ K=8 T32** vs KEEP.
3. Legacy index / vci-outer remat (old SV4) ≈ KEEP wall time; **active SV4 is gather+psum** (`keep_idx`/`remat_idx`; sources ready; µs TBD). keep2 replaced.
4. Block-128 (SV7) and stream (SV1) are cheap baselines; legacy SV8/SV6 green archived above; **active Case-3 SV5/6/8 GREEN** (SV5 9.34 µs → SV8 live 17.04 / spill 18.23 µs).

Raw reports on laptop: `C:\Users\Happy\st_simtvf_parallel\` and remote `/tmp/st_simtvf_parallel/`.


---

## Archived tags (pre-renumber)

| Old tag | New tag | µs (historical) |
|---------|---------|-----------------|
| sv1b_r32_c32_t32_frag_live | sv2_r32_c32_t32_frag_live | 4.93 |
| sv1b_r32_c32_t32_reload | sv2_r32_c32_t32_reload | 4.24 |
| sv2g_m24_vl64_k16_t32_keep | sv3_m24_vl64_k16_t32_keep | (see SV3 report) |
| sv2g_m32_vl64_k16_t32_keep | sv3_m32_vl64_k16_t32_keep | (see SV3 report) |
| sv2g_m32_vl64_k16_t32_split_cm16 | sv3_m32_vl64_k16_t32_split_cm16 | (see SV3 report) |
| sv6_r64_c128_g64_t32 | sv7_r64_c128_g64_t32 | 14.44 |
| sv6_r64_c128_g128_t32 | sv7_r64_c128_g128_t32 | 14.69 |
| sv2_e256_k*_t* (topk KEEP micro) | sv9_e256_k*_t*_keep | see legacy |
| sv3_e256_k*_t* (reload remat) | sv9_*_remat_scores | see legacy |
| sv4_e256_k*_t* (index remat topk) | sv9_*_remat_idx | see legacy |
| sv7_n128_t* (1D block reduce) | removed | 1.14 / 1.22 |
| sv8_32x32_t* (old 32×32 reduce) | removed (SV8=quant e2e) | 1.65 / 1.48 |

## Pending (sources ready)

| tag | status |
|-----|--------|
| sv6_r64_c128_g32_t32 | NEW mid-G; µs TBD |
| sv9_e256_k1_t32_keep | e2e; µs TBD |
| sv9_e256_k8_t32_keep | e2e; µs TBD |
| sv9_e256_k8_t32_remat_scores | e2e; µs TBD |
| sv9_e256_k8_t32_remat_idx | e2e; µs TBD |
| sv4_e256_b{8,16}_t32_{keep_idx,remat_idx} | sources ready; µs TBD (keep2 replaced) |
| sv4v_e256_b{8,16}_t64_{keep_idx,remat_idx} | sources ready; µs TBD |
