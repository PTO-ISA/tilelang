# ST PTO-DSL Phase 1 — Three-way compare (Simt A vs TileLang SimdVF B vs ptodsl D)

**Date:** 2026-09-28 ~21:55 HKT  
**SOC:** Ascend950PR_9599 (opsim target)  
**Sources (box):** `/workspace/st_simtvf_parallel/kernels_ptodsl/`  
**Plan:** [`ST_PTODSL_PLAN.md`](ST_PTODSL_PLAN.md)

## Tags tried

| Twin | Tag | Opsim | Notes |
|------|-----|-------|-------|
| SV1d | `sv1d_e256_t32` | **PASS** | opsim `core0.veccore0` **0.89 µs** (Ascend950PR_9599); emit-mlir OK |
| CF1d | `cf1d_e256_k8_t32_keep` | **PASS** | opsim `core0.veccore0` **0.9 µs** |
| SP1d | `sp1d_t32_k2_h128_g32_t32_keep_pos` | **PASS** | opsim `core0.veccore0` **22.84 µs**; emit-mlir ~175 lines after `pto.for_` rewrite |

## Three-way metrics (primary dims)

Existing A/B numbers from suite PASS tables / `ST_VMI_COMPARE.md` / `ST_CF_IMPLEMENT.md` / `PASS_TABLE_sp1_sp6.txt`.  
**D column filled from opsim `core0.veccore0` only — do not invent IPC.**

| Case | A Simt tag | A µs | A status | B SimdVF tag | B µs | B status | D ptodsl tag | D µs | D status |
|------|------------|-----:|----------|--------------|-----:|----------|--------------|-----:|----------|
| SV1 | `sv1_e256_t32` | 1.21 | PASS | `sv1v_e256_t64` | 0.90 | PASS | `sv1d_e256_t32` | **0.89** | **PASS** |
| CF1 | `cf1_e256_k8_t32_keep` | 0.97 | PASS | `cf1v_e256_k8_t64_keep` | 1.01 | PASS | `cf1d_e256_k8_t32_keep` | **0.9** | **PASS** |
| SP1 | `sp1_t32_k2_h128_g32_t32_keep_pos` | 11.3 | PASS | *(no SP1v twin in suite yet)* | — | — | `sp1d_t32_k2_h128_g32_t32_keep_pos` | **22.84** | **PASS** |

IPC / EXIPC for D: **TBD** (blocked).

## Per-case RF / UB / CF instruction map (from DSL source)

### SV1d — stream eltwise (`kernels_ptodsl/sv1d_stream_eltwise.py`)

| Stage | Simt A | DSL D (explicit) |
|-------|--------|------------------|
| GM→UB | `T.copy` A,B | `pto.mte_load` → `a_ub`, `b_ub` |
| RF | fragment `t1,t2,t3` via Parallel | `vload` → vregs `a,b`; `vadd` → `c`; `vstore` → `c_ub` |
| CF | none | none |
| UB→GM | `T.copy` C | `pto.mte_store` |
| Mapping | — | **(1) ld+st+compute** per VL=64 chunk (`E/VL` iterations) |
| Forbidden check | — | no AABBCC / vf_fuse |

### CF1d — thresh kill KEEP (`kernels_ptodsl/cf1d_pred_thresh_keep.py`)

| Stage | Simt A | DSL D (explicit) |
|-------|--------|------------------|
| GM→UB | copy scores, thresh | `mte_load` → `scores_ub`, `thr_ub` |
| RF KEEP | fragment `scores[E]` across `serial(K)` | list of vregs (`E/VL`) live across `for k in K` |
| CF | `if scores[i] > thr: scores[i]=NEG` | `vcmp(..., "gt")` + `vsel(kill, neg_inf, scores)` |
| UB→GM | copy out | `vstore` → `out_ub` + `mte_store` |
| Mapping | — | **(3) stay-alive** scores across K |
| Forbidden check | — | no AABBCC / vf_fuse |

### SP1d — dual scatter keep_pos (`kernels_ptodsl/sp1d_dual_scatter_keep_pos.py`)

| Stage | Simt A | DSL D (explicit) |
|-------|--------|------------------|
| GM→UB | V,Sf,Pos,Expert | `mte_load` four buffers |
| Pad | Parallel `if Expert[p]<0: zero` | `pto.for_(Eexp)` zero-init Vexp (VL) + VL-padded Sf scratch (Expert loaded for IO parity) |
| RF KEEP | `v_frag[H]`, `sf_frag[Hs]` across K | V: 2×VL `vload` stay-alive; Sf: `vgather`→VL (`create_mask(Hs,size=VL)`) stay-alive across K |
| pos keep | shared `pos_row[K]` snapshot | UB `pos_row` via size=1 vload/vstore snapshot |
| Scatter | `vexp[pos,j]=v_frag[j]` if pos≥0 | `pto.for_(Eexp)` + `vcmps(pos,p)` + `vsel` into `vexp_ub` / VL-padded `sfexp_pad`; pack pad→compact Sf via size=1 moves |
| Mapping | — | **(3) stay-alive** V/Sf; shared pos snapshot |
| Arm | keep_pos only | keep_pos only (no remat_pos in Phase 1) |
| Note | — | Avoid Python-unrolled T×K×Eexp (was ~87k MLIR / bisheng spill). Avoid size=Hs contiguous Sf stores (numeric garbage) and Hs-stride `vscatter`/`masked_store` (alignment). |

## Opsim status (honest)

Attempted on **pto-b10** via laptop SSH (`04894df4-…`):

- Working PYTHONPATH: `.camodel_deps_vmi018` (+ optional MLIR) — bare `PTOAS-vmi/ptodsl` lacks `ptoas.mlir`
- **SV1d:** emit-mlir OK + opsim **PASS** — wall `core0.veccore0` **0.89 µs** (log `/tmp/st_simtvf_parallel/ptodsl_phase1/sv1d_opsim.log`)
- **CF1d:** emit-mlir OK + opsim **PASS** — wall **0.9 µs**
- **SP1d:** emit-mlir OK (~175 lines) + opsim **PASS** — wall `core0.veccore0` **22.84 µs** (log `/tmp/st_simtvf_parallel/ptodsl_phase1/sp1d_opsim.log`). Fix: `pto.for_` + VL `vsel` scatter; Sf via VL-pad + size=1 pack (aligned UB).
- IPC/EXIPC for D: **not harvested** (no inventing). Wall µs from `core0.veccore0` only.


**Follow-up finding:** `PYTHONPATH=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018` imports `ptodsl` successfully (bundled wheel). Prefer this over bare `PTOAS-vmi/ptodsl` (missing `ptoas.mlir`). Oneshot prefers `PTODSL_DEPS` accordingly. Compile/opsim still attempted under this path.

**Unblock path (for follow-up):** restore a coherent `ptodsl`+`ptoas`+MLIR wheel pair that pto-vmi `dsl/run_st.sh` historically used (check `PROJECT_PATH` / `whl/` / camodel deps that export `ptoas.mlir`), then:

```bash
bash oneshot_ptodsl_phase1.sh
# or per tag:
python kernels_ptodsl/sv1d_stream_eltwise.py 256 32
python kernels_ptodsl/cf1d_pred_thresh_keep.py 256 8 32
python kernels_ptodsl/sp1d_dual_scatter_keep_pos.py 32 2 128 32 32
# under sim_dsl.sh --soc-version Ascend950PR_9599
```

## Push

Pushed to PR #272 branch `feat/st-simtvf-parallel-sv1-sv9`: commit `e36e76a6` (2026-09-28 ~21:02 HKT).

SP1d fix follow-up: commit `b3abea66` — `st: fix SP1d ptodsl scatter via pto.for_/vscatter` (2026-09-28 ~21:55 HKT).
