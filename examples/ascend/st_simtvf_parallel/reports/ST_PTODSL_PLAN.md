# ST PTO-DSL Plan (layers A–E) — Phase 1

**Date:** 2026-09-28 HKT (Phase 1+2)  
**Pilot owner:** Lok Chan  
**Box tree:** `/workspace/st_simtvf_parallel/`  
**PR worktree:** `/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel` → GitHub `WenboCodes/tilelang-deepseek` PR #272 (`feat/st-simtvf-parallel-sv1-sv9` → `pto-dev`)

## Five layers

| Layer | Name | What | Status |
|-------|------|------|--------|
| **A** | SimtVF (TileLang) | `T.SimtVF` + `alloc_fragment` + `target=ascend` + deps-native | Landed SV1–SV9, CF1–CF6, SP1–SP6 |
| **B** | SimdVF / VMI (TileLang) | `T.SimdVF` + `target=pto` + overlay; twins `*v` | Partial (SV1V PASS; many CF\*V PASS; SV2V–SV9V historically COMPILE_FAIL) |
| **C** | *(reserved)* | Future TileLang→VMI schedule variants | — |
| **D** | **PTO-DSL explicit** | `@pto.jit(mode=explicit, backend=vpto)` + `pto.mte_*` + UB `castptr` + `pto.vmi.vload/vstore/vcmp/vsel` | **Phase 1 pilot (this doc)** |
| **E** | Raw CCE / ISA | Hand CCE or pto-vmi `cce/` mirrors | Reference only via pto-vmi |

## Phase 1 scope (STRICT fidelity)

Three layer-D twins of existing Simt STs — **same loops/arms**; fragment→VMI mapping ∈ {(1) ld+st+compute, (2) fused compute-inline, (3) stay-alive/forward}.  
**Forbidden:** VMI-only schedules Simt lacks (AABBCC, `vf_fuse` bolted on).

| Twin | Simt source | Tag (primary) | Mapping |
|------|-------------|---------------|---------|
| **SV1d** | `kernels/sv1_stream_eltwise.py` | `sv1d_e256_t32` | (1) stream ld+add+st per VL chunk |
| **CF1d** | `kernels/cf1_pred_thresh_keep.py` | `cf1d_e256_k8_t32_keep` | (3) scores KEEP in vregs; `vcmp`+`vsel` thresh kill |
| **SP1d** | `kernels/sp1_dual_scatter_vsf.py` `keep_pos` | `sp1d_t32_k2_h128_g32_t32_keep_pos` | (3) V/Sf KEEP; `pos_row` UB snapshot; predicated scatter |

## Deliverables (Phase 1)

1. `kernels_ptodsl/{sv1d,cf1d,sp1d}_*.py` + thin ACL harness (`--emit-mlir` / opsim via `sim_dsl.sh`)
2. This plan + `reports/ST_PTODSL_PHASE1_COMPARE.md`
3. `oneshot_ptodsl_phase1.sh` (+ laptop relay)
4. Sync + push to PR #272 when sources/docs coherent (even if opsim blocked)

## Env (pto-b10)

- Exemplars: `~/projects/pto-vmi` (`TopkGateVfKernel`, `MergeModeKernel`, `MaskedGatherKernel`)
- Prefer: `source ~/projects/env.sh` → `PTOAS_ROOT=/home/happybot/PTOAS-vmi`, `.venv-npu`, `cann_91b3`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- Run like pto-vmi: `scripts/sim_dsl.sh --soc-version Ascend950PR_9599` or `dsl/run_st.sh -c …`
- **Do not** mix TileLang Simt deps-native with VMI overlay; DSL path is independent (`ptodsl` + vpto)

## Out of scope (Phase 1)

- Remat arms, CF2–CF6 / SP2–SP6 DSL twins  
- Full EXIPC rooftop harvest (nice-to-have if opsim green)  
- Force-push / dirty `~/projects/tilelang-deepseek` commits


## Phase 2 scope — RF ladder (SV2d / SV4d / SV9d)

Layer-D twins stressing **RF keep vs remat/reload**. Same loops/arms as Simt; mapping ∈ {(1),(2),(3)}; no AABBCC / vf_fuse. Prefer `pto.for_` / stay-alive vregs over Python-unrolled blow-ups (SP1d lesson).

| Twin | Simt source | Arms / primary tags | Mapping |
|------|-------------|---------------------|---------|
| **SV2d** | `kernels/sv2_eltwise_bcast_rf.py` | `sv2d_r32_c128_t32_scale_keep_stream`, `sv2d_r32_c128_t32_full_reload` (+ neg `row_keep_full` FAIL spill; C256 confirm) | **KEEP scales / STREAM rows** (not TILE rows[]); see `ST_RF_KEEP_STREAM_AUDIT.md` |
| **SV4d** | `kernels/sv4_index_gather_psum.py` | `sv4d_e256_b8_t32_keep_idx`, `sv4d_e256_b8_t32_remat_idx` | keep_idx=(3) idxs+acc; remat_idx=(1) on idx + (3) on acc |
| **SV9d** | `kernels/sv9_topk_e2e.py` | `sv9d_e256_k8_t32_keep` (+ remat_scores, remat_idx) | keep=(3); remat_scores=(1+3); remat_idx=(3) scores + remat i |

### Phase 2 deliverables

1. `kernels_ptodsl/{sv2d,sv4d,sv9d}_*.py`
2. `reports/ST_PTODSL_PHASE2_COMPARE.md`
3. `oneshot_ptodsl_phase2.sh` (+ laptop relay)
4. Opsim Ascend950PR_9599 wall µs; push PR #272

### Out of scope (Phase 2)

- CF2–CF6 / SP2–SP6 DSL twins  
- Full EXIPC rooftop (nice-to-have)  
- Dirty `~/projects/tilelang-deepseek` commits
