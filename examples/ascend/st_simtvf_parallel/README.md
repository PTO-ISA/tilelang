# SimtVF Parallel ST suite (SV1–SV9 contiguous)

Sensitivity STs for SimtVF + `T.Parallel` + fragment RF.  
**Canonical design (purpose + GPU/NPU):** [`reports/ST_LIST_DESIGN.md`](reports/ST_LIST_DESIGN.md).  
**Case-3 quant umbrella:** [`reports/ST_CASE3_DESIGN.md`](reports/ST_CASE3_DESIGN.md).

## Contract
- `target=ascend` + `execution_backend=cython` (**not** pto) for SimtVF
- deps-native `libtilelang.so` (not overlay)
- VMI twins: `target=pto` + overlay — never mix libs
- env: `.venv-npu` + `cann_91b3` + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`

## Active kernels (SV1–SV9)
| ST | Kernel | Notes |
|----|--------|-------|
| SV1 | `kernels/sv1_stream_eltwise.py` | stream eltwise |
| SV2 | `kernels/sv2_eltwise_bcast_rf.py` | live RF / reload (ex-SV1B) |
| SV3 | `kernels/sv3_gemv_partial_keep.py` | GEMV Acc KEEP/split (ex-SV2G) |
| SV4 | `kernels/sv4_index_gather_psum.py` | gather + psum (`keep_idx` / `remat_idx`); VMI twin `sv4v_*` |
| SV5 | `kernels/sv5_reduce_small_eltwise.py` | Case-3 small-G |
| SV6 | `kernels/sv6_reduce_mid_eltwise.py` | Case-3 mid-G (**NEW**) |
| SV7 | `kernels/sv7_reduce_large_eltwise.py` | Case-3 large-G (ex-SV6) |
| SV8 | `kernels/sv8_case3_bcast.py` | quant e2e live / spill_dist |
| SV9 | `kernels/sv9_topk_e2e.py` | topk e2e keep / remat_scores / remat_idx |
| VMI | `sv5v` / `sv6v` / `sv7v` / `sv8v` | Case-3 twins |

## Removed
Legacy micros in `kernels/_removed_legacy/` (old topk SV2/3/4, old SV5/6, old SV7 1D, old SV8 32×32).  
Two finals: **SV8 quant e2e** + **SV9 topk e2e**.

## Reports
- [`reports/ST_LIST_DESIGN.md`](reports/ST_LIST_DESIGN.md) — SV1–SV9 purpose + GPU/NPU
- [`reports/ST_SV1_AND_SV2.md`](reports/ST_SV1_AND_SV2.md)
- [`reports/ST_SV3_GEMV.md`](reports/ST_SV3_GEMV.md)
- [`reports/ST_SV4_INDEX_GATHER_PSUM.md`](reports/ST_SV4_INDEX_GATHER_PSUM.md) (keep2 redirect: `ST_SV4_INDEX_KEEP2.md`)
- [`reports/ST_CASE3_DESIGN.md`](reports/ST_CASE3_DESIGN.md) + SV5/SV6/SV7/SV8 + VMI
- [`reports/ST_SV9_TOPK_E2E.md`](reports/ST_SV9_TOPK_E2E.md)
- [`reports/RESULTS.md`](reports/RESULTS.md)

## Run (all Simt SV1–SV9)
```bash
bash oneshot_sv1_sv9.sh
# laptop relay:
powershell -File .\\run_via_laptop_sv1_sv9.ps1
```

Per-family: `oneshot_sv2.sh`, `oneshot_sv3.sh`, `oneshot_sv4.sh`, `oneshot_sv5_sv6_sv7_sv8.sh`, `oneshot_sv9.sh`.


## Phase 1 — PTO-DSL twins (layer D)

Explicit `@pto.jit(mode=explicit, backend=vpto)` twins of Simt SV1 / CF1 / SP1(`keep_pos`):

| Twin | Kernel | Primary tag |
|------|--------|-------------|
| SV1d | `kernels_ptodsl/sv1d_stream_eltwise.py` | `sv1d_e256_t32` |
| CF1d | `kernels_ptodsl/cf1d_pred_thresh_keep.py` | `cf1d_e256_k8_t32_keep` |
| SP1d | `kernels_ptodsl/sp1d_dual_scatter_keep_pos.py` | `sp1d_t32_k2_h128_g32_t32_keep_pos` |

Plan / compare: [`reports/ST_PTODSL_PLAN.md`](reports/ST_PTODSL_PLAN.md), [`reports/ST_PTODSL_PHASE1_COMPARE.md`](reports/ST_PTODSL_PHASE1_COMPARE.md).

```bash
bash oneshot_ptodsl_phase1.sh
# laptop relay:
powershell -File .\run_via_laptop_ptodsl_phase1.ps1
```

Requires working `ptodsl` + `ptoas.mlir` (see compare MD blocker notes). Prefer same env as `~/projects/pto-vmi` DSL cases (`PTOAS-vmi` + `.venv-npu` + `cann_91b3`).


## SP1–SP6 (single-axis redesign 2026-10-07)

Contract: [`reports/ST_SP1_TO_SP6_ACCESS.md`](reports/ST_SP1_TO_SP6_ACCESS.md) · PASS: [`PASS_TABLE_sp1_sp6.txt`](PASS_TABLE_sp1_sp6.txt)

| ST | Sensitivity | Kernel |
|----|-------------|--------|
| SP1 | Pos KEEP vs remat (dual co-scatter) | `kernels/sp1_dual_scatter_vsf.py` |
| SP2 | Acc KEEP/remat **+** sf0_w0/sf1_w1 tax (one case) | `kernels/sp2_dual_gather_wreduce.py` |
| SP3 | fp32 vs **A5 bit-reinterpret e8m0** (soft LUT retired; no native e8m0 vcvt on A5) | `kernels/sp3_sf_pack_ue8m0.py` |
| SP3d | VMI bit-shift compose twin (full source; UNRUN) | `kernels_ptodsl/sp3d_sf_pack_e8m0.py` |
| SP4 | pad early-skip vs mask | `kernels/sp4_pad_gather.py` |
| SP5 | sideband vs interleave | `kernels/sp5_sideband_vs_interleave.py` |
| SP6 | soft LUT 4-bit e2m1 dequant (appendix) | `kernels/sp6_fp4_unpack.py` |

```bash
bash oneshot_sp1_sp6.sh
# laptop relay:
powershell -File .\run_via_laptop_sp1_sp6.ps1
```

SP3/SP3d `e8m0` = **A5 bit-reinterpret UNRUN** (`vload ui8→vcvt ui32→vshls(23)→vinterpret_cast f32→brc+vmul`). No soft Pow2 LUT. No native e8m0 vcvt on A5 (A6 may). Cube MX out of scope.

