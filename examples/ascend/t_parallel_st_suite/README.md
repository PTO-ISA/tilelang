# SimtVF Parallel ST suite (SV / CF / SP)

Sensitivity STs comparing **Simt `T.Parallel`** vs **SIMD (PTO-DSL)** on Ascend950PR (opsim).

Primary metric: **total VF cycles** (sum of vector-pipe / VF launch cycles on `core0.veccore0`). Wall µs is secondary. Compare each Simt arm to the **best SIMD (PTO-DSL) schedule** at the same shape — label it “SIMD (PTO-DSL)”, not “*d twin”.

`ST_VF_MODE` selects the TileLang frame and the compile target for that same `T.Parallel` source. The default `simt-asc` is `T.SimtVF` with `target=ascend`. The other three selections are `simt-pto`, `simd-asc`, and `simd-pto`. The SIMD (PTO-DSL) numbers in [`SUMMARY.md`](SUMMARY.md) still come from `kernels_ptodsl/`.

| Deck | Summary |
|------|---------|
| [`docs/st-deck-v2.pptx`](docs/st-deck-v2.pptx) | Slide deck |
| [`SUMMARY.md`](SUMMARY.md) | Sensitivity axes + VF results |

## Case list (21 Simt + 21 SIMD = 42/42)

### SV1–SV9 — vector / RF

| ST | Sensitivity (one line) | Simt | SIMD (PTO-DSL) |
|----|------------------------|------|----------------|
| SV1 | Stream eltwise baseline | `kernels/sv1_stream_eltwise.py` | `kernels_ptodsl/sv1d_stream_eltwise.py` |
| SV2 | Input KEEP / STREAM (+ fold-scale reload foil) | `kernels/sv2_eltwise_bcast_rf.py` | `kernels_ptodsl/sv2d_eltwise_bcast_rf.py` |
| SV3 | GEMV Acc KEEP vs split_cm16 | `kernels/sv3_gemv_partial_keep.py` | `kernels_ptodsl/sv3d_gemv_partial_keep.py` |
| SV4 | Index gather + psum: keep_idx vs remat_idx | `kernels/sv4_index_gather_psum.py` | `kernels_ptodsl/sv4d_index_gather_psum.py` |
| SV5 | Case-3 small-G: keep_in_rf vs ub_stream | `kernels/sv5_reduce_small_eltwise.py` | `kernels_ptodsl/sv5d_reduce_small.py` |
| SV6 | Case-3 mid-G: keep_in_warp vs reload | `kernels/sv6_reduce_mid_eltwise.py` | `kernels_ptodsl/sv6d_reduce_mid.py` |
| SV7 | Case-3 large-G: multiwarp_ub vs reload | `kernels/sv7_reduce_large_eltwise.py` | `kernels_ptodsl/sv7d_reduce_large.py` |
| SV8 | Quant e2e: live vs spill_dist | `kernels/sv8_case3_bcast.py` | `kernels_ptodsl/sv8d_case3_bcast.py` |
| SV9 | TopK e2e: keep / remat_scores / remat_idx | `kernels/sv9_topk_e2e.py` | `kernels_ptodsl/sv9d_topk_e2e.py` |

### CF1–CF6 — control flow

| ST | Sensitivity (one line) | Simt | SIMD (PTO-DSL) |
|----|------------------------|------|----------------|
| CF1 | Parallel if vs scalar thresh; scores KEEP | `kernels/cf1_pred_thresh_keep.py` | `kernels_ptodsl/cf1d_pred_thresh_keep.py` |
| CF2 | Remat scores + shared-kill publish | `kernels/cf2_remat_thresh_kill_shared.py` | `kernels_ptodsl/cf2d_remat_thresh_kill_shared.py` |
| CF3 | Index remat inside CF predicate | `kernels/cf3_remat_idx.py` | `kernels_ptodsl/cf3d_remat_idx.py` |
| CF4 | Nested if (pfat skew) | `kernels/cf4_nested_if.py` | `kernels_ptodsl/cf4d_nested_if.py` |
| CF5 | Div + near-0 ULP branch | `kernels/cf5_div_ulp_branch.py` | `kernels_ptodsl/cf5d_div_ulp_branch.py` |
| CF6 | Newton recip + near-0 branch | `kernels/cf6_newton_branch.py` | `kernels_ptodsl/cf6d_newton_branch.py` |

### SP1–SP6 — sparse / packing

| ST | Sensitivity (one line) | Simt | SIMD (PTO-DSL) |
|----|------------------------|------|----------------|
| SP1 | Pos KEEP vs remat (dual co-scatter) | `kernels/sp1_dual_scatter_vsf.py` | `kernels_ptodsl/sp1d_dual_scatter_keep_pos.py` |
| SP2 | Acc KEEP/remat × sf0_w0/sf1_w1 tax | `kernels/sp2_dual_gather_wreduce.py` | `kernels_ptodsl/sp2d_dual_gather_wreduce.py` |
| SP3 | fp32 SF vs A5 bit-reinterpret e8m0 | `kernels/sp3_sf_pack_ue8m0.py` | `kernels_ptodsl/sp3d_sf_pack_e8m0.py` |
| SP4 | Pad gather early-skip vs mask | `kernels/sp4_pad_gather.py` | `kernels_ptodsl/sp4d_pad_gather.py` |
| SP5 | Sideband vs interleave gather | `kernels/sp5_sideband_vs_interleave.py` | `kernels_ptodsl/sp5d_sideband_vs_interleave.py` |
| SP6 | Soft 4-bit e2m1 LUT unpack (± SF) | `kernels/sp6_fp4_unpack.py` | `kernels_ptodsl/sp6d_fp4_unpack.py` |

## How to run

Export the variables below, source the CANN env script, then run an oneshot from this directory. Oneshots exit with an error when a required variable is unset.

`oneshot_pto_isa_regress.sh` sets `TILELANG_DEPS` to this TileLang checkout (the directory that contains `examples/`) and, when `ST_SIMTVF_OUT` is unset, writes under `/tmp/t_parallel_st_suite_pto_isa`.

| Variable | Required for | Meaning |
|----------|----------------|---------|
| `PY` | every oneshot | NPU virtualenv interpreter. `PYTHON_BIN` is accepted when `PY` is unset. |
| `ASCEND_HOME_PATH` | every oneshot | CANN toolkit root. Oneshots run `source "$ASCEND_HOME_PATH/set_env.sh"`. |
| `TILELANG_DEPS` | Simt SV/CF/SP | Tree whose `build/lib/libtilelang.so` is used. The full regress sets this to the checkout. Also the prefix `run_opsim_topk.py` searches for `build/lib/libtvm_ffi.so`. |
| `CAMODEL_DEPS` | Simt SV/CF/SP | Camodel prefix. `bin/` is prepended to `PATH` and the prefix is added to `PYTHONPATH`. |
| `SIM_DSL` | Simt SV/CF/SP | Opsim launcher (`sim_dsl.sh`). When unset, `$PTOAS_ROOT/scripts/sim_dsl.sh` is used if `PTOAS_ROOT` is set. |
| `PTOAS_ROOT` | PTO-DSL oneshots | PTOAS checkout. `ptodsl/` is added to `PYTHONPATH`. |
| `PTODSL_DEPS` | PTO-DSL oneshots | PTO-DSL / camodel prefix (`bin/` on `PATH`, prefix on `PYTHONPATH`). |
| `MLIR_PYTHON_ROOT` | PTO-DSL oneshots | MLIR Python package root, added to `PYTHONPATH`. |
| `RUN_CF_MB_OPSIM` | Simt opsim | Path to `run_cf_mb_opsim.py`, used when `/tmp/run_cf_mb_opsim.py` is absent and `$TILELANG_DEPS/examples/ascend/run_cf_mb_opsim.py` is absent. |
| `BISHENG_LIBSTDCXX_FIX` | harness compile | `bisheng_libstdcxx_clang_fix.h`, passed to bisheng as `-include`. When unset, the harness uses that header from this directory or from `$TILELANG_DEPS/examples/ascend/msprof_res/orig_pto_vmi_simd/` when the file is there. |
| `ST_SIMTVF_OUT` | optional | Output directory. Simt family default: `/tmp/t_parallel_st_suite`. PTO-DSL family default: a subdirectory of that path. |
| `SOC` | optional | Opsim soc version. `SOC_VERSION` is the same knob. Default `Ascend950PR_9599`. |
| `ST_VF_MODE` | Simt family kernels | One of `simt-asc` (default), `simt-pto`, `simd-asc`, `simd-pto`. See the frame × target table below. |
| `ST_VF_MODES` | `oneshot_vf_mode_regress.sh` | Space- or comma-separated modes. When this and `ST_VF_MODE` are unset, the regress runs all four: `simt-asc simt-pto simd-asc simd-pto`. `ST_VF_MODES` wins when both are set. |
| `ST_VF_REGRESS_OUT` | regress | Root for per-mode OUT dirs (`$root/simt-asc`, `$root/simt-pto`, `$root/simd-asc`, `$root/simd-pto`). Default `/tmp/t_parallel_st_suite_vf_mode`. |
| `ST_VF_ONESHOTS` | regress | Space- or comma-separated oneshot scripts. Default: `oneshot_sv1_sv9.sh`, `oneshot_cf1_cf6.sh`, `oneshot_sp1_sp6.sh`. |
| `ST_VF_SUMMARIZE_ONLY` | regress | `1` reprints the PASS/FAIL table and skips oneshots. |

`harvest_pass_table.sh` uses `PY` and `TILELANG_DEPS`. PTO-DSL oneshots source `$HOME/projects/env.sh` when that file exists, then source `$ASCEND_HOME_PATH/set_env.sh`.

```bash
export PY=/path/to/.venv-npu/bin/python
export ASCEND_HOME_PATH=/path/to/cann
source "$ASCEND_HOME_PATH/set_env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Simt SV/CF/SP, including the full regress
export CAMODEL_DEPS=/path/to/camodel-deps
export PTOAS_ROOT=/path/to/PTOAS
export SIM_DSL="$PTOAS_ROOT/scripts/sim_dsl.sh"
export TILELANG_DEPS="$(cd ../../.. && pwd)"   # in-tree libtilelang.so; the full regress sets this itself
export BISHENG_LIBSTDCXX_FIX=/path/to/bisheng_libstdcxx_clang_fix.h
# export RUN_CF_MB_OPSIM=/path/to/run_cf_mb_opsim.py

# SIMD (PTO-DSL) oneshots
export PTODSL_DEPS=/path/to/ptodsl-deps
export MLIR_PYTHON_ROOT=/path/to/mlir/python_packages/mlir_core

export ST_SIMTVF_OUT=/tmp/t_parallel_st_suite
export SOC=Ascend950PR_9599
export ST_VF_MODE=simt-asc   # or simt-pto, simd-asc, simd-pto; unset defaults to simt-asc

cd examples/ascend/t_parallel_st_suite

# Full primary regress (Simt + SIMD)
bash oneshot_pto_isa_regress.sh

# Or family-by-family:
bash oneshot_sv1_sv9.sh              # Simt SV1–SV9
bash oneshot_cf1_cf6.sh              # Simt CF1–CF6
bash oneshot_sp1_sp6.sh              # Simt SP1–SP6
bash oneshot_ptodsl_sv1_sv9d.sh      # SIMD SV1d–SV9d
bash oneshot_ptodsl_cf1d_cf6d.sh     # SIMD CF1d–CF6d
bash oneshot_ptodsl_sp1d_sp6d.sh     # SIMD SP1d–SP6d (includes SP4d/SP5d)
```

Fill the placeholder exports above (`/path/to/cann`, `/path/to/.venv-npu/bin/python`, and the dependency prefixes). Machine-specific install paths stay in the shell environment.

### TileLang VF frame and compile target (`ST_VF_MODE`)

Kernels under `kernels/` call `vf_region(threads)` and compile with `compile_target()` (`vf_mode.py`). One `T.Parallel` body runs under one of four frame × target pairs:

| `ST_VF_MODE` | Aliases | Frame | `tilelang.compile` target |
|--------------|---------|-------|---------------------------|
| `simt-asc` (default) | `simt`, `simtvf`, `simt-ascend` | `T.SimtVF(threads=threads)` | `ascend` |
| `simt-pto` | `simtvf-pto` | `T.SimtVF(threads=threads)` | `pto` |
| `simd-asc` | `simd`, `simdvf`, `simd-ascend` | `T.SimdVF()` | `ascend` |
| `simd-pto` | `simdvf-pto` | `T.SimdVF()` | `pto` |

The harness prints one line per kernel process, for example `ST_VF_MODE=simt-asc frame=SimtVF target=ascend`. `kernels_ptodsl/` stays the hand-written PTO-DSL sensitivity arm; `ST_VF_MODE` does not select those files.

PTO modes compile with `target=pto`. They are not remapped to `ascend`. If the PTO backend is missing or lowering fails, that case is `FAIL compile` in the regress table. SimdVF and PTO modes may fail to compile or fail opsim on some ST cases with the current compiler. The table lists PASS and FAIL per case and mode.

```bash
# One mode on the existing Simt-family oneshots
export ST_VF_MODE=simd-pto
bash oneshot_sv1_sv9.sh

# All four modes. Writes $ST_VF_REGRESS_OUT/<mode>/
# (default /tmp/t_parallel_st_suite_vf_mode) and prints a tag × mode table.
bash oneshot_vf_mode_regress.sh

# One mode, or a family subset
ST_VF_MODES=simt-asc bash oneshot_vf_mode_regress.sh
ST_VF_ONESHOTS=oneshot_cf1_cf6.sh bash oneshot_vf_mode_regress.sh

# Reprint the table from existing OUT dirs
ST_VF_SUMMARIZE_ONLY=1 bash oneshot_vf_mode_regress.sh
```

Cells are `PASS`, `FAIL compile`, `FAIL opsim`, `FAIL no-pass-line`, or `MISSING`. The table is also written to `$ST_VF_REGRESS_OUT/VF_MODE_PASS_TABLE.txt`.

Harness / opsim helpers: `common_asc_harness.py`, `vf_mode.py`, `common_pto_harness.py`, `run_opsim_generic.py`, `run_opsim_topk.py` (SV9).
Harvest: `harvest_report.py`, `harvest_pass_table.sh`, `harvest_simt_vmi_compare.py`.

### Known blocker on this tip

PTO-ISA `pto-dev` tip may emit CANN MicroAPI DMA symbols (`asc_copy_gm2ub_align`, L2 cache mode, pack helpers) missing from `cann_91b3` / `cann_92b1` on pto-b10. Live opsim can fail until a matching toolkit is available. VF numbers in [`SUMMARY.md`](SUMMARY.md) are from Wenbo tip `3828bbcf` / prior opsim (2026-10-07–08), not re-validated on this tip.
