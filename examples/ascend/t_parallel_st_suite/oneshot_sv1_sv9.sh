#!/usr/bin/env bash
# Full SimtVF Parallel matrix SV1–SV9 on pto-b10 (REGRESSION-2 recipe).
# Prefer deps-native lib; target=ascend cython NOT pto. VMI twins are separate.
set -euo pipefail
OUT="${ST_SIMTVF_OUT:-/tmp/t_parallel_st_suite}"
LOG=$OUT/oneshot_sv1_sv9.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== t_parallel_st_suite SV1–SV9 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

# Required paths are environment variables. See README.md.
if [[ -z "${PY:-}" && -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
fi
: "${PY:?Set PY to the NPU venv interpreter (PYTHON_BIN is also accepted)}"
: "${ASCEND_HOME_PATH:?Set ASCEND_HOME_PATH to the CANN toolkit root (directory containing set_env.sh)}"
: "${TILELANG_DEPS:?Set TILELANG_DEPS to the TileLang tree that contains build/lib/libtilelang.so}"
: "${CAMODEL_DEPS:?Set CAMODEL_DEPS to the camodel dependency prefix (its bin/ is prepended to PATH)}"
if [[ -z "${SIM_DSL:-}" && -n "${PTOAS_ROOT:-}" ]]; then
  SIM_DSL="$PTOAS_ROOT/scripts/sim_dsl.sh"
fi
: "${SIM_DSL:?Set SIM_DSL to sim_dsl.sh, or set PTOAS_ROOT to use \$PTOAS_ROOT/scripts/sim_dsl.sh}"
DEPS=$TILELANG_DEPS
ASC=$ASCEND_HOME_PATH
CAMO=$CAMODEL_DEPS
SIM=$SIM_DSL
SOC="${SOC:-${SOC_VERSION:-Ascend950PR_9599}}"
export PY SIM_DSL SOC
BK=/tmp/libtilelang.so.deps_backup_ab
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

HERE=$(cd "$(dirname "$0")" && pwd)
SUITE=${SUITE:-$HERE}
if [[ ! -f "$SUITE/common_asc_harness.py" ]]; then
  SUITE=/tmp/t_parallel_st_suite_suite
fi
export ST_SIMTVF_OUT=$OUT
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

if [[ ! -f /tmp/run_cf_mb_opsim.py ]]; then
  _opsim_src=""
  for c in \
    "${RUN_CF_MB_OPSIM:-}" \
    "$DEPS/examples/ascend/run_cf_mb_opsim.py" \
    "$HERE/run_cf_mb_opsim.py"
  do
    if [[ -n "$c" && -f "$c" ]]; then
      _opsim_src="$c"
      break
    fi
  done
  if [[ -z "$_opsim_src" ]]; then
    echo "error: run_cf_mb_opsim.py not found. Set RUN_CF_MB_OPSIM or place it at \$TILELANG_DEPS/examples/ascend/run_cf_mb_opsim.py" >&2
    exit 1
  fi
  cp -f "$_opsim_src" /tmp/run_cf_mb_opsim.py
fi
cp -f "$SUITE"/run_opsim_*.py /tmp/ 2>/dev/null || true
cp -f "$SUITE"/harvest_report.py /tmp/ 2>/dev/null || true
sed -i 's/\r$//' "$SUITE"/*.py "$SUITE"/kernels/*.py /tmp/run_opsim_*.py 2>/dev/null || true

if [[ ! -f "$BK" ]]; then cp -a "$DEPS/build/lib/libtilelang.so" "$BK"; fi
cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
echo "DEPS_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib"' EXIT

set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1
echo "PY=$PY ASCEND_HOME_PATH=$ASC TILELANG_DEPS=$DEPS CAMODEL_DEPS=$CAMO SIM_DSL=$SIM SOC=$SOC ST_VF_MODE=${ST_VF_MODE:-simt}"

SUMMARY="$OUT/SUMMARY_sv1_sv9_raw.txt"
: > "$SUMMARY"

compile_run_generic() {
  local SCRIPT=$1; shift
  local TAG=$1; shift
  echo "==== COMPILE $TAG ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Immutable|alloc_var|Error|Traceback|COMPILE_OK|RuntimeError|Unresolved|layout|InverseAffine' "$OUT/logs/compile_${TAG}.log" | tail -50 | tee -a "$SUMMARY" || true
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
  ls -la "$OUT"/sources/*${TAG}* 2>/dev/null | tee -a "$SUMMARY" || true
  set +e
  "$SIM" --soc-version "$SOC" --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time|Total cycles|IPC|membar|MTE' "$OUT/opsim_${TAG}.log" | head -12 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

compile_run_topk() {
  local TAG=$1; shift
  echo "==== COMPILE $TAG ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SUITE/kernels/sv9_topk_e2e.py" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Error|Traceback|COMPILE_OK|RuntimeError' "$OUT/logs/compile_${TAG}.log" | tail -40 | tee -a "$SUMMARY" || true
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
  set +e
  "$SIM" --soc-version "$SOC" --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_topk.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time|Total cycles|IPC' "$OUT/opsim_${TAG}.log" | head -8 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

echo "==== PHASE SV1 stream eltwise ===="
compile_run_generic "$SUITE/kernels/sv1_stream_eltwise.py" sv1_e256_t32 256 32
compile_run_generic "$SUITE/kernels/sv1_stream_eltwise.py" sv1_e2048_t32 2048 32

echo "==== PHASE SV2 live RF eltwise + scale bcast (ex-SV1B) ===="
compile_run_generic "$SUITE/kernels/sv2_eltwise_bcast_rf.py" sv2_r32_c32_t32_frag_live 32 32 32 frag_live
compile_run_generic "$SUITE/kernels/sv2_eltwise_bcast_rf.py" sv2_r32_c32_t32_reload 32 32 32 reload

echo "==== PHASE SV3 GEMV Acc KEEP / split (ex-SV2G) ===="
compile_run_generic "$SUITE/kernels/sv3_gemv_partial_keep.py" sv3_m24_vl64_k16_t32_keep 24 64 16 32 keep
compile_run_generic "$SUITE/kernels/sv3_gemv_partial_keep.py" sv3_m32_vl64_k16_t32_keep 32 64 16 32 keep
compile_run_generic "$SUITE/kernels/sv3_gemv_partial_keep.py" sv3_m32_vl64_k16_t32_split_cm16 32 64 16 32 split 16

echo "==== PHASE SV4 gather+psum (keep_idx / remat_idx) ===="
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e256_b8_t32_keep_idx 256 8 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e256_b8_t32_remat_idx 256 8 32 remat_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e256_b16_t32_keep_idx 256 16 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e256_b16_t32_remat_idx 256 16 32 remat_idx

# --- near-spill ladder (2026-10-05): N=E/T elems/thread; keep holds 2N, remat N.
#     crossover at N=32 (E=1024,T=32). remat_idx is COMPILE_FAIL (kept as evidence).
#     third arm remat_idx_calc recomputes the VCI instead of re-reading shared UB.
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e256_b8_t32_remat_idx_calc 256 8 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e512_b8_t32_keep_idx 512 8 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e512_b8_t32_remat_idx_calc 512 8 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1024_b8_t32_keep_idx 1024 8 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1024_b8_t32_remat_idx_calc 1024 8 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1536_b8_t32_keep_idx 1536 8 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1536_b8_t32_remat_idx_calc 1536 8 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e2048_b8_t32_keep_idx 2048 8 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e2048_b8_t32_remat_idx_calc 2048 8 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e256_b16_t32_remat_idx_calc 256 16 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1024_b16_t32_keep_idx 1024 16 32 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1024_b16_t32_remat_idx_calc 1024 16 32 remat_idx_calc
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1024_b8_t16_keep_idx 1024 8 16 keep_idx
compile_run_generic "$SUITE/kernels/sv4_index_gather_psum.py" sv4_e1024_b8_t16_remat_idx_calc 1024 8 16 remat_idx_calc

echo "==== PHASE SV5/SV6/SV7 Case-3 reduce → reduced eltwise ===="
compile_run_generic "$SUITE/kernels/sv5_reduce_small_eltwise.py" sv5_r64_c128_g16_t32_keep_reg 64 128 16 32 keep_reg
compile_run_generic "$SUITE/kernels/sv5_reduce_small_eltwise.py" sv5_r64_c128_g16_t32_ub_reload 64 128 16 32 ub_reload
compile_run_generic "$SUITE/kernels/sv6_reduce_mid_eltwise.py"   sv6_r64_c128_g32_t32 64 128 32 32
compile_run_generic "$SUITE/kernels/sv7_reduce_large_eltwise.py" sv7_r64_c128_g64_t32 64 128 64 32
compile_run_generic "$SUITE/kernels/sv7_reduce_large_eltwise.py" sv7_r64_c128_g128_t32 64 128 128 32

echo "==== PHASE SV8 Case-3 quant e2e ===="
compile_run_generic "$SUITE/kernels/sv8_case3_bcast.py" sv8_r64_c128_g16_t32_live 64 128 16 32 live
compile_run_generic "$SUITE/kernels/sv8_case3_bcast.py" sv8_r64_c128_g16_t32_spill_dist 64 128 16 32 spill_dist

echo "==== PHASE SV9 topk e2e ===="
compile_run_topk sv9_e256_k1_t32_keep 256 1 32 keep
compile_run_topk sv9_e256_k8_t32_keep 256 8 32 keep
compile_run_topk sv9_e256_k8_t32_remat_scores 256 8 32 remat_scores
compile_run_topk sv9_e256_k8_t32_remat_idx 256 8 32 remat_idx
# optional T128 keep
# compile_run_topk sv9_e256_k8_t128_keep 256 8 128 keep

"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv1_sv9.log" || true

echo "==== SUMMARY_sv1_sv9_raw ===="
cat "$SUMMARY"
echo DONE_SV1_SV9
