#!/usr/bin/env bash
# SimtVF Parallel CF1–CF6 control-flow micros on pto-b10 (Simt only; no VMI twins).
# Prefer deps-native lib; target=ascend cython NOT pto.
set -euo pipefail
OUT="${ST_SIMTVF_OUT:-/tmp/t_parallel_st_suite}"
LOG=$OUT/oneshot_cf1_cf6.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== t_parallel_st_suite CF1–CF6 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

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

SUMMARY="$OUT/SUMMARY_cf1_cf6_raw.txt"
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
    grep -E 'Immutable|alloc_var|Error|Traceback|COMPILE_OK|RuntimeError|Unresolved|layout|InverseAffine|TypeError' "$OUT/logs/compile_${TAG}.log" | tail -50 | tee -a "$SUMMARY" || true
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

echo "==== PHASE CF1 thresh kill KEEP ===="
compile_run_generic "$SUITE/kernels/cf1_pred_thresh_keep.py" cf1_e256_k1_t32_keep 256 1 32
compile_run_generic "$SUITE/kernels/cf1_pred_thresh_keep.py" cf1_e256_k8_t32_keep 256 8 32

echo "==== PHASE CF2 remat + kill-in-shared ===="
compile_run_generic "$SUITE/kernels/cf2_remat_thresh_kill_shared.py" cf2_e256_k8_t32_remat 256 8 32

echo "==== PHASE CF3 remat_idx ===="
compile_run_generic "$SUITE/kernels/cf3_remat_idx.py" cf3_e256_k8_t32_remat_idx 256 8 32

echo "==== PHASE CF4 nested if ===="
compile_run_generic "$SUITE/kernels/cf4_nested_if.py" cf4_e256_t32_pfat5 256 32 5
compile_run_generic "$SUITE/kernels/cf4_nested_if.py" cf4_e256_t32_pfat25 256 32 25

echo "==== PHASE CF5 div+near0 branch ===="
compile_run_generic "$SUITE/kernels/cf5_div_ulp_branch.py" cf5_e256_t32_pnear5 256 32 5
compile_run_generic "$SUITE/kernels/cf5_div_ulp_branch.py" cf5_e256_t32_pnear25 256 32 25

echo "==== PHASE CF6 Newton+near0 ===="
compile_run_generic "$SUITE/kernels/cf6_newton_branch.py" cf6_e256_t32_pnear5 256 32 5
compile_run_generic "$SUITE/kernels/cf6_newton_branch.py" cf6_e256_t32_pnear25 256 32 25

echo "==== SUMMARY_cf1_cf6_raw ===="
cat "$SUMMARY"
echo DONE_CF1_CF6
