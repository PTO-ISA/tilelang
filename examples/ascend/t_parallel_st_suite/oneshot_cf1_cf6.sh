#!/usr/bin/env bash
# SimtVF Parallel CF1–CF6 control-flow micros on pto-b10 (Simt only; no VMI twins).
# Prefer deps-native lib; target=ascend cython NOT pto.
set -euo pipefail
OUT=/tmp/t_parallel_st_suite
LOG=$OUT/oneshot_cf1_cf6.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== t_parallel_st_suite CF1–CF6 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

DEPS=${TILELANG_DEPS:-/home/happybot/projects/tilelang-pto-vmi-deps-stack}
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
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
  for c in \
    /mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/examples/ascend/run_cf_mb_opsim.py \
    /home/happybot/projects/tilelang-deepseek/examples/ascend/run_cf_mb_opsim.py
  do
    [[ -f "$c" ]] && cp -f "$c" /tmp/run_cf_mb_opsim.py && break
  done
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
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
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
