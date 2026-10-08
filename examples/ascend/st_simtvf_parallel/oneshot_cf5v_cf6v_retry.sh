#!/usr/bin/env bash
# Retry CF5V/CF6V only (overlay). Restores deps lib on EXIT.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
SUITE=${SUITE:-/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel}
SUMMARY=$OUT/SUMMARY_cf5v_cf6v_retry.txt
: > "$SUMMARY"
exec > >(tee -a "$OUT/oneshot_cf5v_cf6v_retry.log") 2>&1
DEPS=/home/happybot/projects/tilelang-pto-vmi-deps-stack
OV=/mnt/fluxdata/happybot/projects/tilelang-pr258-overlay
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
BK=/tmp/libtilelang.so.deps_backup_ab
cp -f "$SUITE"/run_opsim_generic.py /tmp/
cp -a "$OV/build/lib/libtilelang.so" "$DEPS/build/lib/libtilelang.so"
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib from $BK"' EXIT
set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC ST_SIMTVF_OUT=$OUT TILELANG_DISABLE_CACHE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TILELANG_DISABLE_DATA_RACE_CHECK=1
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
run1() {
  local SCRIPT=$1 TAG=$2; shift 2
  echo "==== COMPILE $TAG ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'VerifyParallel|unsupported|Error|COMPILE_OK' "$OUT/logs/compile_${TAG}.log" | tail -20 | tee -a "$SUMMARY" || true
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
  set +e
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  echo "$TAG $(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1) us=$(grep core0.veccore0 "$OUT/opsim_${TAG}.log" | head -2 | tr '\n' ' ')" | tee -a "$SUMMARY"
}
run1 "$SUITE/kernels/cf5v_div_ulp_branch.py" cf5v_e256_t64_pnear5 256 64 5
run1 "$SUITE/kernels/cf5v_div_ulp_branch.py" cf5v_e256_t64_pnear25 256 64 25
run1 "$SUITE/kernels/cf6v_newton_branch.py" cf6v_e256_t64_pnear5 256 64 5
run1 "$SUITE/kernels/cf6v_newton_branch.py" cf6v_e256_t64_pnear25 256 64 25
cat "$SUMMARY"
echo DONE_CF5V_CF6V_RETRY
