#!/usr/bin/env bash
# Retry SP1 keep_pos, SP2 both, SP3 ue8m0 after layout/gold fixes.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
LOG=$OUT/oneshot_sp_retry.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources"
exec > >(tee -a "$LOG") 2>&1
echo "=== SP retry $(date -Is) ==="

DEPS=/home/happybot/projects/tilelang-pto-vmi-deps-stack
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
BK=/tmp/libtilelang.so.deps_backup_ab
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

HERE=$(cd "$(dirname "$0")" && pwd)
SUITE=${SUITE:-$HERE}
export ST_SIMTVF_OUT=$OUT
cp -f "$SUITE"/run_opsim_*.py /tmp/ 2>/dev/null || true
sed -i 's/\r$//' "$SUITE"/*.py "$SUITE"/kernels/*.py /tmp/run_opsim_*.py 2>/dev/null || true
if [[ ! -f "$BK" ]]; then cp -a "$DEPS/build/lib/libtilelang.so" "$BK"; fi
cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib"' EXIT
set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TILELANG_DISABLE_DATA_RACE_CHECK=1
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

SUMMARY="$OUT/SUMMARY_sp_retry_raw.txt"
: > "$SUMMARY"
compile_run_generic() {
  local SCRIPT=$1; shift; local TAG=$1; shift
  echo "==== COMPILE $TAG ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Immutable|alloc_var|Error|Traceback|COMPILE_OK|RuntimeError|Unresolved|layout|InverseAffine|TypeError' "$OUT/logs/compile_${TAG}.log" | tail -40 | tee -a "$SUMMARY" || true
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
  set +e
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time' "$OUT/opsim_${TAG}.log" | head -4 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

compile_run_generic "$SUITE/kernels/sp1_dual_scatter_vsf.py" sp1_t32_k2_h128_g32_t32_keep_pos 32 2 128 32 32 keep_pos
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf1_w1_keep_pos 32 2 128 32 32 sf1 w1 keep_pos
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf1_w1_remat_pos 32 2 128 32 32 sf1 w1 remat_pos
compile_run_generic "$SUITE/kernels/sp3_sf_pack_ue8m0.py" sp3_m32_h128_g32_t32_ue8m0 32 128 32 32 ue8m0
echo "==== RETRY SUMMARY ===="; cat "$SUMMARY"; echo DONE_RETRY
