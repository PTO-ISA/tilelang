#!/usr/bin/env bash
# VMI / SimdVF CF1V–CF6V twins (target=pto + OVERLAY lib) on pto-b10.
# Mask/select control flow; lanes=64 primary (avoid known t32 ABI blockers).
# Do NOT run while any Simt oneshot holds deps-native libtilelang.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
LOG=$OUT/oneshot_cf1v_cf6v.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== st_simtvf_parallel CF1V–CF6V (VMI) oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

DEPS=/home/happybot/projects/tilelang-pto-vmi-deps-stack
OV=/mnt/fluxdata/happybot/projects/tilelang-pr258-overlay
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
BK=/tmp/libtilelang.so.deps_backup_ab
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

HERE=$(cd "$(dirname "$0")" && pwd)
SUITE=${SUITE:-$HERE}
if [[ ! -f "$SUITE/common_pto_harness.py" ]]; then
  SUITE=/tmp/st_simtvf_parallel_suite
fi
export ST_SIMTVF_OUT=$OUT
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

SUMMARY="$OUT/SUMMARY_cf1v_cf6v_raw.txt"
: > "$SUMMARY"

# Refuse to fight Simt oneshot over libtilelang
BUSY=$(pgrep -af 'oneshot_sv[0-9]|oneshot_cf[0-9]_|sv[0-9]_|cf[0-9]_' || true)
BUSY=$(echo "$BUSY" | grep -vE 'sv[0-9]v|cf[0-9]v|oneshot_sv[0-9]v|oneshot_cf[0-9]v|pgrep' || true)
if [[ -n "${BUSY}" ]]; then
  echo "BLOCKED: Simt oneshot still running — leave overlay alone. Re-run later."
  echo "$BUSY"
  echo "BLOCKED_SIMT_BUSY" | tee -a "$SUMMARY"
  exit 3
fi

if [[ ! -f /tmp/run_cf_mb_opsim.py ]]; then
  for c in \
    /mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/examples/ascend/run_cf_mb_opsim.py \
    /home/happybot/projects/tilelang-deepseek/examples/ascend/run_cf_mb_opsim.py
  do
    [[ -f "$c" ]] && cp -f "$c" /tmp/run_cf_mb_opsim.py && break
  done
fi
cp -f "$SUITE"/run_opsim_generic.py /tmp/ 2>/dev/null || true
cp -f "$SUITE"/harvest_cf_summary.py /tmp/ 2>/dev/null || true
cp -f "$SUITE"/harvest_simt_vmi_compare.py /tmp/ 2>/dev/null || true
sed -i 's/\r$//' "$SUITE"/*.py "$SUITE"/kernels/*.py /tmp/run_opsim_*.py 2>/dev/null || true

if [[ ! -f "$BK" ]]; then cp -a "$DEPS/build/lib/libtilelang.so" "$BK"; fi
if [[ ! -f "$OV/build/lib/libtilelang.so" ]]; then
  echo "BLOCKED: overlay lib missing at $OV/build/lib/libtilelang.so"
  echo "BLOCKED_NO_OVERLAY" | tee -a "$SUMMARY"
  exit 4
fi

# Install OVERLAY for SimdVF (GetPredicate PrimExpr + SimdVFLower*)
cp -a "$OV/build/lib/libtilelang.so" "$DEPS/build/lib/libtilelang.so"
echo "OVERLAY_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
echo "OV_nm:"; nm -D "$DEPS/build/lib/libtilelang.so" 2>/dev/null | grep -E 'GetPredicate|SimdVFLower' | head -20 || true
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib from $BK"' EXIT

set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1

compile_run_generic() {
  local SCRIPT=$1; shift
  local TAG=$1; shift
  echo "==== COMPILE $TAG (SimdVF/pto/overlay) ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Immutable|alloc_var|Error|Traceback|COMPILE_OK|RuntimeError|Unresolved|layout|InverseAffine|SimdVF|GetPredicate|if_then_else|VerifyParallel' \
      "$OUT/logs/compile_${TAG}.log" | tail -50 | tee -a "$SUMMARY" || true
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

echo "==== PHASE CF1V thresh kill KEEP (mask select) ===="
compile_run_generic "$SUITE/kernels/cf1v_pred_thresh_keep.py" cf1v_e256_k1_t64_keep 256 1 64
compile_run_generic "$SUITE/kernels/cf1v_pred_thresh_keep.py" cf1v_e256_k8_t64_keep 256 8 64

echo "==== PHASE CF2V remat + kill-in-shared (mask select) ===="
compile_run_generic "$SUITE/kernels/cf2v_remat_thresh_kill_shared.py" cf2v_e256_k8_t64_remat 256 8 64

echo "==== PHASE CF3V remat_idx (mask select) ===="
compile_run_generic "$SUITE/kernels/cf3v_remat_idx.py" cf3v_e256_k8_t64_remat_idx 256 8 64

echo "==== PHASE CF4V nested mask select ===="
compile_run_generic "$SUITE/kernels/cf4v_nested_if.py" cf4v_e256_t64_pfat5 256 64 5
compile_run_generic "$SUITE/kernels/cf4v_nested_if.py" cf4v_e256_t64_pfat25 256 64 25

echo "==== PHASE CF5V div+near0 mask select ===="
compile_run_generic "$SUITE/kernels/cf5v_div_ulp_branch.py" cf5v_e256_t64_pnear5 256 64 5
compile_run_generic "$SUITE/kernels/cf5v_div_ulp_branch.py" cf5v_e256_t64_pnear25 256 64 25

echo "==== PHASE CF6V Newton+near0 mask select ===="
compile_run_generic "$SUITE/kernels/cf6v_newton_branch.py" cf6v_e256_t64_pnear5 256 64 5
compile_run_generic "$SUITE/kernels/cf6v_newton_branch.py" cf6v_e256_t64_pnear25 256 64 25

echo "==== HARVEST CF VMI ===="
"$PY" "$SUITE/harvest_cf_summary.py" 2>&1 | tee "$OUT/harvest_cf1v_cf6v.log" || true

echo "==== SUMMARY_cf1v_cf6v_raw ===="
cat "$SUMMARY"
echo DONE_CF1V_CF6V
