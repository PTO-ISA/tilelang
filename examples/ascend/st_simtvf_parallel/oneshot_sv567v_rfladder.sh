#!/usr/bin/env bash
# Case-3 SV5V/SV6V/SV7V RF-capacity ladder — SimdVF / VMI twins (target=pto + OVERLAY lib) on pto-b10
# Arms: sv5v {keep_in_rf|ub_stream} / sv6v {reload[,keep_in_warp]} / sv7v {reload|multiwarp_ub}
# Do NOT run while Simt oneshot_sv5_sv6_sv8.sh holds deps-native libtilelang.
set -euo pipefail
OUT=${OUT:-/tmp/st_simtvf_parallel}
LOG=$OUT/oneshot_sv567v_rfladder.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== st_simtvf_parallel Case-3 SV5V/SV6V/SV7V/SV8V (VMI) oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

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

SUMMARY="$OUT/SUMMARY_sv567v_rfladder_raw.txt"
: > "$SUMMARY"

# Refuse to fight Simt oneshot over libtilelang
BUSY=$(pgrep -af 'oneshot_sv5_sv6_sv8\.sh|oneshot_sv5_sv6_sv7_sv8\.sh|oneshot_sv567_rfladder\.sh|sv5_reduce_small_eltwise\.py|sv6_reduce_mid_eltwise\.py|sv7_reduce_large_eltwise\.py|sv8_case3_bcast\.py' || true)
BUSY=$(echo "$BUSY" | grep -vE 'sv5v|sv6v|sv7v|sv8v|oneshot_sv5v|oneshot_sv567v|pgrep' || true)
if [[ -n "${BUSY}" ]]; then
  echo "BLOCKED: Simt Case-3 oneshot still running — leave overlay alone. Re-run later."
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
cp -f "$SUITE"/run_opsim_topk.py /tmp/ 2>/dev/null || true
cp -f "$SUITE"/harvest_report.py /tmp/ 2>/dev/null || true
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
# DEPS before CAMO on PYTHONPATH (SV3V lesson)
export PYTHONPATH="$SUITE:$DEPS:$DEPS/build:$CAMO:${ASC}/python/site-packages:${PYTHONPATH:-}"
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
    grep -E 'Immutable|alloc_var|Error|Traceback|COMPILE_OK|RuntimeError|Unresolved|layout|InverseAffine|SimdVF|GetPredicate' \
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

echo "==== PHASE SV5V (G=16: keep_in_rf vs ub_stream, lane-legal 1-D Parallel) ===="
compile_run_generic "$SUITE/kernels/sv5v_reduce_small_eltwise.py" sv5v_r64_c128_g16_t64_ub_stream  64 128 16 64 ub_stream
compile_run_generic "$SUITE/kernels/sv5v_reduce_small_eltwise.py" sv5v_r64_c128_g16_t64_keep_in_rf 64 128 16 64 keep_in_rf

echo "==== PHASE SV6V (G=32: primary reload; optional keep probe) ===="
compile_run_generic "$SUITE/kernels/sv6v_reduce_mid_eltwise.py" sv6v_r64_c128_g32_t64_reload       64 128 32 64 reload
compile_run_generic "$SUITE/kernels/sv6v_reduce_mid_eltwise.py" sv6v_r64_c128_g32_t64_keep_in_warp 64 128 32 64 keep_in_warp

echo "==== PHASE SV7V (G=128 + G=64: reload + chunked multiwarp_ub) ===="
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g128_t64_reload       64 128 128 64 reload
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g128_t64_multiwarp_ub 64 128 128 64 multiwarp_ub
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g64_t64_reload        64 128 64 64 reload
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g64_t64_multiwarp_ub  64 128 64 64 multiwarp_ub

"$PY" "$SUITE/harvest_sv567_rfladder.py" "$OUT" 2>&1 | tee "$OUT/harvest_sv567_rfladder_run.log" || true
cp -f "$OUT/RESULT.md" "$OUT/RESULT_sv567_rfladder.md" 2>/dev/null || true
"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv567v_rfladder.log" || true

echo "==== SUMMARY_sv567v_rfladder_raw ===="
cat "$SUMMARY"
echo DONE_SV567V_RFLADDER
