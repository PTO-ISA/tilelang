#!/usr/bin/env bash
# Case-3 SV5V/SV6V/SV7V/SV8V — SimdVF / VMI twins (target=pto + OVERLAY lib) on pto-b10
# Do NOT run while Simt oneshot_sv5_sv6_sv8.sh holds deps-native libtilelang.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
LOG=$OUT/oneshot_sv5v_sv6v_sv8v.log
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

SUMMARY="$OUT/SUMMARY_sv5v_sv6v_sv8v_raw.txt"
: > "$SUMMARY"

# Refuse to fight Simt oneshot over libtilelang
BUSY=$(pgrep -af 'oneshot_sv5_sv6_sv8\.sh|sv5_reduce_small_eltwise\.py|sv6_reduce_large_eltwise\.py|sv8_case3_bcast\.py' || true)
BUSY=$(echo "$BUSY" | grep -vE 'sv5v|sv6v|sv8v|oneshot_sv5v|pgrep' || true)
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

echo "==== PHASE SV5V (small-G reduce → reduced eltwise, SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv5v_reduce_small_eltwise.py" sv5v_r64_c128_g16_t64_keep_reg 64 128 16 64 keep_reg
compile_run_generic "$SUITE/kernels/sv5v_reduce_small_eltwise.py" sv5v_r64_c128_g16_t64_ub_reload 64 128 16 64 ub_reload
compile_run_generic "$SUITE/kernels/sv6v_reduce_mid_eltwise.py" sv6v_r64_c128_g32_t64 64 128 32 64

echo "==== PHASE SV6V (large-G reduce → reduced eltwise, SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g64_t64 64 128 64 64
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g128_t64 64 128 128 64

echo "==== PHASE SV8V Case-3 (live + spill_dist, SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv8v_case3_bcast.py" sv8v_r64_c128_g16_t64_live 64 128 16 64 live
compile_run_generic "$SUITE/kernels/sv8v_case3_bcast.py" sv8v_r64_c128_g16_t64_spill_dist 64 128 16 64 spill_dist

"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv5v_sv6v_sv8v.log" || true
"$PY" "$SUITE/harvest_simt_vmi_compare.py" --out "$OUT/reports/ST_VMI_COMPARE.md" 2>&1 | tee "$OUT/harvest_simt_vmi_compare.log" || true
cp -f "$OUT/reports/ST_VMI_COMPARE.md" "$SUITE/reports/ST_VMI_COMPARE.md" 2>/dev/null || true

echo "==== SUMMARY_sv5v_sv6v_sv8v_raw ===="
cat "$SUMMARY"
echo DONE_SV5V_SV6V_SV8V
