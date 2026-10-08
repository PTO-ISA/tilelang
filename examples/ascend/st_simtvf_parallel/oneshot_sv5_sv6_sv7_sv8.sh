#!/usr/bin/env bash
# Case-3 SV5/SV6/SV7/SV8 — reduce→reduced-eltwise (+ SV8 bcast) on pto-b10
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
LOG=$OUT/oneshot_sv5_sv6_sv7_sv8.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== st_simtvf_parallel Case-3 SV5/SV6/SV7/SV8 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

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
  SUITE=/tmp/st_simtvf_parallel_suite
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

SUMMARY="$OUT/SUMMARY_sv5_sv6_sv7_sv8_raw.txt"
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
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time|Total cycles|IPC|membar|MTE' "$OUT/opsim_${TAG}.log" | head -12 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

echo "==== PHASE SV5 (G=16 small: keep_in_rf vs ub_stream — group-input RF capacity) ===="
compile_run_generic "$SUITE/kernels/sv5_reduce_small_eltwise.py" sv5_r64_c128_g16_t32_keep_in_rf 64 128 16 32 keep_in_rf
compile_run_generic "$SUITE/kernels/sv5_reduce_small_eltwise.py" sv5_r64_c128_g16_t32_ub_stream  64 128 16 32 ub_stream

echo "==== PHASE SV6 (G=32 mid: keep_in_warp vs reload) ===="
compile_run_generic "$SUITE/kernels/sv6_reduce_mid_eltwise.py" sv6_r64_c128_g32_t32_keep_in_warp 64 128 32 32 keep_in_warp
compile_run_generic "$SUITE/kernels/sv6_reduce_mid_eltwise.py" sv6_r64_c128_g32_t32_reload       64 128 32 32 reload

echo "==== PHASE SV7 (G=128 primary + G=64: multiwarp_ub vs reload) ===="
compile_run_generic "$SUITE/kernels/sv7_reduce_large_eltwise.py" sv7_r64_c128_g128_t32_multiwarp_ub 64 128 128 32 multiwarp_ub
compile_run_generic "$SUITE/kernels/sv7_reduce_large_eltwise.py" sv7_r64_c128_g128_t32_reload       64 128 128 32 reload
compile_run_generic "$SUITE/kernels/sv7_reduce_large_eltwise.py" sv7_r64_c128_g64_t32_multiwarp_ub  64 128 64 32 multiwarp_ub
compile_run_generic "$SUITE/kernels/sv7_reduce_large_eltwise.py" sv7_r64_c128_g64_t32_reload        64 128 64 32 reload

echo "==== PHASE SV8 Case-3 (live + spill_dist) ===="
compile_run_generic "$SUITE/kernels/sv8_case3_bcast.py" sv8_r64_c128_g16_t32_live 64 128 16 32 live
compile_run_generic "$SUITE/kernels/sv8_case3_bcast.py" sv8_r64_c128_g16_t32_spill_dist 64 128 16 32 spill_dist

"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv5_sv6_sv8.log" || true

echo "==== SUMMARY_sv5_sv6_sv7_sv8_raw ===="
cat "$SUMMARY"
echo DONE_SV5_SV6_SV7_SV8
