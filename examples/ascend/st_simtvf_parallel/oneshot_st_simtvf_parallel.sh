#!/usr/bin/env bash
# SimtVF Parallel ST suite on pto-b10 (REGRESSION-2 recipe).
# Prefer deps-native lib; target=ascend cython NOT pto.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
LOG=$OUT/oneshot.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== st_simtvf_parallel oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

DEPS=${TILELANG_DEPS:-/home/happybot/projects/tilelang-pto-vmi-deps-stack}
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
BK=/tmp/libtilelang.so.deps_backup_ab
# Prefer AB-green Sep18 2-arg SimtVF lib; fall back to st backup if missing
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

HERE=$(cd "$(dirname "$0")" && pwd)
# Prefer unpacked suite next to this script; also accept /tmp copies
SUITE=${SUITE:-$HERE}
if [[ ! -f "$SUITE/common_asc_harness.py" ]]; then
  SUITE=/tmp/st_simtvf_parallel_suite
fi
export ST_SIMTVF_OUT=$OUT
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

# Ensure ACL opsim helper
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

# Backup + restore deps lib (never leave overlay)
if [[ ! -f "$BK" ]]; then cp -a "$DEPS/build/lib/libtilelang.so" "$BK"; fi
cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
echo "DEPS_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
nm -D "$DEPS/build/lib/libtilelang.so" 2>/dev/null | grep -E 'GetPredicate' | head -5 || true

set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1

SUMMARY="$OUT/SUMMARY_raw.txt"
: > "$SUMMARY"

compile_run_topk() {
  local ST=$1 E=$2 K=$3 T=$4
  local TAG="${ST}_e${E}_k${K}_t${T}"
  local PYFILE="$SUITE/kernels/${ST}_"*
  # map st id to file
  local SCRIPT
  case "$ST" in
    sv2) SCRIPT="$SUITE/kernels/sv2_topk_keep.py" ;;
    sv3) SCRIPT="$SUITE/kernels/sv3_topk_reload_remat.py" ;;
    sv4) SCRIPT="$SUITE/kernels/sv4_topk_index_remat.py" ;;
    *) echo "bad ST $ST"; return 0 ;;
  esac
  echo "==== COMPILE $TAG ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SCRIPT" "$E" "$K" "$T" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'int2|make_int2|excess|stl_pair|Error|COMPILE_OK|PATCHED' "$OUT/logs/compile_${TAG}.log" | tail -40 | tee -a "$SUMMARY" || true
    return 0
  fi
  set +e
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_topk.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time' "$OUT/opsim_${TAG}.log" | head -5 | tr '\n' ' ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

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
    return 0
  fi
  set +e
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  echo "$TAG $PASSLINE" | tee -a "$SUMMARY"
}

echo "==== PHASE SV2 matrix (priority) ===="
# E=256 K={1,8} threads={32,128}
compile_run_topk sv2 256 1 32
compile_run_topk sv2 256 8 32
compile_run_topk sv2 256 1 128
compile_run_topk sv2 256 8 128

echo "==== PHASE SV3 vs SV2 @ T=32 ===="
compile_run_topk sv3 256 1 32
compile_run_topk sv3 256 8 32

echo "==== PHASE SV4 vs SV2 K=8 T=32/128 ===="
compile_run_topk sv4 256 8 32
compile_run_topk sv4 256 8 128

echo "==== PHASE SV1 SV7 SV8 ===="
compile_run_generic "$SUITE/kernels/sv1_stream_eltwise.py" sv1_e256_t32 256 32
compile_run_generic "$SUITE/kernels/sv1_stream_eltwise.py" sv1_e2048_t32 2048 32
compile_run_generic "$SUITE/kernels/sv7_block_reduce_128.py" sv7_n128_t32 128 32
compile_run_generic "$SUITE/kernels/sv7_block_reduce_128.py" sv7_n128_t128 128 128
compile_run_generic "$SUITE/kernels/sv8_block_reduce_32x32.py" sv8_32x32_t32 32
compile_run_generic "$SUITE/kernels/sv8_block_reduce_32x32.py" sv8_32x32_t128 128

echo "==== PHASE SV5 SV6 best-effort ===="
compile_run_generic "$SUITE/kernels/sv5_bcast_multiconsumer.py" sv5_e256_t32_frag 256 32 frag
compile_run_generic "$SUITE/kernels/sv5_bcast_multiconsumer.py" sv5_e256_t32_reload 256 32 reload
compile_run_generic "$SUITE/kernels/sv6_group_scale_remat.py" sv6_r64_c128_g16_t32_remat 64 128 16 32 remat
compile_run_generic "$SUITE/kernels/sv6_group_scale_remat.py" sv6_r64_c128_g16_t32_reload 64 128 16 32 reload

# Always restore deps lib
cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
echo "restored deps lib"

# Harvest into suite reports (also OUT)
"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest.log" || true
# Copy SUMMARY into suite if mounted; else leave under OUT
cp -f "$OUT/SUMMARY.md" "$SUITE/reports/SUMMARY.md" 2>/dev/null || true
cp -f "$OUT"/reports/*.md "$SUITE/reports/" 2>/dev/null || true
# Also dump per-tag md from harvest's REPORTS path which is suite/reports
ls -la "$SUITE/reports" "$OUT" | head -80

echo "==== SUMMARY_raw ===="
cat "$SUMMARY"
echo DONE_ST_SIMTVF_PARALLEL
