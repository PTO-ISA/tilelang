#!/usr/bin/env bash
# Full VMI / SimdVF matrix SV1V–SV9V (target=pto + OVERLAY lib) on pto-b10
# Includes Case-3 twins SV5V–SV8V plus new SV1V/SV2V/SV3V/SV4V/SV9V.
# Source+compile ready; opsim is sequential later — safe when Simt idle.
# Do NOT run while any Simt oneshot holds deps-native libtilelang.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
LOG=$OUT/oneshot_sv1v_sv9v.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== st_simtvf_parallel SV1V–SV9V (VMI) oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

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

SUMMARY="$OUT/SUMMARY_sv1v_sv9v_raw.txt"
: > "$SUMMARY"

# Refuse to fight Simt oneshot over libtilelang
BUSY=$(pgrep -af 'oneshot_sv[0-9]|sv[0-9]_' || true)
BUSY=$(echo "$BUSY" | grep -vE 'sv[0-9]v|oneshot_sv[0-9]v|pgrep' || true)
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

echo "==== PHASE SV1V stream eltwise (SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv1v_stream_eltwise.py" sv1v_e256_t64 256 64
compile_run_generic "$SUITE/kernels/sv1v_stream_eltwise.py" sv1v_e2048_t64 2048 64

echo "==== PHASE SV2V live RF eltwise + scale bcast (SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv2v_eltwise_bcast_rf.py" sv2v_r32_c32_t64_frag_live 32 32 64 frag_live
compile_run_generic "$SUITE/kernels/sv2v_eltwise_bcast_rf.py" sv2v_r32_c32_t64_reload 32 32 64 reload

echo "==== PHASE SV3V GEMV Acc KEEP / split (SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv3v_gemv_partial_keep.py" sv3v_m24_vl64_k16_t64_keep 24 64 16 64 keep
compile_run_generic "$SUITE/kernels/sv3v_gemv_partial_keep.py" sv3v_m32_vl64_k16_t64_keep 32 64 16 64 keep
compile_run_generic "$SUITE/kernels/sv3v_gemv_partial_keep.py" sv3v_m32_vl64_k16_t64_split_cm16 32 64 16 64 split 16

echo "==== PHASE SV4V gather+psum keep_idx / remat_idx (SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv4v_index_gather_psum.py" sv4v_e256_b8_t64_keep_idx 256 8 64 keep_idx
compile_run_generic "$SUITE/kernels/sv4v_index_gather_psum.py" sv4v_e256_b8_t64_remat_idx 256 8 64 remat_idx
compile_run_generic "$SUITE/kernels/sv4v_index_gather_psum.py" sv4v_e256_b16_t64_keep_idx 256 16 64 keep_idx
compile_run_generic "$SUITE/kernels/sv4v_index_gather_psum.py" sv4v_e256_b16_t64_remat_idx 256 16 64 remat_idx

echo "==== PHASE SV5V (small-G reduce → reduced eltwise, SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv5v_reduce_small_eltwise.py" sv5v_r64_c128_g16_t64_keep_reg 64 128 16 64 keep_reg
compile_run_generic "$SUITE/kernels/sv5v_reduce_small_eltwise.py" sv5v_r64_c128_g16_t64_ub_reload 64 128 16 64 ub_reload
compile_run_generic "$SUITE/kernels/sv6v_reduce_mid_eltwise.py" sv6v_r64_c128_g32_t64 64 128 32 64

echo "==== PHASE SV7V (large-G reduce → reduced eltwise, SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g64_t64 64 128 64 64
compile_run_generic "$SUITE/kernels/sv7v_reduce_large_eltwise.py" sv7v_r64_c128_g128_t64 64 128 128 64

echo "==== PHASE SV8V Case-3 (live + spill_dist, SimdVF) ===="
compile_run_generic "$SUITE/kernels/sv8v_case3_bcast.py" sv8v_r64_c128_g16_t64_live 64 128 16 64 live
compile_run_generic "$SUITE/kernels/sv8v_case3_bcast.py" sv8v_r64_c128_g16_t64_spill_dist 64 128 16 64 spill_dist

echo "==== PHASE SV9V topk e2e (SimdVF) ===="
compile_run_topk() {
  local TAG=$1; shift
  echo "==== COMPILE $TAG (SimdVF/pto/overlay) ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$SUITE/kernels/sv9v_topk_e2e.py" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Error|Traceback|COMPILE_OK|RuntimeError|SimdVF|GetPredicate|alloc_var' \
      "$OUT/logs/compile_${TAG}.log" | tail -40 | tee -a "$SUMMARY" || true
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
  set +e
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_topk.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time|Total cycles|IPC' "$OUT/opsim_${TAG}.log" | head -8 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}
compile_run_topk sv9v_e256_k1_t64_keep 256 1 64 keep
compile_run_topk sv9v_e256_k8_t64_keep 256 8 64 keep
compile_run_topk sv9v_e256_k8_t64_remat_scores 256 8 64 remat_scores
compile_run_topk sv9v_e256_k8_t64_remat_idx 256 8 64 remat_idx

# Discover harvest script name from base

"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv1v_sv9v.log" || true
"$PY" "$SUITE/harvest_simt_vmi_compare.py" --out "$OUT/reports/ST_VMI_COMPARE.md" 2>&1 | tee "$OUT/harvest_simt_vmi_compare.log" || true
cp -f "$OUT/reports/ST_VMI_COMPARE.md" "$SUITE/reports/ST_VMI_COMPARE.md" 2>/dev/null || true

echo "==== SUMMARY_sv1v_sv9v_raw ===="
cat "$SUMMARY"
echo DONE_SV1V_SV9V
