#!/usr/bin/env bash
# Parallel VMI / SimdVF opsim for SV1V–SV9V (overlay). Compile missing .so then -P opsim.
# Never leave overlay installed — trap restores deps backup.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
SUMMARY=$OUT/SUMMARY_sv1v_sv9v_raw.txt
SUITE=${SUITE:-/tmp/st_simtvf_parallel_suite}
LOG=$OUT/parallel_resume_sv1v_sv9v.log
JOBS=${JOBS:-5}
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports" "$OUT/parallel_tmp"
exec > >(tee -a "$LOG") 2>&1
echo "=== PARALLEL_RESUME_VMI start $(date -Is) host=$(hostname) JOBS=$JOBS ==="

DEPS=/home/happybot/projects/tilelang-pto-vmi-deps-stack
OV=/mnt/fluxdata/happybot/projects/tilelang-pr258-overlay
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
BK=/tmp/libtilelang.so.deps_backup_ab
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

HERE=$(cd "$(dirname "$0")" && pwd)
if [[ ! -f "$SUITE/common_pto_harness.py" ]]; then
  SUITE=$HERE
fi
if [[ ! -f "$SUITE/common_pto_harness.py" ]]; then
  SUITE=/tmp/st_simtvf_parallel_suite
fi
export ST_SIMTVF_OUT=$OUT
export PYTHONPATH="$SUITE:${PYTHONPATH:-}"

# Refuse if a *real* Simt oneshot/resume script is running (ignore stale bash -c monitors)
BUSY=$(pgrep -af '/oneshot_sv[0-9][^v].*\.sh|/parallel_resume_sv1_sv9\.sh|oneshot_sv1_sv9\.sh|oneshot_sv5_sv6|oneshot_sv6_sv8|oneshot_sv9\.sh' || true)
BUSY=$(echo "$BUSY" | grep -vE 'sv[0-9]v|oneshot_sv[0-9]v|parallel_resume_sv1v|pgrep|grep -E' || true)
# Also block if run_opsim_generic is mid-flight on a non-v tag (active Simt opsim)
SIMT_OP=$(pgrep -af 'run_opsim_generic.py sv[0-9]_' || true)
SIMT_OP=$(echo "$SIMT_OP" | grep -vE 'sv[0-9]v_|pgrep' || true)
if [[ -n "${BUSY}${SIMT_OP}" ]]; then
  echo "BLOCKED: Simt oneshot/resume/opsim still running"
  echo "$BUSY"
  echo "$SIMT_OP"
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
  echo "BLOCKED: overlay lib missing"
  exit 4
fi

# Install OVERLAY for SimdVF
cp -a "$OV/build/lib/libtilelang.so" "$DEPS/build/lib/libtilelang.so"
echo "OVERLAY_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib from $BK $(date -Is)"' EXIT

set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1

: >> "$SUMMARY"
echo "PARALLEL_RESUME_VMI $(date -Is) JOBS=$JOBS" >> "$SUMMARY"

compile_one() {
  local KIND=$1; shift
  local TAG=$1; shift
  echo "==== COMPILE $TAG ===="
  set +e
  if [[ "$KIND" == topk ]]; then
    "$PY" "$SUITE/kernels/sv9v_topk_e2e.py" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  else
    local SCRIPT=$1; shift
    "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  fi
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
}

echo "==== PHASE COMPILE missing VMI .so ===="
[[ -f $OUT/so/sv1v_e256_t64.so ]] || compile_one generic sv1v_e256_t64 "$SUITE/kernels/sv1v_stream_eltwise.py" 256 64
[[ -f $OUT/so/sv1v_e2048_t64.so ]] || compile_one generic sv1v_e2048_t64 "$SUITE/kernels/sv1v_stream_eltwise.py" 2048 64
[[ -f $OUT/so/sv2v_r32_c32_t64_frag_live.so ]] || compile_one generic sv2v_r32_c32_t64_frag_live "$SUITE/kernels/sv2v_eltwise_bcast_rf.py" 32 32 64 frag_live
[[ -f $OUT/so/sv2v_r32_c32_t64_reload.so ]] || compile_one generic sv2v_r32_c32_t64_reload "$SUITE/kernels/sv2v_eltwise_bcast_rf.py" 32 32 64 reload
[[ -f $OUT/so/sv3v_m24_vl64_k16_t64_keep.so ]] || compile_one generic sv3v_m24_vl64_k16_t64_keep "$SUITE/kernels/sv3v_gemv_partial_keep.py" 24 64 16 64 keep
[[ -f $OUT/so/sv3v_m32_vl64_k16_t64_keep.so ]] || compile_one generic sv3v_m32_vl64_k16_t64_keep "$SUITE/kernels/sv3v_gemv_partial_keep.py" 32 64 16 64 keep
[[ -f $OUT/so/sv3v_m32_vl64_k16_t64_split_cm16.so ]] || compile_one generic sv3v_m32_vl64_k16_t64_split_cm16 "$SUITE/kernels/sv3v_gemv_partial_keep.py" 32 64 16 64 split 16
[[ -f $OUT/so/sv4v_e256_b8_t64_keep_idx.so ]] || compile_one generic sv4v_e256_b8_t64_keep_idx "$SUITE/kernels/sv4v_index_gather_psum.py" 256 8 64 keep_idx
[[ -f $OUT/so/sv4v_e256_b8_t64_remat_idx.so ]] || compile_one generic sv4v_e256_b8_t64_remat_idx "$SUITE/kernels/sv4v_index_gather_psum.py" 256 8 64 remat_idx
[[ -f $OUT/so/sv4v_e256_b16_t64_keep_idx.so ]] || compile_one generic sv4v_e256_b16_t64_keep_idx "$SUITE/kernels/sv4v_index_gather_psum.py" 256 16 64 keep_idx
[[ -f $OUT/so/sv4v_e256_b16_t64_remat_idx.so ]] || compile_one generic sv4v_e256_b16_t64_remat_idx "$SUITE/kernels/sv4v_index_gather_psum.py" 256 16 64 remat_idx
[[ -f $OUT/so/sv5v_r64_c128_g16_t64_keep_reg.so ]] || compile_one generic sv5v_r64_c128_g16_t64_keep_reg "$SUITE/kernels/sv5v_reduce_small_eltwise.py" 64 128 16 64 keep_reg
[[ -f $OUT/so/sv5v_r64_c128_g16_t64_ub_reload.so ]] || compile_one generic sv5v_r64_c128_g16_t64_ub_reload "$SUITE/kernels/sv5v_reduce_small_eltwise.py" 64 128 16 64 ub_reload
[[ -f $OUT/so/sv6v_r64_c128_g32_t64.so ]] || compile_one generic sv6v_r64_c128_g32_t64 "$SUITE/kernels/sv6v_reduce_mid_eltwise.py" 64 128 32 64
[[ -f $OUT/so/sv7v_r64_c128_g64_t64.so ]] || compile_one generic sv7v_r64_c128_g64_t64 "$SUITE/kernels/sv7v_reduce_large_eltwise.py" 64 128 64 64
[[ -f $OUT/so/sv7v_r64_c128_g128_t64.so ]] || compile_one generic sv7v_r64_c128_g128_t64 "$SUITE/kernels/sv7v_reduce_large_eltwise.py" 64 128 128 64
[[ -f $OUT/so/sv8v_r64_c128_g16_t64_live.so ]] || compile_one generic sv8v_r64_c128_g16_t64_live "$SUITE/kernels/sv8v_case3_bcast.py" 64 128 16 64 live
[[ -f $OUT/so/sv8v_r64_c128_g16_t64_spill_dist.so ]] || compile_one generic sv8v_r64_c128_g16_t64_spill_dist "$SUITE/kernels/sv8v_case3_bcast.py" 64 128 16 64 spill_dist
[[ -f $OUT/so/sv9v_e256_k1_t64_keep.so ]] || compile_one topk sv9v_e256_k1_t64_keep 256 1 64 keep
[[ -f $OUT/so/sv9v_e256_k8_t64_keep.so ]] || compile_one topk sv9v_e256_k8_t64_keep 256 8 64 keep
[[ -f $OUT/so/sv9v_e256_k8_t64_remat_scores.so ]] || compile_one topk sv9v_e256_k8_t64_remat_scores 256 8 64 remat_scores
[[ -f $OUT/so/sv9v_e256_k8_t64_remat_idx.so ]] || compile_one topk sv9v_e256_k8_t64_remat_idx 256 8 64 remat_idx

is_pass() {
  local TAG=$1
  local R=$OUT/opsim_${TAG}_result.txt
  [[ -f "$R" ]] || return 1
  grep -qE 'ok=True|^PASS|PASS' "$R" 2>/dev/null
}

OPSIM_TAGS=()
for t in \
  sv1v_e256_t64 sv1v_e2048_t64 \
  sv2v_r32_c32_t64_frag_live sv2v_r32_c32_t64_reload \
  sv3v_m24_vl64_k16_t64_keep sv3v_m32_vl64_k16_t64_keep sv3v_m32_vl64_k16_t64_split_cm16 \
  sv4v_e256_b8_t64_keep_idx sv4v_e256_b8_t64_remat_idx sv4v_e256_b16_t64_keep_idx sv4v_e256_b16_t64_remat_idx \
  sv5v_r64_c128_g16_t64 sv6v_r64_c128_g32_t64 \
  sv7v_r64_c128_g64_t64 sv7v_r64_c128_g128_t64 \
  sv8v_r64_c128_g16_t64_live sv8v_r64_c128_g16_t64_spill_dist \
  sv9v_e256_k1_t64_keep sv9v_e256_k8_t64_keep sv9v_e256_k8_t64_remat_scores sv9v_e256_k8_t64_remat_idx
do
  if [[ ! -f "$OUT/so/${t}.so" ]]; then
    echo "SKIP_NO_SO $t" | tee -a "$SUMMARY"
    continue
  fi
  if is_pass "$t"; then
    echo "SKIP_ALREADY_PASS $t" | tee -a "$SUMMARY"
    continue
  fi
  OPSIM_TAGS+=("$t")
done
echo "OPSIM_PARALLEL_TAGS=${OPSIM_TAGS[*]:-}"

run_one_opsim() {
  local TAG=$1
  local KIND=generic
  case "$TAG" in
    sv9v_*) KIND=topk ;;
  esac
  local RUNNER=/tmp/run_opsim_generic.py
  [[ "$KIND" == topk ]] && RUNNER=/tmp/run_opsim_topk.py
  echo "==== OPSIM $TAG ($KIND) ===="
  set +e
  "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
    "$RUNNER" -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time' "$OUT/opsim_${TAG}.log" | head -4 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

export -f run_one_opsim is_pass
export OUT SIM SUMMARY

echo "==== PHASE PARALLEL OPSIM -P $JOBS ===="
if [[ ${#OPSIM_TAGS[@]} -gt 0 ]]; then
  printf '%s\n' "${OPSIM_TAGS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one_opsim "$@"' _ {}
else
  echo "NO_OPSIM_NEEDED"
fi

echo "==== HARVEST ===="
"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv1v_sv9v.log" || true
"$PY" "$SUITE/harvest_simt_vmi_compare.py" --out "$OUT/reports/ST_VMI_COMPARE.md" \
  2>&1 | tee "$OUT/harvest_simt_vmi_compare.log" || true
cp -f "$OUT/reports/ST_VMI_COMPARE.md" "$SUITE/reports/ST_VMI_COMPARE.md" 2>/dev/null || true

# PASS table for VMI
{
  printf '%-48s %-6s %8s %12s\n' TAG PASS us IPC
  printf '%s\n' '--------------------------------------------------------------------------------'
  for t in \
    sv1v_e256_t64 sv1v_e2048_t64 \
    sv2v_r32_c32_t64_frag_live sv2v_r32_c32_t64_reload \
    sv3v_m24_vl64_k16_t64_keep sv3v_m32_vl64_k16_t64_keep sv3v_m32_vl64_k16_t64_split_cm16 \
    sv4v_e256_b8_t64_keep_idx sv4v_e256_b8_t64_remat_idx sv4v_e256_b16_t64_keep_idx sv4v_e256_b16_t64_remat_idx \
    sv5v_r64_c128_g16_t64 sv6v_r64_c128_g32_t64 \
    sv7v_r64_c128_g64_t64 sv7v_r64_c128_g128_t64 \
    sv8v_r64_c128_g16_t64_live sv8v_r64_c128_g16_t64_spill_dist \
    sv9v_e256_k1_t64_keep sv9v_e256_k8_t64_keep sv9v_e256_k8_t64_remat_scores sv9v_e256_k8_t64_remat_idx
  do
    st=MISSING; us=-; ipc=n/a
    if is_pass "$t"; then st=PASS
    elif [[ -f "$OUT/opsim_${t}_result.txt" ]]; then st=FAIL
    fi
    if [[ -f "$OUT/reports/${t}.md" ]]; then
      us=$(grep -oE 'wall_us: [0-9.]+' "$OUT/reports/${t}.md" 2>/dev/null | head -1 | awk '{print $2}')
      ipc=$(grep -oE 'IPC_proxy=[0-9.]+' "$OUT/reports/${t}.md" 2>/dev/null | head -1 | sed 's/IPC_proxy=//')
    fi
    printf '%-48s %-6s %8s %12s\n' "$t" "$st" "${us:--}" "${ipc:-n/a}"
  done
} | tee "$OUT/PASS_TABLE_sv1v_sv9v.txt"

echo "==== SUMMARY_sv1v_sv9v_raw ===="
tail -80 "$SUMMARY" || true
echo DONE_PARALLEL_SV1V_SV9V
