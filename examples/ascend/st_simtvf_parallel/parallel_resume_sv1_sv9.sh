#!/usr/bin/env bash
# Stop sequential oneshot; compile remaining; parallel opsim for SV1-SV9 leftovers.
set -euo pipefail
OUT=/tmp/st_simtvf_parallel
SUMMARY=$OUT/SUMMARY_sv1_sv9_raw.txt
SUITE=/tmp/st_simtvf_parallel_suite
LOG=$OUT/parallel_resume_sv1_sv9.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports" "$OUT/parallel_tmp"
exec > >(tee -a "$LOG") 2>&1
echo "=== PARALLEL_RESUME start $(date -Is) host=$(hostname) ==="

DEPS=/home/happybot/projects/tilelang-pto-vmi-deps-stack
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
BK=/tmp/libtilelang.so.deps_backup_ab
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

# --- 1) kill sequential oneshot tree only ---
echo "=== KILL sequential oneshot ==="
ROOT=$(pgrep -f '/tmp/st_simtvf_parallel_suite/oneshot_sv1_sv9.sh' | head -1 || true)
echo "ROOT=$ROOT"
if [[ -n "${ROOT:-}" ]]; then
  # kill process group if possible; else walk children
  kill -- -"$ROOT" 2>/dev/null || true
  # collect pid tree via /proc
  PIDS="$ROOT"
  QUEUE="$ROOT"
  while [[ -n "$QUEUE" ]]; do
    NEXT=""
    for p in $QUEUE; do
      for c in $(ls /proc/$p/task/$p/children 2>/dev/null); do
        PIDS="$PIDS $c"
        NEXT="$NEXT $c"
      done
      # fallback: ps --ppid
      for c in $(ps --ppid "$p" -o pid= 2>/dev/null); do
        PIDS="$PIDS $c"
        NEXT="$NEXT $c"
      done
    done
    QUEUE="$NEXT"
  done
  echo "KILL_PIDS=$PIDS"
  kill $PIDS 2>/dev/null || true
  sleep 2
  kill -9 $PIDS 2>/dev/null || true
fi
pkill -f '/tmp/st_simtvf_parallel_suite/oneshot_sv1_sv9.sh' 2>/dev/null || true
# only kill sim_dsl / msopprof clearly tied to st_simtvf_parallel OUT
pkill -f 'sim_dsl.sh --soc-version Ascend950PR_9599 --output /tmp/st_simtvf_parallel/opsim_' 2>/dev/null || true
sleep 1
if pgrep -f '/tmp/st_simtvf_parallel_suite/oneshot_sv1_sv9.sh' >/dev/null; then
  echo "WARN oneshot still alive"; pgrep -af oneshot_sv1_sv9 || true
else
  echo "ONESHOT_STOPPED"
fi

# --- 2) pin deps + marker ---
cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib $(date -Is)"' EXIT
echo "DEPS_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
echo "PARALLEL_RESUME $(date -Is)" >> "$SUMMARY"

# --- 3) env ---
set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1
export ST_SIMTVF_OUT=$OUT

cp -f "$SUITE"/run_opsim_*.py /tmp/ 2>/dev/null || true
cp -f "$SUITE"/harvest_report.py /tmp/ 2>/dev/null || true
sed -i 's/\r$//' "$SUITE"/*.py "$SUITE"/kernels/*.py /tmp/run_opsim_*.py 2>/dev/null || true

need_opsim() {
  local TAG=$1
  # already have PASS line in this SUMMARY run? still re-run if no PASS
  if grep -qE "^${TAG} PASS" "$SUMMARY" 2>/dev/null && [[ -f "$OUT/so/${TAG}.so" ]]; then
    return 1
  fi
  return 0
}

# Remaining matrix tags
REMAIN=(
  sv4_e256_b8_t32_keep_idx
  sv4_e256_b8_t32_remat_idx
  sv4_e256_b16_t32_keep_idx
  sv4_e256_b16_t32_remat_idx
  sv5_r64_c128_g16_t32
  sv6_r64_c128_g32_t32
  sv7_r64_c128_g64_t32
  sv7_r64_c128_g128_t32
  sv8_r64_c128_g16_t32_live
  sv8_r64_c128_g16_t32_spill_dist
  sv9_e256_k1_t32_keep
  sv9_e256_k8_t32_keep
  sv9_e256_k8_t32_remat_scores
  sv9_e256_k8_t32_remat_idx
)

echo "=== STATUS before compile ==="
for t in "${REMAIN[@]}"; do
  so=NO; pass=NO
  [[ -f "$OUT/so/${t}.so" ]] && so=YES
  grep -qE "^${t} PASS" "$SUMMARY" && pass=YES || true
  echo "  $t so=$so pass=$pass"
done

# --- 4) COMPILE missing .so (sequential; safe) ---
compile_one() {
  local kind=$1; shift
  local TAG=$1; shift
  echo "==== COMPILE $TAG ====" | tee -a "$SUMMARY"
  set +e
  if [[ "$kind" == topk ]]; then
    "$PY" "$SUITE/kernels/sv9_topk_e2e.py" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  else
    local SCRIPT=$1; shift
    "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  fi
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Error|Traceback|COMPILE_OK|RuntimeError|Immutable|Unresolved' "$OUT/logs/compile_${TAG}.log" | tail -40 | tee -a "$SUMMARY" || true
    return 0
  fi
  echo "COMPILE_OK $TAG" | tee -a "$SUMMARY"
}

echo "==== PHASE PARALLEL COMPILE missing ===="
# sv4 b16 / sv5 / sv6 may already have so+PASS — skip those
[[ -f $OUT/so/sv4_e256_b8_t32_keep_idx.so ]] || compile_one generic sv4_e256_b8_t32_keep_idx "$SUITE/kernels/sv4_index_gather_psum.py" 256 8 32 keep_idx
[[ -f $OUT/so/sv4_e256_b8_t32_remat_idx.so ]] || compile_one generic sv4_e256_b8_t32_remat_idx "$SUITE/kernels/sv4_index_gather_psum.py" 256 8 32 remat_idx
[[ -f $OUT/so/sv4_e256_b16_t32_keep_idx.so ]] || compile_one generic sv4_e256_b16_t32_keep_idx "$SUITE/kernels/sv4_index_gather_psum.py" 256 16 32 keep_idx
[[ -f $OUT/so/sv4_e256_b16_t32_remat_idx.so ]] || compile_one generic sv4_e256_b16_t32_remat_idx "$SUITE/kernels/sv4_index_gather_psum.py" 256 16 32 remat_idx
[[ -f $OUT/so/sv5_r64_c128_g16_t32_keep_reg.so ]] || compile_one generic sv5_r64_c128_g16_t32_keep_reg "$SUITE/kernels/sv5_reduce_small_eltwise.py" 64 128 16 32 keep_reg
[[ -f $OUT/so/sv5_r64_c128_g16_t32_ub_reload.so ]] || compile_one generic sv5_r64_c128_g16_t32_ub_reload "$SUITE/kernels/sv5_reduce_small_eltwise.py" 64 128 16 32 ub_reload
[[ -f $OUT/so/sv6_r64_c128_g32_t32.so ]] || compile_one generic sv6_r64_c128_g32_t32 "$SUITE/kernels/sv6_reduce_mid_eltwise.py" 64 128 32 32
[[ -f $OUT/so/sv7_r64_c128_g64_t32.so ]] || compile_one generic sv7_r64_c128_g64_t32 "$SUITE/kernels/sv7_reduce_large_eltwise.py" 64 128 64 32
[[ -f $OUT/so/sv7_r64_c128_g128_t32.so ]] || compile_one generic sv7_r64_c128_g128_t32 "$SUITE/kernels/sv7_reduce_large_eltwise.py" 64 128 128 32
# sv8 may have stale .so from prior naming — recompile if we want fresh; keep existing if present
[[ -f $OUT/so/sv8_r64_c128_g16_t32_live.so ]] || compile_one generic sv8_r64_c128_g16_t32_live "$SUITE/kernels/sv8_case3_bcast.py" 64 128 16 32 live
[[ -f $OUT/so/sv8_r64_c128_g16_t32_spill_dist.so ]] || compile_one generic sv8_r64_c128_g16_t32_spill_dist "$SUITE/kernels/sv8_case3_bcast.py" 64 128 16 32 spill_dist
[[ -f $OUT/so/sv9_e256_k1_t32_keep.so ]] || compile_one topk sv9_e256_k1_t32_keep 256 1 32 keep
[[ -f $OUT/so/sv9_e256_k8_t32_keep.so ]] || compile_one topk sv9_e256_k8_t32_keep 256 8 32 keep
[[ -f $OUT/so/sv9_e256_k8_t32_remat_scores.so ]] || compile_one topk sv9_e256_k8_t32_remat_scores 256 8 32 remat_scores
[[ -f $OUT/so/sv9_e256_k8_t32_remat_idx.so ]] || compile_one topk sv9_e256_k8_t32_remat_idx 256 8 32 remat_idx

# --- 5) build opsim tag list (need .so and no PASS yet) ---
OPSIM_TAGS=()
for t in "${REMAIN[@]}"; do
  if [[ -f "$OUT/so/${t}.so" ]] && ! grep -qE "^${t} PASS" "$SUMMARY"; then
    OPSIM_TAGS+=("$t")
  elif [[ -f "$OUT/so/${t}.so" ]] && grep -qE "^${t} PASS" "$SUMMARY"; then
    echo "SKIP_OPSIM_ALREADY_PASS $t"
  else
    echo "SKIP_OPSIM_NO_SO $t"
  fi
done
echo "OPSIM_PARALLEL_TAGS=${OPSIM_TAGS[*]:-}"

run_one_opsim() {
  local TAG=$1
  local RUNNER=/tmp/run_opsim_generic.py
  case "$TAG" in
    sv9_*) RUNNER=/tmp/run_opsim_topk.py ;;
  esac
  local TMP=$OUT/parallel_tmp/opsim_${TAG}.summary
  local OLOG=$OUT/opsim_${TAG}.log
  {
    echo "==== OPSIM $TAG start $(date -Is) ===="
    set +e
    "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${TAG}" \
      "$RUNNER" -- "$TAG" "$OUT" >"$OLOG" 2>&1
    local RC=$?
    set -e
    local PASSLINE US
    PASSLINE=$(grep -E '^PASS|^FAIL' "$OLOG" | tail -1 || true)
    US=$(grep -E 'core0\.veccore0' "$OLOG" | head -2 | awk '{print $2}' | head -1 || true)
    IPC=$(grep -Ei 'IPC' "$OLOG" | head -3 | tr '\n' ' ' || true)
    {
      flock 9
      echo "$TAG ${PASSLINE:-NO_PASSLINE} us=${US:-?} ipc=${IPC:-?} rc=$RC" >> "$SUMMARY"
    } 9>>"$SUMMARY.lock"
    echo "$TAG ${PASSLINE:-NO_PASSLINE} us=${US:-?} rc=$RC" > "$TMP"
    echo "==== OPSIM $TAG done $(date -Is) ${PASSLINE:-NO_PASSLINE} us=${US:-?} ===="
  }
}
export -f run_one_opsim
export SIM OUT SUMMARY

echo "==== PHASE PARALLEL OPSIM -P 6 ===="
if [[ ${#OPSIM_TAGS[@]} -gt 0 ]]; then
  printf '%s\n' "${OPSIM_TAGS[@]}" | xargs -P 6 -I{} bash -c 'run_one_opsim "$@"' _ {}
else
  echo "NO_OPSIM_NEEDED"
fi

echo "==== HARVEST ===="
"$PY" "$SUITE/harvest_report.py" 2>&1 | tee "$OUT/harvest_sv1_sv9.log" || true

echo "==== FINAL SUMMARY ===="
cat "$SUMMARY"
echo DONE_PARALLEL_SV1_SV9
