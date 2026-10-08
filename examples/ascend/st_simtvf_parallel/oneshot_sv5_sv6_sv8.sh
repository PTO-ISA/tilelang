#!/usr/bin/env bash
# Case-3 SV5/SV6/SV8 — reduce→reduced-eltwise (+ SV8 bcast) on pto-b10
#
# Env knobs:
#   ST_SIMTVF_OUT  — OUT dir (default /tmp/st_simtvf_parallel). Point it at a
#                    private dir (e.g. /tmp/pr272_sv5_arms_YYYYMMDD) to avoid
#                    colliding with other agents' jobs.
#   PHASES         — comma list of phases to run: sv5,sv6,sv8 (default all).
#                    e.g. PHASES=sv5 for the SV5 arm sensitivity sweep only.
#   DEPS_PRIVATE=1 — do NOT swap the shared deps-stack libtilelang.so. Instead
#                    hardlink-copy the deps stack to $OUT/deps and drop the
#                    deps-native lib in there. Use this whenever another agent
#                    may be holding the overlay lib (VMI jobs) on pto-b10.
set -euo pipefail
OUT=${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel}
PHASES=${PHASES:-sv5,sv6,sv8}
has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }
LOG=$OUT/oneshot_sv5_sv6_sv8.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== st_simtvf_parallel Case-3 SV5/SV6/SV8 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

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
# Keep the opsim driver inside $OUT when OUT is a private job dir, so we never
# overwrite /tmp/run_opsim_generic.py underneath another agent's job.
cp -f "$SUITE"/run_opsim_*.py "$OUT"/ 2>/dev/null || true
cp -f "$SUITE"/harvest_report.py "$OUT"/ 2>/dev/null || true
if [[ "$OUT" == "/tmp/st_simtvf_parallel" ]]; then
  cp -f "$SUITE"/run_opsim_*.py /tmp/ 2>/dev/null || true
  cp -f "$SUITE"/harvest_report.py /tmp/ 2>/dev/null || true
fi
RUNPY="$OUT/run_opsim_generic.py"
[[ -f "$RUNPY" ]] || RUNPY=/tmp/run_opsim_generic.py
sed -i 's/\r$//' "$SUITE"/*.py "$SUITE"/kernels/*.py "$OUT"/run_opsim_*.py 2>/dev/null || true

if [[ ! -f "$BK" ]]; then cp -a "$DEPS/build/lib/libtilelang.so" "$BK"; fi
if [[ "${DEPS_PRIVATE:-0}" == "1" ]]; then
  # Private hardlink copy of the deps stack — shared libtilelang.so untouched,
  # so a concurrent VMI/overlay job on pto-b10 is never clobbered.
  PRIV="$OUT/deps"
  if [[ ! -d "$PRIV" ]]; then cp -al "$DEPS" "$PRIV"; fi
  rm -f "$PRIV/build/lib/libtilelang.so"   # break the hardlink BEFORE writing
  cp -a "$BK" "$PRIV/build/lib/libtilelang.so"
  DEPS="$PRIV"
  echo "DEPS_PRIVATE=$DEPS"
  echo "DEPS_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
else
  cp -a "$BK" "$DEPS/build/lib/libtilelang.so"
  echo "DEPS_LIB=$(ls -la $DEPS/build/lib/libtilelang.so)"
  trap 'cp -a "$BK" "$DEPS/build/lib/libtilelang.so"; echo "restored deps lib"' EXIT
fi

set +u; source "$ASC/set_env.sh"; set -u
export ASCEND_HOME_PATH=$ASC
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$SUITE:$CAMO:$DEPS:$DEPS/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TILELANG_DISABLE_DATA_RACE_CHECK=1

SUMMARY="$OUT/SUMMARY_sv5_sv6_sv8_raw.txt"
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
    "$RUNPY" -- "$TAG" "$OUT" 2>&1 | tee "$OUT/opsim_${TAG}.log"
  set -e
  local PASSLINE US
  PASSLINE=$(grep -E '^PASS|^FAIL' "$OUT/opsim_${TAG}.log" | tail -1 || true)
  US=$(grep -E 'core0\.veccore0|duration_time|Total cycles|IPC|membar|MTE' "$OUT/opsim_${TAG}.log" | head -12 | tr '\n' ' | ' || true)
  echo "$TAG $PASSLINE us=$US" | tee -a "$SUMMARY"
}

if has_phase sv5; then
echo "==== PHASE SV5 (small-block reduce → reduced eltwise) ===="
# Post-reduce inverse-scale residency sensitivity: keep_reg (live fragment/RF)
# vs ub_reload (store to shared UB, reload in a later Parallel loop).
# Shape matrix hunts the crossover: R sweep at G=16, then G=8 / G=32 at R=64.
SV5=$SUITE/kernels/sv5_reduce_small_eltwise.py
# shape-major so each keep_reg/ub_reload pair completes together
SV5_SHAPES=${SV5_SHAPES:-"64:128:16 128:128:16 256:128:16 64:128:8 64:128:32"}
for sh in $SV5_SHAPES; do
  IFS=: read -r SR SC SG <<< "$sh"
  for arm in keep_reg ub_reload; do
    compile_run_generic "$SV5" "sv5_r${SR}_c${SC}_g${SG}_t32_${arm}" "$SR" "$SC" "$SG" 32 "$arm"
  done
done
# legacy pre-arm baseline (reduce + plain frag→UB copy, no subsequent compute)
if [[ "${SV5_LEGACY:-0}" == "1" ]]; then
  compile_run_generic "$SV5" sv5_r64_c128_g16_t32 64 128 16 32 keep_reg
fi
fi

if has_phase sv6; then
echo "==== PHASE SV6 (large-block reduce → reduced eltwise) ===="
compile_run_generic "$SUITE/kernels/sv6_reduce_large_eltwise.py" sv6_r64_c128_g64_t32 64 128 64 32
# optional CG=1 stress — may fail layout; non-fatal
compile_run_generic "$SUITE/kernels/sv6_reduce_large_eltwise.py" sv6_r64_c128_g128_t32 64 128 128 32
fi

if has_phase sv8; then
echo "==== PHASE SV8 Case-3 (live + spill_dist) ===="
compile_run_generic "$SUITE/kernels/sv8_case3_bcast.py" sv8_r64_c128_g16_t32_live 64 128 16 32 live
compile_run_generic "$SUITE/kernels/sv8_case3_bcast.py" sv8_r64_c128_g16_t32_spill_dist 64 128 16 32 spill_dist
fi

ST_SIMTVF_OUT="$OUT" ST_SIMTVF_REPORTS="$OUT/reports" "$PY" "$SUITE/harvest_report.py" \
  2>&1 | tee "$OUT/harvest_sv5_sv6_sv8.log" || true
if [[ -f "$SUITE/harvest_sv5_arms.py" ]]; then
  "$PY" "$SUITE/harvest_sv5_arms.py" "$OUT" 2>&1 | tee -a "$OUT/harvest_sv5_sv6_sv8.log" || true
fi

echo "==== SUMMARY_sv5_sv6_sv8_raw ===="
cat "$SUMMARY"
echo DONE_SV5_SV6_SV8
