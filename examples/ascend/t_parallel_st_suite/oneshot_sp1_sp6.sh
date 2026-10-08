#!/usr/bin/env bash
# SimtVF Parallel SP1–SP6 single-axis redesign (2026-10-07).
# SP1 Pos KEEP/remat · SP2 Acc/Pos transitional · SP3 fp32 + e8m0 A5 bit-reinterpret
# (soft Pow2 LUT retired; no native e8m0 vcvt on A5) · SP4 pad · SP5 · SP6 soft LUT appendix.
# Prefer deps-native lib; target=ascend cython NOT pto. VMI twins later.
set -euo pipefail
OUT=/tmp/t_parallel_st_suite
LOG=$OUT/oneshot_sp1_sp6.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== t_parallel_st_suite SP1-SP6 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

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
  SUITE=/tmp/t_parallel_st_suite_suite
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

SUMMARY="$OUT/SUMMARY_sp1_sp6_raw.txt"
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
    grep -E 'Immutable|alloc_var|Error|Traceback|COMPILE_OK|RuntimeError|Unresolved|layout|InverseAffine|TypeError|reinterpret' "$OUT/logs/compile_${TAG}.log" | tail -60 | tee -a "$SUMMARY" || true
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

# Practical dims for first green: T=32,K=2,H=128,G=32,Thr=32
echo "==== PHASE SP1 dual scatter keep_pos / remat_pos ===="
compile_run_generic "$SUITE/kernels/sp1_dual_scatter_vsf.py" sp1_t32_k2_h128_g32_t32_keep_pos 32 2 128 32 32 keep_pos
compile_run_generic "$SUITE/kernels/sp1_dual_scatter_vsf.py" sp1_t32_k2_h128_g32_t32_remat_pos 32 2 128 32 32 remat_pos

echo "==== PHASE SP2 Acc KEEP/remat × sf0/sf1 tax (one case) ===="
# Acc KEEP + tax pair
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf0_w0_keep_acc 32 2 128 32 32 sf0 w0 keep_acc
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf1_w1_keep_acc 32 2 128 32 32 sf1 w1 keep_acc
# Acc remat + sf1_w1 (and sf0_w0 for completeness)
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf1_w1_remat_acc 32 2 128 32 32 sf1 w1 remat_acc
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf0_w0_remat_acc 32 2 128 32 32 sf0 w0 remat_acc
# Legacy Pos arms (Acc KEEP) — historical continuity
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf1_w1_keep_pos 32 2 128 32 32 sf1 w1 keep_pos
compile_run_generic "$SUITE/kernels/sp2_dual_gather_wreduce.py" sp2_t32_k2_h128_g32_t32_sf1_w1_remat_pos 32 2 128 32 32 sf1 w1 remat_pos

echo "==== PHASE SP3 fp32 vs A5 bit-reinterpret e8m0 (soft LUT retired; no native e8m0 vcvt) ===="
compile_run_generic "$SUITE/kernels/sp3_sf_pack_ue8m0.py" sp3_m32_h128_g32_t32_fp32 32 128 32 32 fp32
compile_run_generic "$SUITE/kernels/sp3_sf_pack_ue8m0.py" sp3_m32_h128_g32_t32_e8m0 32 128 32 32 e8m0

echo "==== PHASE SP4 pad gather ===="
compile_run_generic "$SUITE/kernels/sp4_pad_gather.py" sp4_e64_h128_t32_pad25 64 128 32 25

echo "==== PHASE SP5 sideband vs interleave ===="
compile_run_generic "$SUITE/kernels/sp5_sideband_vs_interleave.py" sp5_n64_h128_g32_qg32_t32_sideband 64 128 32 32 32 sideband
compile_run_generic "$SUITE/kernels/sp5_sideband_vs_interleave.py" sp5_n64_h128_g32_qg32_t32_interleave 64 128 32 32 32 interleave

echo "==== PHASE SP6 soft LUT 4-bit e2m1 dequant (appendix) ===="
compile_run_generic "$SUITE/kernels/sp6_fp4_unpack.py" sp6_n32_h128_g32_t32_unpack 32 128 32 32 unpack
compile_run_generic "$SUITE/kernels/sp6_fp4_unpack.py" sp6_n32_h128_g32_t32_unpack_sf 32 128 32 32 unpack_sf

echo "==== SUMMARY_sp1_sp6_raw ===="
cat "$SUMMARY"
echo DONE_SP1_SP6
