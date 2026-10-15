#!/usr/bin/env bash
# SimtVF Parallel SP1–SP6 single-axis redesign (2026-10-07).
# SP1 Pos KEEP/remat · SP2 Acc/Pos transitional · SP3 fp32 + e8m0 A5 bit-reinterpret
# (soft Pow2 LUT retired; no native e8m0 vcvt on A5) · SP4 pad · SP5 · SP6 soft LUT appendix.
# Compile target follows ST_VF_MODE (default simt-asc → target=ascend, cython).
# simt-pto / simd-pto compile target=pto. kernels_ptodsl/ is separate.
set -euo pipefail
OUT="${ST_SIMTVF_OUT:-/tmp/t_parallel_st_suite}"
LOG=$OUT/oneshot_sp1_sp6.log
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
exec > >(tee -a "$LOG") 2>&1
echo "=== t_parallel_st_suite SP1-SP6 oneshot $(date -Is) host=$(hostname) user=$(whoami) ==="

# Required paths are environment variables. See README.md.
if [[ -z "${PY:-}" && -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
fi
: "${PY:?Set PY to the NPU venv interpreter (PYTHON_BIN is also accepted)}"
: "${ASCEND_HOME_PATH:?Set ASCEND_HOME_PATH to the CANN toolkit root (directory containing set_env.sh)}"
: "${TILELANG_DEPS:?Set TILELANG_DEPS to the TileLang tree that contains build/lib/libtilelang.so}"
: "${CAMODEL_DEPS:?Set CAMODEL_DEPS to the camodel dependency prefix (its bin/ is prepended to PATH)}"
if [[ -z "${SIM_DSL:-}" && -n "${PTOAS_ROOT:-}" ]]; then
  SIM_DSL="$PTOAS_ROOT/scripts/sim_dsl.sh"
fi
: "${SIM_DSL:?Set SIM_DSL to sim_dsl.sh, or set PTOAS_ROOT to use \$PTOAS_ROOT/scripts/sim_dsl.sh}"
DEPS=$TILELANG_DEPS
ASC=$ASCEND_HOME_PATH
CAMO=$CAMODEL_DEPS
SIM=$SIM_DSL
SOC="${SOC:-${SOC_VERSION:-Ascend950PR_9599}}"
export PY SIM_DSL SOC
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
  _opsim_src=""
  for c in \
    "${RUN_CF_MB_OPSIM:-}" \
    "$DEPS/examples/ascend/run_cf_mb_opsim.py" \
    "$HERE/run_cf_mb_opsim.py"
  do
    if [[ -n "$c" && -f "$c" ]]; then
      _opsim_src="$c"
      break
    fi
  done
  if [[ -z "$_opsim_src" ]]; then
    echo "error: run_cf_mb_opsim.py not found. Set RUN_CF_MB_OPSIM or place it at \$TILELANG_DEPS/examples/ascend/run_cf_mb_opsim.py" >&2
    exit 1
  fi
  cp -f "$_opsim_src" /tmp/run_cf_mb_opsim.py
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
echo "PY=$PY ASCEND_HOME_PATH=$ASC TILELANG_DEPS=$DEPS CAMODEL_DEPS=$CAMO SIM_DSL=$SIM SOC=$SOC ST_VF_MODE=${ST_VF_MODE:-simt-asc}"

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
  "$SIM" --soc-version "$SOC" --output "$OUT/opsim_${TAG}" \
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
