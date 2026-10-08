#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "$HERE/../../.." && pwd)
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel_pto_isa}"
export ST_SIMTVF_OUT="$OUT" TILELANG_DEPS="$DST" TILELANG_ROOT="$DST"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TILELANG_DISABLE_CACHE=1 TILELANG_DISABLE_DATA_RACE_CHECK=1
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
CAMO=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
export ASCEND_HOME_PATH="$ASC" PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$HERE:$DST:$DST/build:$CAMO:${ASC}/python/site-packages"
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
SUMMARY="$OUT/SUMMARY_simt_compilecheck_raw.txt"
: > "$SUMMARY"
LOG="$OUT/oneshot_simt_compilecheck.log"
exec > >(tee -a "$LOG") 2>&1
set +u; source "$ASC/bin/setenv.bash"; set -u
echo "=== PTO-ISA Simt compilecheck $(date -Is) DST=$DST ==="
ls -la "$DST/build/lib/libtilelang.so"

run() {
  local script="$1"; shift
  local tag="$1"; shift
  echo "==== COMPILE $tag ====" | tee -a "$SUMMARY"
  set +e
  "$PY" "$HERE/$script" "$@" 2>&1 | tee "$OUT/logs/compile_${tag}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 || ! -f "$OUT/so/${tag}.so" ]]; then
    echo "COMPILE_FAIL $tag rc=$rc" | tee -a "$SUMMARY"
    grep -E "undeclared identifier|unknown type name|Compilation Failed|COMPILE_OK|SimtVF FFI" "$OUT/logs/compile_${tag}.log" | head -20 | tee -a "$SUMMARY" || true
  else
    echo "COMPILE_OK $tag" | tee -a "$SUMMARY"
  fi
}

# Representative primary-scope compile probes (full matrix would repeat same CANN wall)
run kernels/sv1_stream_eltwise.py sv1_e256_t32 256 32
run kernels/sv2_eltwise_bcast_rf.py sv2_r32_c64_t32_frag_live 32 64 32 frag_live
run kernels/sv3_gemv_partial_keep.py sv3_m32_vl64_k16_t32_keep 32 64 16 32 keep
run kernels/cf1_pred_thresh_keep.py cf1_e256_k8_t32_keep 256 8 32
run kernels/cf6_newton_branch.py cf6_e256_t32_pnear5 256 32 5
run kernels/sp1_dual_scatter_vsf.py sp1_t32_k2_h128_g32_t32_keep_pos 32 2 128 32 32 keep_pos
run kernels/sp2_dual_gather_wreduce.py sp2_t32_k2_h128_g32_t32_sf0_w0_keep_acc 32 2 128 32 32 sf0 w0 keep_acc
run kernels/sp3_sf_pack_ue8m0.py sp3_m32_h128_g32_t32_e8m0 32 128 32 32 e8m0
run kernels/sp6_fp4_unpack.py sp6_n32_h128_g32_t32_unpack 32 128 32 32 unpack

echo "=== DONE $(date -Is) ===" | tee -a "$SUMMARY"
