#!/usr/bin/env bash
# PTO-DSL Layer D — full SV1d–SV9d VMI rooftop / sensitivity suite.
# Run on pto-b10 login node (Ascend950PR_9599 opsim). NO board .39.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d}"
mkdir -p "$OUT"
PY="${PYTHON_BIN:-$HOME/projects/tilelang-deepseek/.venv-npu/bin/python}"
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3}"
PTOAS_ROOT="${PTOAS_ROOT:-$HOME/PTOAS-vmi}"
SIM_DSL="${SIM_DSL:-$PTOAS_ROOT/scripts/sim_dsl.sh}"
SOC="${SOC_VERSION:-Ascend950PR_9599}"

set +u
[[ -f "$HOME/projects/env.sh" ]] && source "$HOME/projects/env.sh" || true
[[ -f "$ASC/set_env.sh" ]] && source "$ASC/set_env.sh" || true
set -u
export ASCEND_HOME_PATH="$ASC"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHON_BIN="$PY"
DEPS="${PTODSL_DEPS:-/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018}"
MLIR_PY="${MLIR_PYTHON_ROOT:-/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core}"
export PYTHONPATH="${DEPS}:${PTOAS_ROOT}/ptodsl:${MLIR_PY}:${PYTHONPATH:-}"
export PATH="${DEPS}/bin:${PATH}"

echo "[ptodsl-sv1-9d] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-sv1-9d] BLOCKED: cannot import ptodsl"
  exit 2
fi

SUMMARY="$OUT/SUMMARY.tsv"
echo -e "tag\tstatus\twall_us\tmaxabs\tnote" > "$SUMMARY"

extract_wall() {
  local log="$1"
  # Prefer msprof core0.veccore0 duration_time(us)
  local us
  us=$(grep -E 'core0\.veccore0|duration_time' "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  if [[ -z "$us" ]]; then
    us=$(grep -oE 'duration_time\(us\)[=: ]+[0-9.]+' "$log" 2>/dev/null | grep -oE '[0-9.]+' | head -1 || true)
  fi
  # Also search opsim output dirs
  if [[ -z "$us" ]]; then
    us=$(grep -RhoE 'duration_time[^0-9]*([0-9]+\.[0-9]+)' "$OUT" 2>/dev/null | head -1 | grep -oE '[0-9.]+$' || true)
  fi
  echo "${us:-}"
}

run_one() {
  local tag="$1"; shift
  local pyfile="$1"; shift
  local odir="$OUT/opsim_${tag}"
  mkdir -p "$odir"
  echo "==== $tag ===="
  local rc=0
  if [[ -x "$SIM_DSL" ]]; then
    "$SIM_DSL" --soc-version "$SOC" --output "$odir" \
      "$ROOT/$pyfile" -- "$@" > "$OUT/${tag}.log" 2>&1 || rc=$?
  else
    (cd "$ROOT" && "$PY" "$pyfile" "$@") > "$OUT/${tag}.log" 2>&1 || rc=$?
  fi
  # Also capture tee to stdout briefly
  tail -20 "$OUT/${tag}.log" || true

  local status="FAIL_OR_BLOCKED"
  local maxabs=""
  local wall=""
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$OUT/${tag}.log" | head -1 | cut -d= -f2 || true)
  elif grep -qiE 'exceeded vf stack|COMPILE|Error|Traceback|spill' "$OUT/${tag}.log" 2>/dev/null; then
    status="COMPILE_FAIL"
  fi
  # wall from opsim dir
  wall=$(grep -RhoE 'duration_time\(us\)[,=: ]+[0-9.]+' "$odir" "$OUT/${tag}.log" 2>/dev/null \
    | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  if [[ -z "$wall" ]]; then
    # alternate msprof table format
    wall=$(grep -R 'core0.veccore0' -A2 "$odir" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  fi
  echo -e "${tag}\t${status}\t${wall:-NA}\t${maxabs:-NA}\trc=${rc}" | tee -a "$SUMMARY"
  echo "RESULT $tag $status wall_us=${wall:-NA}"
}

# ---- SV1d ----
run_one "sv1d_e256_t32" "kernels_ptodsl/sv1d_stream_eltwise.py" 256 32

# ---- SV2d primary sensitivity ----
run_one "sv2d_r16_c64_t32_input_keep" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 16 64 32 input_keep
run_one "sv2d_r32_c64_t32_input_stream" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 64 32 input_stream
run_one "sv2d_r32_c128_t32_input_stream" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 128 32 input_stream
run_one "sv2d_r32_c64_t32_fold_scale_keep" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 64 32 fold_scale_keep
run_one "sv2d_r32_c64_t32_fold_scale_reload" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 64 32 fold_scale_reload

# ---- SV3d NEW ----
run_one "sv3d_m24_vl64_k16_t32_keep" "kernels_ptodsl/sv3d_gemv_partial_keep.py" 24 64 16 32 keep
run_one "sv3d_m32_vl64_k16_t32_keep" "kernels_ptodsl/sv3d_gemv_partial_keep.py" 32 64 16 32 keep
run_one "sv3d_m32_vl64_k16_t32_split_cm16" "kernels_ptodsl/sv3d_gemv_partial_keep.py" 32 64 16 32 split_cm16 16

# ---- SV4d ----
run_one "sv4d_e256_b8_t32_keep_idx" "kernels_ptodsl/sv4d_index_gather_psum.py" 256 8 32 keep_idx
run_one "sv4d_e256_b8_t32_remat_idx" "kernels_ptodsl/sv4d_index_gather_psum.py" 256 8 32 remat_idx

# ---- SV5d NEW RF ladder ----
run_one "sv5d_r64_c128_g16_t32_keep_in_rf" "kernels_ptodsl/sv5d_reduce_small.py" 64 128 16 32 keep_in_rf
run_one "sv5d_r64_c128_g16_t32_ub_stream" "kernels_ptodsl/sv5d_reduce_small.py" 64 128 16 32 ub_stream

# ---- SV6d NEW ----
run_one "sv6d_r64_c128_g32_t32_keep_in_warp" "kernels_ptodsl/sv6d_reduce_mid.py" 64 128 32 32 keep_in_warp
run_one "sv6d_r64_c128_g32_t32_reload" "kernels_ptodsl/sv6d_reduce_mid.py" 64 128 32 32 reload

# ---- SV7d NEW ----
run_one "sv7d_r64_c128_g64_t32_multiwarp_ub" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 64 32 multiwarp_ub
run_one "sv7d_r64_c128_g64_t32_reload" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 64 32 reload
run_one "sv7d_r64_c128_g128_t32_multiwarp_ub" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 128 32 multiwarp_ub
run_one "sv7d_r64_c128_g128_t32_reload" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 128 32 reload

# ---- SV8d NEW ----
run_one "sv8d_r64_c128_g16_t32_live" "kernels_ptodsl/sv8d_case3_bcast.py" 64 128 16 32 live
run_one "sv8d_r64_c128_g16_t32_spill_dist" "kernels_ptodsl/sv8d_case3_bcast.py" 64 128 16 32 spill_dist

# ---- SV9d ----
run_one "sv9d_e256_k8_t32_keep" "kernels_ptodsl/sv9d_topk_e2e.py" 256 8 32 keep
run_one "sv9d_e256_k8_t32_remat_scores" "kernels_ptodsl/sv9d_topk_e2e.py" 256 8 32 remat_scores
run_one "sv9d_e256_k8_t32_remat_idx" "kernels_ptodsl/sv9d_topk_e2e.py" 256 8 32 remat_idx

echo "[ptodsl-sv1-9d] done. SUMMARY:"
cat "$SUMMARY"
echo "[ptodsl-sv1-9d] Logs under $OUT"
