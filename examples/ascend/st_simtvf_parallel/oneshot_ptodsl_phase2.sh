#!/usr/bin/env bash
# Phase-2 PTO-DSL RF ladder: SV2d / SV4d / SV9d (explicit vpto).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_phase2}"
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

echo "[ptodsl-phase2] OUT=$OUT PY=$PY"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-phase2] BLOCKED: cannot import ptodsl"
  exit 2
fi

run_one() {
  local tag="$1"; shift
  local pyfile="$1"; shift
  local odir="$OUT/opsim_${tag}"
  mkdir -p "$odir"
  echo "==== $tag ===="
  if [[ -x "$SIM_DSL" ]]; then
    "$SIM_DSL" --soc-version "$SOC" --output "$odir" \
      "$ROOT/$pyfile" -- "$@" 2>&1 | tee "$OUT/${tag}.log" || true
  else
    (cd "$ROOT" && "$PY" "$pyfile" "$@") 2>&1 | tee "$OUT/${tag}.log" || true
  fi
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    echo "RESULT $tag PASS"
    grep -E 'core0.veccore0|duration_time' "$OUT/${tag}.log" | head -5 || true
  else
    echo "RESULT $tag FAIL_OR_BLOCKED (see $OUT/${tag}.log)"
    tail -30 "$OUT/${tag}.log" || true
  fi
}

# SV2d: scale ALWAYS KEEP; input KEEP (R16) vs STREAM (R32/64). fold_scale_reload demoted.
run_one "sv2d_r16_c64_t32_input_keep" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 16 64 32 input_keep
run_one "sv2d_r32_c64_t32_input_stream" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 64 32 input_stream
run_one "sv2d_r64_c64_t32_input_stream" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 64 64 32 input_stream
run_one "sv2d_r32_c128_t32_input_stream" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 128 32 input_stream
# Optional: input_keep at R32 expected spill (cliff foil)
run_one "sv2d_r32_c64_t32_input_keep" "kernels_ptodsl/sv2d_eltwise_bcast_rf.py" 32 64 32 input_keep
run_one "sv4d_e256_b8_t32_keep_idx" "kernels_ptodsl/sv4d_index_gather_psum.py" 256 8 32 keep_idx
run_one "sv4d_e256_b8_t32_remat_idx" "kernels_ptodsl/sv4d_index_gather_psum.py" 256 8 32 remat_idx
run_one "sv9d_e256_k8_t32_keep" "kernels_ptodsl/sv9d_topk_e2e.py" 256 8 32 keep
run_one "sv9d_e256_k8_t32_remat_scores" "kernels_ptodsl/sv9d_topk_e2e.py" 256 8 32 remat_scores
run_one "sv9d_e256_k8_t32_remat_idx" "kernels_ptodsl/sv9d_topk_e2e.py" 256 8 32 remat_idx

echo "[ptodsl-phase2] done. Logs under $OUT"
