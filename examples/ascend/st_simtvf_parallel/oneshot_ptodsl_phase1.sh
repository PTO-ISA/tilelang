#!/usr/bin/env bash
# Phase-1 PTO-DSL twins: SV1d / CF1d / SP1d (explicit vpto).
# Run on pto-b10 login node after a working ptodsl+ptoas.mlir env is sourced.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_phase1}"
mkdir -p "$OUT"
PY="${PYTHON_BIN:-$HOME/projects/tilelang-deepseek/.venv-npu/bin/python}"
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3}"
PTOAS_ROOT="${PTOAS_ROOT:-$HOME/PTOAS-vmi}"
SIM_DSL="${SIM_DSL:-$PTOAS_ROOT/scripts/sim_dsl.sh}"
SOC="${SOC_VERSION:-Ascend950PR_9599}"

set +u
# Prefer project env if present
if [[ -f "$HOME/projects/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/projects/env.sh" || true
fi
if [[ -f "$ASC/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ASC/set_env.sh"
fi
set -u
export ASCEND_HOME_PATH="$ASC"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHON_BIN="$PY"
# Put ptodsl ahead; caller may override PYTHONPATH for a green ptoas.mlir
DEPS="${PTODSL_DEPS:-/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018}"
MLIR_PY="${MLIR_PYTHON_ROOT:-/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core}"
export PYTHONPATH="${DEPS}:${PTOAS_ROOT}/ptodsl:${MLIR_PY}:${PYTHONPATH:-}"
export PATH="${DEPS}/bin:${PATH}"

echo "[ptodsl-phase1] OUT=$OUT PY=$PY PTOAS_ROOT=$PTOAS_ROOT"
echo "[ptodsl-phase1] import check..."
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-phase1] BLOCKED: cannot import ptodsl.pto (ptoas.mlir missing/broken)."
  echo "[ptodsl-phase1] Sources still at $ROOT/kernels_ptodsl — fix env then re-run."
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
  if grep -q "^PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null || grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    echo "RESULT $tag PASS"
  else
    echo "RESULT $tag FAIL_OR_BLOCKED (see $OUT/${tag}.log)"
  fi
}

run_one "sv1d_e256_t32" "kernels_ptodsl/sv1d_stream_eltwise.py" 256 32
run_one "cf1d_e256_k8_t32_keep" "kernels_ptodsl/cf1d_pred_thresh_keep.py" 256 8 32
run_one "sp1d_t32_k2_h128_g32_t32_keep_pos" "kernels_ptodsl/sp1d_dual_scatter_keep_pos.py" 32 2 128 32 32

echo "[ptodsl-phase1] done. Logs under $OUT"
