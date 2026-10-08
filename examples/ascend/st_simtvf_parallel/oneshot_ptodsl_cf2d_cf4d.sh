#!/usr/bin/env bash
# PTO-DSL Layer D — CF2d / CF3d / CF4d (remat / remat_idx / nested vsel).
# Run on pto-b10 login node (Ascend950PR_9599 opsim). NO board .39.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_cf2d_cf4d}"
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

echo "[ptodsl-cf2d-cf4d] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-cf2d-cf4d] BLOCKED: cannot import ptodsl"
  exit 2
fi

SUMMARY="$OUT/SUMMARY.tsv"
echo -e "tag\tstatus\twall_us\tmaxabs\tnote" > "$SUMMARY"

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
  tail -30 "$OUT/${tag}.log" || true

  local status="FAIL_OR_BLOCKED"
  local maxabs=""
  local wall=""
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$OUT/${tag}.log" | head -1 | cut -d= -f2 || true)
  elif grep -qiE 'exceeded vf stack|COMPILE|Error|Traceback|spill|AttributeError' "$OUT/${tag}.log" 2>/dev/null; then
    status="COMPILE_FAIL"
  fi
  wall=$(awk '/core0.veccore0[[:space:]]+[0-9]/ {print $2; exit}' "$OUT/${tag}.log" 2>/dev/null || true)
  if [[ -z "$wall" ]]; then
    wall=$(grep -RhoE 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$odir" 2>/dev/null \
      | head -1 | awk '{print $2}' || true)
  fi
  echo -e "${tag}\t${status}\t${wall:-NA}\t${maxabs:-NA}\trc=${rc}" | tee -a "$SUMMARY"
  echo "RESULT $tag $status wall_us=${wall:-NA}"
}

emit_one() {
  local tag="$1"; shift
  local pyfile="$1"; shift
  echo "==== emit-mlir $tag ===="
  (cd "$ROOT" && "$PY" "$pyfile" "$@" --emit-mlir) > "$OUT/${tag}.mlir" 2>"$OUT/${tag}_emit.err" || {
    echo "EMIT_FAIL $tag (see $OUT/${tag}_emit.err)"
    return 0
  }
  local nlines
  nlines=$(wc -l < "$OUT/${tag}.mlir" | tr -d ' ')
  echo "EMIT_OK $tag lines=$nlines -> $OUT/${tag}.mlir"
}

emit_one "cf2d_e256_k8_t32_remat" "kernels_ptodsl/cf2d_remat_thresh_kill_shared.py" 256 8 32
emit_one "cf3d_e256_k8_t32_remat_idx" "kernels_ptodsl/cf3d_remat_idx.py" 256 8 32
emit_one "cf4d_e256_t32_pfat5" "kernels_ptodsl/cf4d_nested_if.py" 256 32 5

run_one "cf2d_e256_k8_t32_remat" "kernels_ptodsl/cf2d_remat_thresh_kill_shared.py" 256 8 32
run_one "cf3d_e256_k8_t32_remat_idx" "kernels_ptodsl/cf3d_remat_idx.py" 256 8 32
run_one "cf4d_e256_t32_pfat5" "kernels_ptodsl/cf4d_nested_if.py" 256 32 5
run_one "cf4d_e256_t32_pfat25" "kernels_ptodsl/cf4d_nested_if.py" 256 32 25

echo "[ptodsl-cf2d-cf4d] done. SUMMARY:"
cat "$SUMMARY"
echo "[ptodsl-cf2d-cf4d] Logs under $OUT"
