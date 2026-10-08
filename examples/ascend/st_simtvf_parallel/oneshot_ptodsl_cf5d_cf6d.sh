#!/usr/bin/env bash
# PTO-DSL Layer D — CF5d / CF6d (vector Div/Newton in RF; no CF5v/CF6v ABI tax).
# Run on pto-b10 login node (Ascend950PR_9599 opsim). NO board .39.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/ptodsl_cf5d_cf6d}"
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

echo "[ptodsl-cf5d-cf6d] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-cf5d-cf6d] BLOCKED: cannot import ptodsl"
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
  # Prefer msprof table "core0.veccore0  <us>" (avoid matching tiny unrelated floats)
  wall=$(grep -E 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$OUT/${tag}.log" 2>/dev/null \
    | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  if [[ -z "$wall" ]]; then
    wall=$(grep -RhoE 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$odir" 2>/dev/null \
      | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
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

# emit-mlir first (cheap compile check)
emit_one "cf5d_e256_t32_pnear5" "kernels_ptodsl/cf5d_div_ulp_branch.py" 256 32 5
emit_one "cf6d_e256_t32_pnear5" "kernels_ptodsl/cf6d_newton_branch.py" 256 32 5

# opsim primary + pnear25 sensitivity
run_one "cf5d_e256_t32_pnear5" "kernels_ptodsl/cf5d_div_ulp_branch.py" 256 32 5
run_one "cf5d_e256_t32_pnear25" "kernels_ptodsl/cf5d_div_ulp_branch.py" 256 32 25
run_one "cf6d_e256_t32_pnear5" "kernels_ptodsl/cf6d_newton_branch.py" 256 32 5
run_one "cf6d_e256_t32_pnear25" "kernels_ptodsl/cf6d_newton_branch.py" 256 32 25

echo "[ptodsl-cf5d-cf6d] done. SUMMARY:"
cat "$SUMMARY"
echo "[ptodsl-cf5d-cf6d] Logs under $OUT"
