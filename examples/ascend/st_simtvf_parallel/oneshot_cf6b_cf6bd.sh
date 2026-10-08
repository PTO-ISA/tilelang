#!/usr/bin/env bash
# CF6b (Simt fat-hard/thin-easy) + CF6bd (PTO-DSL lockstep rooftop) on pto-b10.
# SOC Ascend950PR_9599. Uses .camodel_deps_vmi018 (NOT bare PTOAS-vmi/ptodsl).
# Tags: cf6b_e256_t32_phard{5,25} + cf6bd_e256_t32_phard{5,25}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel/cf6b_cf6bd}"
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/sources" "$OUT/reports"
PY="${PYTHON_BIN:-$HOME/projects/tilelang-deepseek/.venv-npu/bin/python}"
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3}"
PTOAS_ROOT="${PTOAS_ROOT:-$HOME/PTOAS-vmi}"
SIM_DSL="${SIM_DSL:-$PTOAS_ROOT/scripts/sim_dsl.sh}"
SOC="${SOC_VERSION:-Ascend950PR_9599}"
DEPS_STACK="${TILELANG_DEPS_STACK:-/home/happybot/projects/tilelang-pto-vmi-deps-stack}"
CAMO="${PTODSL_DEPS:-/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018}"
MLIR_PY="${MLIR_PYTHON_ROOT:-/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core}"
BK=/tmp/libtilelang.so.deps_backup_ab
if [[ ! -f "$BK" ]]; then BK=/tmp/libtilelang.so.deps_backup_st_simtvf; fi

export ST_SIMTVF_OUT="$OUT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TILELANG_DISABLE_CACHE=1
export TILELANG_DISABLE_DATA_RACE_CHECK=1
export PYTHON_BIN="$PY"
export ASCEND_HOME_PATH="$ASC"

set +u
[[ -f "$HOME/projects/env.sh" ]] && source "$HOME/projects/env.sh" || true
[[ -f "$ASC/set_env.sh" ]] && source "$ASC/set_env.sh" || true
set -u

echo "[cf6b-cf6bd] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT CAMO=$CAMO"
echo "[cf6b-cf6bd] $(date -Is) host=$(hostname)"

# --- helpers: harvest log-truth (PASS line + core0.veccore0), not SUMMARY rc ---
harvest_one() {
  local tag="$1" log="$2" odir="$3"
  local status="FAIL_OR_BLOCKED" maxabs="" wall=""
  # Simt run_opsim_generic: "PASS maxabs=..."; DSL: "PASS <tag> ... maxabs=..."
  if grep -qE "^PASS( |$)" "$log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$log" | head -1 | cut -d= -f2 || true)
  elif grep -qiE 'exceeded vf stack|COMPILE_FAIL|Error|Traceback|spill|AttributeError|Immutable' "$log" 2>/dev/null; then
    status="COMPILE_FAIL"
  elif grep -qiE "^FAIL " "$log" 2>/dev/null; then
    status="FAIL"
  fi
  wall=$(grep -E 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$log" 2>/dev/null \
    | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  if [[ -z "$wall" && -d "$odir" ]]; then
    wall=$(grep -RhoE 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$odir" 2>/dev/null \
      | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  fi
  echo -e "${tag}\t${status}\t${wall:-NA}\t${maxabs:-NA}" | tee -a "$SUMMARY"
  echo "RESULT $tag $status wall_us=${wall:-NA} maxabs=${maxabs:-NA}"
}

SUMMARY="$OUT/SUMMARY_harvest.tsv"
echo -e "tag\tstatus\twall_us\tmaxabs" > "$SUMMARY"

# ========== PHASE A: Simt CF6b (target=ascend + deps-native lib) ==========
echo "==== PHASE A: Simt CF6b ===="
if [[ ! -f "$BK" && -f "$DEPS_STACK/build/lib/libtilelang.so" ]]; then
  cp -a "$DEPS_STACK/build/lib/libtilelang.so" "$BK"
fi
if [[ -f "$BK" ]]; then
  cp -a "$BK" "$DEPS_STACK/build/lib/libtilelang.so"
  echo "DEPS_LIB=$(ls -la $DEPS_STACK/build/lib/libtilelang.so)"
  trap 'cp -a "$BK" "$DEPS_STACK/build/lib/libtilelang.so" 2>/dev/null; echo "restored deps lib"' EXIT
fi
export PATH="$CAMO/bin:$PATH"
export PYTHONPATH="$ROOT:$CAMO:$DEPS_STACK:$DEPS_STACK/build:${ASC}/python/site-packages:${PYTHONPATH:-}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# Ensure run_opsim_generic + ACL helper on /tmp
if [[ ! -f /tmp/run_cf_mb_opsim.py ]]; then
  for c in \
    /mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/examples/ascend/run_cf_mb_opsim.py \
    /home/happybot/projects/tilelang-deepseek/examples/ascend/run_cf_mb_opsim.py
  do
    [[ -f "$c" ]] && cp -f "$c" /tmp/run_cf_mb_opsim.py && break
  done
fi
cp -f "$ROOT/run_opsim_generic.py" /tmp/run_opsim_generic.py
sed -i 's/\r$//' "$ROOT"/*.py "$ROOT"/kernels/*.py /tmp/run_opsim_generic.py 2>/dev/null || true

compile_run_simt() {
  local SCRIPT=$1; shift
  local TAG=$1; shift
  echo "==== COMPILE $TAG ===="
  set +e
  "$PY" "$SCRIPT" "$@" 2>&1 | tee "$OUT/logs/compile_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -ne 0 || ! -f "$OUT/so/${TAG}.so" ]]; then
    echo "COMPILE_FAIL $TAG rc=$RC" | tee -a "$SUMMARY"
    grep -E 'Immutable|Error|Traceback|COMPILE_OK|RuntimeError|TypeError' "$OUT/logs/compile_${TAG}.log" | tail -40 || true
    return 0
  fi
  echo "COMPILE_OK $TAG"
  set +e
  "$SIM_DSL" --soc-version "$SOC" --output "$OUT/opsim_${TAG}" \
    /tmp/run_opsim_generic.py -- "$TAG" "$OUT" 2>&1 | tee "$OUT/${TAG}.log"
  set -e
  harvest_one "$TAG" "$OUT/${TAG}.log" "$OUT/opsim_${TAG}"
}

compile_run_simt "$ROOT/kernels/cf6b_fat_hard_newton.py" cf6b_e256_t32_phard5 256 32 5
compile_run_simt "$ROOT/kernels/cf6b_fat_hard_newton.py" cf6b_e256_t32_phard25 256 32 25

# ========== PHASE B: PTO-DSL CF6bd (camodel deps first) ==========
echo "==== PHASE B: PTO-DSL CF6bd ===="
# Prefer camodel deps; do NOT let bare PTOAS ptodsl win (missing ptoas.mlir)
export PYTHONPATH="${CAMO}:${MLIR_PY}:${PYTHONPATH:-}"
export PATH="${CAMO}/bin:${PATH}"

if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[cf6b-cf6bd] BLOCKED: cannot import ptodsl from camodel deps"
  echo -e "cf6bd_IMPORT\tBLOCKED\tNA\tNA" | tee -a "$SUMMARY"
else
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

  run_dsl() {
    local tag="$1"; shift
    local pyfile="$1"; shift
    local odir="$OUT/opsim_${tag}"
    mkdir -p "$odir"
    echo "==== opsim $tag ===="
    local rc=0
    if [[ -x "$SIM_DSL" ]]; then
      "$SIM_DSL" --soc-version "$SOC" --output "$odir" \
        "$ROOT/$pyfile" -- "$@" > "$OUT/${tag}.log" 2>&1 || rc=$?
    else
      (cd "$ROOT" && "$PY" "$pyfile" "$@") > "$OUT/${tag}.log" 2>&1 || rc=$?
    fi
    tail -40 "$OUT/${tag}.log" || true
    harvest_one "$tag" "$OUT/${tag}.log" "$odir"
  }

  emit_one "cf6bd_e256_t32_phard5" "kernels_ptodsl/cf6bd_fat_hard_newton.py" 256 32 5
  run_dsl "cf6bd_e256_t32_phard5" "kernels_ptodsl/cf6bd_fat_hard_newton.py" 256 32 5
  run_dsl "cf6bd_e256_t32_phard25" "kernels_ptodsl/cf6bd_fat_hard_newton.py" 256 32 25
fi

echo "[cf6b-cf6bd] done. SUMMARY_harvest:"
cat "$SUMMARY"
echo "[cf6b-cf6bd] Logs under $OUT"
echo DONE_CF6B_CF6BD
