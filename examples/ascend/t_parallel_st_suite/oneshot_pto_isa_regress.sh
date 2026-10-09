#!/usr/bin/env bash
# Full primary-scope regress on PTO-ISA/tilelang pto-dev checkout.
# Simt SV/CF/SP + SIMD (PTO-DSL) SV/CF/SP (21 + 21, including SP4d/SP5d).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "$HERE/../../.." && pwd)

# Required paths are environment variables. See README.md.
if [[ -z "${PY:-}" && -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
fi
: "${PY:?Set PY to the NPU venv interpreter (PYTHON_BIN is also accepted)}"
: "${ASCEND_HOME_PATH:?Set ASCEND_HOME_PATH to the CANN toolkit root (directory containing set_env.sh)}"
: "${CAMODEL_DEPS:?Set CAMODEL_DEPS to the camodel dependency prefix (its bin/ is prepended to PATH)}"
if [[ -z "${SIM_DSL:-}" && -n "${PTOAS_ROOT:-}" ]]; then
  SIM_DSL="$PTOAS_ROOT/scripts/sim_dsl.sh"
fi
: "${SIM_DSL:?Set SIM_DSL to sim_dsl.sh, or set PTOAS_ROOT to use \$PTOAS_ROOT/scripts/sim_dsl.sh}"
: "${PTOAS_ROOT:?Set PTOAS_ROOT to the PTOAS checkout (ptodsl/ and scripts/sim_dsl.sh)}"
: "${PTODSL_DEPS:?Set PTODSL_DEPS to the PTO-DSL / camodel dependency prefix}"
: "${MLIR_PYTHON_ROOT:?Set MLIR_PYTHON_ROOT to the MLIR Python package root}"
export PY SIM_DSL

export TILELANG_DEPS="$DST"
export ST_SIMTVF_OUT="${ST_SIMTVF_OUT:-/tmp/t_parallel_st_suite_pto_isa}"
export PYTHONPATH="$HERE:$DST:$DST/build:${PYTHONPATH:-}"
export TILELANG_ROOT="$DST"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p "$ST_SIMTVF_OUT"
LOG="$ST_SIMTVF_OUT/oneshot_pto_isa_regress.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== PTO-ISA ST regress $(date -Is) DST=$DST OUT=$ST_SIMTVF_OUT PY=$PY ASCEND_HOME_PATH=$ASCEND_HOME_PATH ==="
ls -la "$DST/build/lib/libtilelang.so" 2>/dev/null || echo "(no in-tree libtilelang.so yet)"
"$PY" -c "import tilelang,tilelang.libinfo as li; print(tilelang.__path__); print(li.find_lib_path('tilelang'))" || true

run() {
  local name="$1"
  echo "==== BEGIN $name $(date -Is) ===="
  set +e
  bash "$HERE/$name"
  local rc=$?
  set -e
  echo "==== END $name rc=$rc $(date -Is) ===="
}

run oneshot_sv1_sv9.sh
run oneshot_cf1_cf6.sh
run oneshot_sp1_sp6.sh
run oneshot_ptodsl_sv1_sv9d.sh
run oneshot_ptodsl_cf1d_cf6d.sh
run oneshot_ptodsl_sp1d_sp6d.sh

echo "=== PTO-ISA ST regress DONE $(date -Is) ==="
