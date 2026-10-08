#!/usr/bin/env bash
# Full primary-scope regress on PTO-ISA/tilelang pto-dev checkout.
# Simt SV/CF/SP + SIMD (PTO-DSL) SV/CF/SP (SP4d/SP5d absent).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$(cd "$HERE/../../.." && pwd)
export TILELANG_DEPS="$DST"
export ST_SIMTVF_OUT="${ST_SIMTVF_OUT:-/tmp/st_simtvf_parallel_pto_isa}"
export PYTHONPATH="$HERE:$DST:$DST/build:${PYTHONPATH:-}"
export TILELANG_ROOT="$DST"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p "$ST_SIMTVF_OUT"
LOG="$ST_SIMTVF_OUT/oneshot_pto_isa_regress.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== PTO-ISA ST regress $(date -Is) DST=$DST OUT=$ST_SIMTVF_OUT ==="
ls -la "$DST/build/lib/libtilelang.so" 2>/dev/null || echo "(no in-tree libtilelang.so yet)"
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
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
