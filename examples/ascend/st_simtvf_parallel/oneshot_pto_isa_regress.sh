#!/usr/bin/env bash
# Full primary-scope regress on PTO-ISA/tilelang pto-dev checkout.
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
ls -la "$DST/build/lib/libtilelang.so"
# Sanity import
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
"$PY" -c "import tilelang,tilelang.libinfo as li; print(tilelang.__path__); print(li.find_lib_path('tilelang'))"

run() {
  local name="$1"
  echo "==== BEGIN $name $(date -Is) ===="
  set +e
  bash "$HERE/$name"
  local rc=$?
  set -e
  echo "==== END $name rc=$rc $(date -Is) ===="
}

# Primary Simt
run oneshot_sv1_sv9.sh
run oneshot_cf1_cf6.sh
run oneshot_sp1_sp6.sh
# SIMD (PTO-DSL) *d
run oneshot_ptodsl_sv1_sv9d.sh
run oneshot_ptodsl_cf1d_cf6d.sh
# SP *d covered partially in phase scripts; also run phase2 if present
if [[ -f "$HERE/oneshot_ptodsl_phase2.sh" ]]; then
  run oneshot_ptodsl_phase2.sh
fi
# Harvest
if [[ -x "$HERE/harvest_pass_table.sh" ]] || [[ -f "$HERE/harvest_pass_table.sh" ]]; then
  bash "$HERE/harvest_pass_table.sh" || true
fi
echo "=== DONE $(date -Is) ==="
