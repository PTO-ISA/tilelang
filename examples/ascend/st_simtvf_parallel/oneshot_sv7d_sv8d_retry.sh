#!/usr/bin/env bash
# Retry SV7d/SV8d only after UB/ensure_layout fixes. NO push.
set -uo pipefail
ROOT=/tmp/pr272_sv1_9d_vmi_20261007
OUT=/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d
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
DEPS="${PTODSL_DEPS:-/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018}"
MLIR_PY="${MLIR_PYTHON_ROOT:-/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core}"
export PYTHONPATH="${DEPS}:${PTOAS_ROOT}/ptodsl:${MLIR_PY}:${PYTHONPATH:-}"
export PATH="${DEPS}/bin:${PATH}"

SUMMARY="$OUT/SUMMARY_sv7d_sv8d_retry.tsv"
echo -e "tag\tstatus\twall_us\tmaxabs\tnote" > "$SUMMARY"

extract_wall() {
  local odir="$1"
  local us=""
  local csv
  csv=$(find "$odir" -name 'core0.veccore0_instr_exe.csv' 2>/dev/null | head -1 || true)
  if [[ -n "$csv" ]]; then
    us=$(python3 -c "
import csv
p='$csv'
rows=list(csv.DictReader(open(p)))
vals=[]
for r in rows:
  for k,v in r.items():
    if 'running_time' in k.lower() or k=='duration_time(us)':
      try: vals.append(float(v))
      except: pass
print(max(vals) if vals else '')
" 2>/dev/null || true)
  fi
  if [[ -z "$us" ]]; then
    us=$(grep -RhoE 'duration_time[^0-9]*([0-9]+\.[0-9]+)' "$odir" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | sort -n | tail -1 || true)
  fi
  # Prefer OpSummary Task Duration if present
  local op
  op=$(find "$odir" -iname '*OpSummary*' 2>/dev/null | head -1 || true)
  if [[ -n "$op" ]]; then
    local alt
    alt=$(python3 -c "
import re
t=open('$op',errors='ignore').read()
# Task Duration(us) style
m=re.findall(r'(?:Task Duration|duration_time)\D+([0-9]+\.[0-9]+)', t)
print(max((float(x) for x in m), default=''))
" 2>/dev/null || true)
    if [[ -n "$alt" ]]; then us="$alt"; fi
  fi
  echo "${us:-}"
}

run_one() {
  local tag="$1"; shift
  local pyfile="$1"; shift
  local odir="$OUT/opsim_${tag}"
  rm -rf "$odir"
  mkdir -p "$odir"
  # bust stale ptodsl cache for this kernel stem
  local stem
  stem=$(basename "$pyfile" .py)
  rm -rf "$HOME/.cache/ptodsl"/kernel_* 2>/dev/null || true
  echo "==== $tag ===="
  local rc=0
  "$SIM_DSL" --soc-version "$SOC" --output "$odir" \
    "$ROOT/$pyfile" -- "$@" > "$OUT/${tag}.log" 2>&1 || rc=$?
  tail -30 "$OUT/${tag}.log" || true
  local status="FAIL_OR_BLOCKED" maxabs="" wall=""
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$OUT/${tag}.log" | head -1 | cut -d= -f2 || true)
  elif grep -qiE 'ensure_layout|exceeded vf stack|ValueError|Assertion failed|Traceback|ub_addr_overflow' "$OUT/${tag}.log"; then
    status="COMPILE_FAIL"
  fi
  wall=$(extract_wall "$odir")
  echo -e "${tag}\t${status}\t${wall:-NA}\t${maxabs:-NA}\trc=${rc}" | tee -a "$SUMMARY"
  echo "RESULT $tag $status wall_us=${wall:-NA} maxabs=${maxabs:-NA} rc=$rc"
}

run_one "sv7d_r64_c128_g64_t32_multiwarp_ub" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 64 32 multiwarp_ub
run_one "sv7d_r64_c128_g64_t32_reload" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 64 32 reload
run_one "sv7d_r64_c128_g128_t32_multiwarp_ub" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 128 32 multiwarp_ub
run_one "sv7d_r64_c128_g128_t32_reload" "kernels_ptodsl/sv7d_reduce_large.py" 64 128 128 32 reload
run_one "sv8d_r64_c128_g16_t32_live" "kernels_ptodsl/sv8d_case3_bcast.py" 64 128 16 32 live
run_one "sv8d_r64_c128_g16_t32_spill_dist" "kernels_ptodsl/sv8d_case3_bcast.py" 64 128 16 32 spill_dist

echo "==== RETRY SUMMARY ===="
cat "$SUMMARY"
