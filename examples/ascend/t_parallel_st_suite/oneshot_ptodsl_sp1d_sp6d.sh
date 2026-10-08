#!/usr/bin/env bash
# PTO-DSL — SP1d / SP2d / SP3d / SP4d / SP5d / SP6d.
# Run on pto-b10 login node (Ascend950PR_9599 opsim).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ST_SIMTVF_OUT:-/tmp/t_parallel_st_suite/ptodsl_sp1d_sp6d}"
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

echo "[ptodsl-sp1d-sp6d] OUT=$OUT PY=$PY SOC=$SOC ROOT=$ROOT"
if ! "$PY" -c 'from ptodsl import pto; print("ptodsl OK")'; then
  echo "[ptodsl-sp1d-sp6d] BLOCKED: cannot import ptodsl"
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
  tail -20 "$OUT/${tag}.log" || true
  local status="FAIL_OR_BLOCKED" maxabs="" wall=""
  if grep -q "PASS ${tag}" "$OUT/${tag}.log" 2>/dev/null; then
    status="PASS"
    maxabs=$(grep -oE 'maxabs=[0-9.eE+-]+' "$OUT/${tag}.log" | head -1 | cut -d= -f2 || true)
  elif grep -qiE 'exceeded vf stack|COMPILE|Error|Traceback|spill|AttributeError|UNRUN' "$OUT/${tag}.log" 2>/dev/null; then
    status="COMPILE_FAIL"
  fi
  wall=$(awk '/core0.veccore0[[:space:]]+[0-9]/ {print $2; exit}' "$OUT/${tag}.log" 2>/dev/null || true)
  if [[ -z "$wall" ]]; then
    wall=$(grep -RhoE 'core0\.veccore0[[:space:]]+[0-9]+\.[0-9]+' "$odir" 2>/dev/null \
      | head -1 | awk '{print $2}' || true)
  fi
  echo -e "${tag}\t${status}\t${wall}\t${maxabs}\trc=${rc}" | tee -a "$SUMMARY"
}

# SP1d keep_pos
run_one "sp1d_t32_k2_h128_g32_t32_keep_pos" "kernels_ptodsl/sp1d_dual_scatter_keep_pos.py" 32 2 128 32 32

# SP2d Acc × sf/w matrix
run_one "sp2d_t32_k2_h128_g32_t32_sf0_w0_keep_acc"  "kernels_ptodsl/sp2d_dual_gather_wreduce.py" 32 2 128 32 32 sf0 w0 keep_acc
run_one "sp2d_t32_k2_h128_g32_t32_sf0_w0_remat_acc" "kernels_ptodsl/sp2d_dual_gather_wreduce.py" 32 2 128 32 32 sf0 w0 remat_acc
run_one "sp2d_t32_k2_h128_g32_t32_sf1_w1_keep_acc"  "kernels_ptodsl/sp2d_dual_gather_wreduce.py" 32 2 128 32 32 sf1 w1 keep_acc
run_one "sp2d_t32_k2_h128_g32_t32_sf1_w1_remat_acc" "kernels_ptodsl/sp2d_dual_gather_wreduce.py" 32 2 128 32 32 sf1 w1 remat_acc

# SP3d fp32 / e8m0
run_one "sp3d_m32_h128_g32_t32_fp32" "kernels_ptodsl/sp3d_sf_pack_e8m0.py" --arm fp32 --tag sp3d_m32_h128_g32_t32_fp32
run_one "sp3d_m32_h128_g32_t32_e8m0" "kernels_ptodsl/sp3d_sf_pack_e8m0.py" --arm e8m0 --tag sp3d_m32_h128_g32_t32_e8m0

# SP4d full-lane mask (pad holes via vsel; no early exit)
run_one "sp4d_e64_h128_t32_pad25" "kernels_ptodsl/sp4d_pad_gather.py" 64 128 32 25

# SP5d sideband (two gathers) vs interleave (one Pack slot, then split)
run_one "sp5d_n64_h128_g32_qg32_t32_sideband" "kernels_ptodsl/sp5d_sideband_vs_interleave.py" 64 128 32 32 32 sideband
run_one "sp5d_n64_h128_g32_qg32_t32_interleave" "kernels_ptodsl/sp5d_sideband_vs_interleave.py" 64 128 32 32 32 interleave

# SP6d soft LUT (gather + vselr) × unpack / unpack_sf
run_one "sp6d_n32_h128_g32_t32_unpack_gather"    "kernels_ptodsl/sp6d_fp4_unpack.py" 32 128 32 32 unpack gather
run_one "sp6d_n32_h128_g32_t32_unpack_vselr"     "kernels_ptodsl/sp6d_fp4_unpack.py" 32 128 32 32 unpack vselr
run_one "sp6d_n32_h128_g32_t32_unpack_sf_gather" "kernels_ptodsl/sp6d_fp4_unpack.py" 32 128 32 32 unpack_sf gather
run_one "sp6d_n32_h128_g32_t32_unpack_sf_vselr"  "kernels_ptodsl/sp6d_fp4_unpack.py" 32 128 32 32 unpack_sf vselr

echo "[ptodsl-sp1d-sp6d] done. SUMMARY:"
cat "$SUMMARY"
