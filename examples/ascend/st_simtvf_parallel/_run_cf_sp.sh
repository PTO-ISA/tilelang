#!/bin/bash
set +e
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
MLIR=/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
OUT=/tmp/st_simtvf_parallel/ptodsl_phase1
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
export ASCEND_HOME_PATH=$ASC
source $ASC/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$DEPS:$MLIR"
export PATH="$DEPS/bin:$PATH"
export PYTHON_BIN=$PY
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C /tmp/st_simtvf_parallel
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C "$WT"
mkdir -p "$OUT"
cd /tmp/st_simtvf_parallel

emit_and_opsim() {
  local name=$1; local py=$2; shift 2
  echo "==== EMIT $name ===="
  timeout 300 $PY "$py" "$@" --emit-mlir > "$OUT/${name}.mlir" 2>"$OUT/${name}_emit.err"
  echo ${name}_EMIT_RC:$?
  wc -l "$OUT/${name}.mlir"
  if [[ ! -s "$OUT/${name}.mlir" ]]; then tail -30 "$OUT/${name}_emit.err"; return 1; fi
  echo "==== OPSIM $name ===="
  local tag
  case $name in
    cf1d) tag=cf1d_e256_k8_t32_keep ;;
    sp1d) tag=sp1d_t32_k2_h128_g32_t32_keep_pos ;;
    *) tag=$name ;;
  esac
  timeout 600 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_${tag}" \
    "/tmp/st_simtvf_parallel/$py" -- "$@" >"$OUT/${name}_opsim.log" 2>&1
  echo ${name}_OPSIM_RC:$?
  grep -E "PASS ${tag}|PASS |FAIL|Traceback|Error" "$OUT/${name}_opsim.log" | head -30
  grep -E 'core0.veccore0|duration_time' "$OUT/${name}_opsim.log" | head -10
  tail -20 "$OUT/${name}_opsim.log"
}

emit_and_opsim cf1d kernels_ptodsl/cf1d_pred_thresh_keep.py 256 8 32
emit_and_opsim sp1d kernels_ptodsl/sp1d_dual_scatter_keep_pos.py 32 2 128 32 32
echo DONE
