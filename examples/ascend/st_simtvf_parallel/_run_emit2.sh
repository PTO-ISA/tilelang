#!/bin/bash
set +e
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
MLIR=/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
OUT=/tmp/st_simtvf_parallel/ptodsl_phase1
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

for spec in "sv1d:kernels_ptodsl/sv1d_stream_eltwise.py:256 32" "cf1d:kernels_ptodsl/cf1d_pred_thresh_keep.py:256 8 32" "sp1d:kernels_ptodsl/sp1d_dual_scatter_keep_pos.py:32 2 128 32 32"; do
  name=${spec%%:*}; rest=${spec#*:}; py=${rest%%:*}; args=${rest#*:}
  echo "==== EMIT $name ===="
  timeout 300 $PY $py $args --emit-mlir > "$OUT/${name}.mlir" 2>"$OUT/${name}_emit.err"
  echo ${name}_EMIT_RC:$?
  wc -l "$OUT/${name}.mlir"
  if [[ ! -s "$OUT/${name}.mlir" ]]; then tail -25 "$OUT/${name}_emit.err"; fi
done

# Try opsim SV1d if mlir ok
if [[ -s "$OUT/sv1d.mlir" ]]; then
  SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
  echo "==== OPSIM SV1d ===="
  timeout 420 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_sv1d_e256_t32" \
    /tmp/st_simtvf_parallel/kernels_ptodsl/sv1d_stream_eltwise.py -- 256 32 \
    >"$OUT/sv1d_opsim.log" 2>&1
  echo SV1d_OPSIM_RC:$?
  grep -E 'PASS sv1d|PASS |FAIL|Traceback|Error|error:' "$OUT/sv1d_opsim.log" | head -50
  tail -40 "$OUT/sv1d_opsim.log"
fi
echo DONE
