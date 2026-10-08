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
# clear ptodsl cache for SP1d
rm -rf /home/happybot/.cache/ptodsl/kernel_* 2>/dev/null
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C /tmp/st_simtvf_parallel
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C "$WT"
mkdir -p "$OUT"
cd /tmp/st_simtvf_parallel
echo "==== EMIT sp1d ===="
timeout 300 $PY kernels_ptodsl/sp1d_dual_scatter_keep_pos.py 32 2 128 32 32 --emit-mlir > "$OUT/sp1d.mlir" 2>"$OUT/sp1d_emit.err"
echo SP1d_EMIT_RC:$?
wc -l "$OUT/sp1d.mlir"
if [[ ! -s "$OUT/sp1d.mlir" ]]; then tail -40 "$OUT/sp1d_emit.err"; exit 1; fi
echo "==== OPSIM sp1d ===="
timeout 900 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_sp1d_t32_k2_h128_g32_t32_keep_pos" \
  /tmp/st_simtvf_parallel/kernels_ptodsl/sp1d_dual_scatter_keep_pos.py -- 32 2 128 32 32 \
  >"$OUT/sp1d_opsim.log" 2>&1
echo SP1d_OPSIM_RC:$?
grep -E 'PASS sp1d|PASS |FAIL|Traceback|Error|VMI-UNSUPPORTED|duration_time' "$OUT/sp1d_opsim.log" | head -40
tail -30 "$OUT/sp1d_opsim.log"
echo DONE
