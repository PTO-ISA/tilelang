#!/bin/bash
set +e
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
MLIR=/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
OUT=/tmp/st_simtvf_parallel/ptodsl_phase2
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
export ASCEND_HOME_PATH=$ASC
source $ASC/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$DEPS:$MLIR"
export PATH="$DEPS/bin:$PATH"
export PYTHON_BIN=$PY
mkdir -p /tmp/st_simtvf_parallel_phase2/kernels_ptodsl "$WT/kernels_ptodsl" "$OUT"
cp -f /tmp/sv2d_eltwise_bcast_rf.py /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/
cp -f /tmp/sv2d_eltwise_bcast_rf.py "$WT/kernels_ptodsl/"
for C in 128 256; do
  $PY /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py 32 $C 32 scale_keep_stream --rf-math
done
rm -rf /home/happybot/.cache/ptodsl/kernel_* 2>/dev/null
for C in 128 256; do
  for arm in scale_keep_stream full_reload row_keep_full row_keep_tile; do
    tag=sv2d_r32_c${C}_t32_$arm
    echo "==== $tag ===="
    timeout 900 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_$tag" \
      /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py -- 32 $C 32 $arm \
      >"$OUT/${tag}.log" 2>&1
    echo RC:$?
    grep -E "PASS |AssertionError|ValueError|RuntimeError|core0.veccore0|mismatch|VMI-UNSUPPORTED|Spilled|spill|error:" "$OUT/${tag}.log" | head -40
  done
done
echo SV2D_C128_C256_DONE
