#!/bin/bash
set -euo pipefail
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
MLIR=/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
OUT=/tmp/st_simtvf_parallel/ptodsl_phase1
export ASCEND_HOME_PATH=$ASC
set +u; source $ASC/set_env.sh; set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$DEPS:$MLIR"
export PATH="$DEPS/bin:$PATH"
export PYTHON_BIN=$PY

# Ensure WT has suite
mkdir -p "$WT"
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C "$WT"
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C /tmp/st_simtvf_parallel
ls "$WT/kernels_ptodsl"
mkdir -p "$OUT"

cd /tmp/st_simtvf_parallel

echo "==== EMIT SV1d ===="
timeout 180 $PY kernels_ptodsl/sv1d_stream_eltwise.py 256 32 --emit-mlir > "$OUT/sv1d.mlir" 2>"$OUT/sv1d_emit.err" || true
echo SV1d_EMIT:$?
wc -l "$OUT/sv1d.mlir" || true
tail -30 "$OUT/sv1d_emit.err" || true

echo "==== EMIT CF1d ===="
timeout 180 $PY kernels_ptodsl/cf1d_pred_thresh_keep.py 256 8 32 --emit-mlir > "$OUT/cf1d.mlir" 2>"$OUT/cf1d_emit.err" || true
echo CF1d_EMIT:$?
wc -l "$OUT/cf1d.mlir" || true
tail -40 "$OUT/cf1d_emit.err" || true

echo "==== EMIT SP1d ===="
timeout 300 $PY kernels_ptodsl/sp1d_dual_scatter_keep_pos.py 32 2 128 32 32 --emit-mlir > "$OUT/sp1d.mlir" 2>"$OUT/sp1d_emit.err" || true
echo SP1d_EMIT:$?
wc -l "$OUT/sp1d.mlir" || true
tail -40 "$OUT/sp1d_emit.err" || true

# If emit worked, try ACL run under sim_dsl for SV1d only (fastest)
SIM=/home/happybot/PTOAS-vmi/scripts/sim_dsl.sh
if [[ -x "$SIM" && -s "$OUT/sv1d.mlir" ]]; then
  echo "==== OPSIM SV1d via sim_dsl ===="
  timeout 300 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_sv1d_e256_t32" \
    /tmp/st_simtvf_parallel/kernels_ptodsl/sv1d_stream_eltwise.py -- 256 32 \
    >"$OUT/sv1d_opsim.log" 2>&1 || true
  echo SV1d_OPSIM_RC:$?
  grep -E 'PASS|FAIL|error|Error|Traceback' "$OUT/sv1d_opsim.log" | head -40 || true
  tail -30 "$OUT/sv1d_opsim.log" || true
fi

echo DONE
