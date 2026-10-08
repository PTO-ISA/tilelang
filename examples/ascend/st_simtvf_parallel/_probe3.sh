#!/bin/bash
set +e
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
MLIR=/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core
ASC=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
PY=/home/happybot/projects/tilelang-deepseek/.venv-npu/bin/python
export ASCEND_HOME_PATH=$ASC
source $ASC/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$DEPS:$MLIR"
export PATH="$DEPS/bin:$PATH"
echo "PYTHONPATH=$PYTHONPATH"
$PY - <<'PYEOF'
from ptodsl import pto
print("pto OK", pto)
print("has vmi", hasattr(pto, "vmi"))
print("vmi ops", [x for x in dir(pto.vmi) if not x.startswith("_")][:50])
PYEOF

# Copy kernels to remote scratch and try emit-mlir on SV1d
SCR=/tmp/st_simtvf_parallel/ptodsl_phase1
mkdir -p $SCR/kernels_ptodsl
# kernels will be synced separately; for now try MaskedGather from deps/ptodsl world
cd /home/happybot/projects/pto-vmi
timeout 120 $PY dsl/MaskedGatherKernel/MaskedGatherKernel_case0_fp32_fp32_32_64.py --emit-mlir > /tmp/mg_mlir.txt 2> /tmp/mg_err.txt
echo EMIT_RC:$?
tail -20 /tmp/mg_err.txt
head -5 /tmp/mg_mlir.txt
wc -l /tmp/mg_mlir.txt
