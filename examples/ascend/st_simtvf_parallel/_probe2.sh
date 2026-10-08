#!/bin/bash
set +e
# Try known working deps stacks for ptodsl
CANDS=(
  "/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018"
  "/mnt/fluxdata/happybot/ptoas-v0.58-py312"
  "/mnt/fluxdata/happybot/ptoas-v0.58-release"
  "/home/happybot/PTOAS-vmi/build-llvm21/python"
)
PY=~/projects/tilelang-deepseek/.venv-npu/bin/python
export ASCEND_HOME_PATH=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
source $ASCEND_HOME_PATH/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLIR_PYTHON_ROOT=/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core

for d in "${CANDS[@]}"; do
  echo "==== TRY $d ===="
  export PYTHONPATH="/home/happybot/PTOAS-vmi/ptodsl:$d:$MLIR_PYTHON_ROOT"
  $PY -c 'from ptodsl import pto; print("OK", pto.__file__ if hasattr(pto,"__file__") else "pto")' 2>&1 | tail -5
done

# Also try run_st.sh list / how TopkGate was meant to run
echo ====RUN_ST_ENV====
cd ~/projects/pto-vmi
# peek end of run_st.sh for PYTHONPATH setup
grep -n 'PYTHONPATH\|ptoas\|ptodsl\|PTOAS' dsl/run_st.sh | head -40
echo ====
# check if chenjinlin paths exist or happybot mirrors
ls /home/chenjinlin/projects/PTOAS/ptodsl 2>/dev/null || echo no_chenjinlin
ls /home/happybot/projects/PTOAS-main/ptodsl/ptodsl 2>/dev/null && echo have_PTOAS_main
# try PTOAS-main with its own build
export PYTHONPATH="/home/happybot/projects/PTOAS-main/ptodsl:/home/happybot/projects/PTOAS-main/build-llvm21/python:$MLIR_PYTHON_ROOT"
$PY -c 'from ptodsl import pto; print("PTOAS-main OK")' 2>&1 | tail -8

# try wheel site-packages style
ls /mnt/fluxdata/happybot/ptoas-v0.58-py312/ | head
ls /mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018/ | head
export PYTHONPATH="/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018:$MLIR_PYTHON_ROOT"
$PY -c 'import ptodsl; from ptodsl import pto; print("deps OK", ptodsl.__file__)' 2>&1 | tail -8
