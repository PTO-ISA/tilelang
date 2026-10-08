#!/bin/bash
set +e
source ~/projects/env.sh >/dev/null 2>&1
export PYTHONPATH="/home/happybot/PTOAS-vmi/ptodsl:${PTOAS_PYTHON_BUILD:-/home/happybot/PTOAS-vmi/build-llvm21/python}:${MLIR_PYTHON_ROOT:-/mnt/fluxdata/happybot/llvm-vpto/build-llvm21/tools/mlir/python_packages/mlir_core}:${PYTHONPATH}"
# also try ptoas package path
for d in /home/happybot/PTOAS-vmi/build-llvm21/python /home/happybot/PTOAS-vmi/python /mnt/fluxdata/happybot/PTOAS-pr1310-build/python; do
  [ -d "$d" ] && PYTHONPATH="$d:$PYTHONPATH"
done
export PYTHONPATH
echo PYTHONPATH_HEAD=$(echo $PYTHONPATH | tr ':' '\n' | head -8)
PY=~/projects/tilelang-deepseek/.venv-npu/bin/python
$PY - <<'PYEOF'
import sys
print("sys.path[:6]=", sys.path[:6])
try:
  import ptoas
  print("ptoas", getattr(ptoas, "__file__", ptoas))
except Exception as e:
  print("ptoas fail", e)
try:
  from ptodsl import pto
  print("pto import OK")
except Exception as e:
  print("pto fail", e)
# find ptoas.mlir
import os
for root in ["/home/happybot/PTOAS-vmi", "/mnt/fluxdata/happybot"]:
  for dirpath, dirs, files in os.walk(root):
    if "ptoas" in dirs and os.path.isdir(os.path.join(dirpath, "ptoas", "mlir")):
      print("FOUND", os.path.join(dirpath, "ptoas"))
      dirs.clear()
    if dirpath.count(os.sep) - root.count(os.sep) > 4:
      dirs.clear()
PYEOF

# try running a tiny emit-mlir from MaskedGather (no opsim)
cd ~/projects/pto-vmi
echo ====EMIT_TRY====
export ASCEND_HOME_PATH=/mnt/fluxdata/Ascend/cann_91b3/cann-9.1.0-beta.3
set +u; source $ASCEND_HOME_PATH/set_env.sh; set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PTOAS_ROOT=/home/happybot/PTOAS-vmi
export PYTHONPATH="/home/happybot/PTOAS-vmi/ptodsl:/home/happybot/PTOAS-vmi/build-llvm21/python:${MLIR_PYTHON_ROOT}:$PYTHONPATH"
timeout 90 $PY dsl/MaskedGatherKernel/MaskedGatherKernel_case0_fp32_fp32_32_64.py --emit-mlir 2>&1 | tail -40
echo EXIT:$?
