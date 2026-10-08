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
export PYTHONPATH="$DEPS:$HOME/PTOAS-vmi/ptodsl:$MLIR:${PYTHONPATH:-}"
export PATH="$DEPS/bin:$PATH"
export PYTHON_BIN=$PY
mkdir -p /tmp/st_simtvf_parallel_phase2/kernels_ptodsl "$OUT"
cp -f "$WT/kernels_ptodsl/sv2d_eltwise_bcast_rf.py" /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/
rm -rf /home/happybot/.cache/ptodsl/kernel_* 2>/dev/null

$PY /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py --rf-math 16 64
$PY /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py --rf-math 32 64

for pair in "16 64 input_keep" "32 64 input_stream" "64 64 input_stream" "32 128 input_stream" "32 64 input_keep"; do
  set -- $pair
  R=$1; C=$2; arm=$3
  tag=sv2d_r${R}_c${C}_t32_$arm
  echo "==== $tag ===="
  timeout 900 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_$tag" \
    /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py -- $R $C 32 $arm \
    >"$OUT/${tag}.log" 2>&1
  echo RC:$?
  grep -E "PASS |AssertionError|ValueError|RuntimeError|core0.veccore0|mismatch|Spilled|spill|exceeded vf|error:" "$OUT/${tag}.log" | head -20
done

$PY - <<'PY'
import re, json
from pathlib import Path
OUT = Path("/tmp/st_simtvf_parallel/ptodsl_phase2")
tags = [
  "sv2d_r16_c64_t32_input_keep",
  "sv2d_r32_c64_t32_input_stream",
  "sv2d_r64_c64_t32_input_stream",
  "sv2d_r32_c128_t32_input_stream",
  "sv2d_r32_c64_t32_input_keep",
]
rows=[]
for tag in tags:
    log = OUT/f"{tag}.log"
    t = log.read_text(errors="replace") if log.is_file() else ""
    st = "PASS" if re.search(r"^PASS ", t, re.M) else ("FAIL_SPILL" if ("exceeded vf stack" in t or "Spilled" in t) else ("FAIL" if ("AssertionError" in t or "RuntimeError" in t) else "UNKNOWN"))
    us=None
    for line in t.splitlines():
        if "core0.veccore0" in line:
            nums=re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums: us=float(nums[0]); break
    rows.append(dict(tag=tag,status=st,us=us))
    print(f"{tag}: {st} us={us}")
Path("/tmp/st_simtvf_parallel/ptodsl_phase2/sv2d_input_harvest.json").write_text(json.dumps(rows,indent=2))
print("WROTE harvest")
PY
echo SV2D_INPUT_DONE
