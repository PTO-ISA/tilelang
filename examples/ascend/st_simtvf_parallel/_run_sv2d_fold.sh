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
mkdir -p /tmp/st_simtvf_parallel_phase2/kernels_ptodsl "$WT/kernels_ptodsl" "$OUT"
cp -f /tmp/sv2d_eltwise_bcast_rf.py /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/
cp -f /tmp/sv2d_eltwise_bcast_rf.py "$WT/kernels_ptodsl/"
rm -rf /home/happybot/.cache/ptodsl/kernel_* 2>/dev/null

$PY /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py --rf-math

# Primary C=128 keep; also C=64 keep (quick) + C=128 reload contrast
for pair in "64 fold_scale_keep" "128 fold_scale_keep" "128 fold_scale_reload"; do
  set -- $pair
  C=$1; arm=$2
  tag=sv2d_r32_c${C}_t32_$arm
  echo "==== $tag ===="
  timeout 900 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_$tag" \
    /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py -- 32 $C 32 $arm \
    >"$OUT/${tag}.log" 2>&1
  echo RC:$?
  grep -E "PASS |AssertionError|ValueError|RuntimeError|core0.veccore0|mismatch|Spilled|spill|error:|VMI-UNSUPPORTED" "$OUT/${tag}.log" | head -40
done

echo "==== HARVEST ===="
$PY - <<'PY'
import csv, re, json
from pathlib import Path
OUT = Path("/tmp/st_simtvf_parallel/ptodsl_phase2")
tags = [
  "sv2d_r32_c64_t32_fold_scale_keep",
  "sv2d_r32_c128_t32_fold_scale_keep",
  "sv2d_r32_c128_t32_fold_scale_reload",
]

def wall_us(tag):
    for p in [OUT / f"{tag}.log", *sorted((OUT / f"opsim_{tag}").rglob("msprof.stdout.log"))]:
        if not p.is_file():
            continue
        for line in p.read_text(errors="replace").splitlines():
            if "core0.veccore0" in line:
                nums = re.findall(r"([0-9]+\.[0-9]+)", line)
                if nums:
                    return float(nums[0])
    return None

def status(tag):
    log = OUT / f"{tag}.log"
    if not log.is_file():
        return "MISSING"
    t = log.read_text(errors="replace")
    if re.search(r"^PASS ", t, re.M):
        return "PASS"
    if "exceeded vf stack" in t or "Spilled" in t:
        return "FAIL_SPILL"
    if "AssertionError" in t or "RuntimeError" in t or "ValueError" in t:
        return "FAIL"
    return "UNKNOWN"

def latest_csv(tag):
    hits = sorted((OUT / f"opsim_{tag}").rglob("core0.veccore0_instr_exe.csv"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None

rows = []
for tag in tags:
    st = status(tag); us = wall_us(tag); csv_path = latest_csv(tag)
    rvecex = rvecld = None
    if csv_path:
        by = {}
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                pipe = (r.get("pipe") or "").strip()
                try:
                    cyc = int(float(r.get("cycles") or 0))
                except Exception:
                    cyc = 0
                by[pipe] = by.get(pipe, 0) + cyc
        rvecex = by.get("RVECEX"); rvecld = by.get("RVECLD")
    row = dict(tag=tag, status=st, us=us, rvecex_cyc=rvecex, rvecld_cyc=rvecld)
    rows.append(row)
    print(f"{tag}: {st} us={us} RVECEX={rvecex} RVECLD={rvecld}")
Path("/tmp/st_simtvf_parallel/ptodsl_phase2/sv2d_fold_harvest.json").write_text(json.dumps(rows, indent=2))
print("WROTE harvest")
PY
echo SV2D_FOLD_DONE
