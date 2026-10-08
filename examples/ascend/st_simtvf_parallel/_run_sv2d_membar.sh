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
rm -rf /home/happybot/.cache/ptodsl/kernel_* 2>/dev/null
for C in 32 128 256; do
  for arm in scale_keep_stream full_reload scale_ub_membar; do
    tag=sv2d_r32_c${C}_t32_$arm
    echo "==== $tag ===="
    timeout 900 "$SIM" --soc-version Ascend950PR_9599 --output "$OUT/opsim_$tag" \
      /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_eltwise_bcast_rf.py -- 32 $C 32 $arm \
      >"$OUT/${tag}.log" 2>&1
    echo RC:$?
    grep -E "PASS |AssertionError|ValueError|RuntimeError|core0.veccore0|mismatch|Spilled|spill|error:" "$OUT/${tag}.log" | head -30
  done
done
echo "==== HARVEST ===="
$PY - <<'PY'
import csv, glob, re
from collections import defaultdict
from pathlib import Path
OUT = Path("/tmp/st_simtvf_parallel/ptodsl_phase2")
FRAMING = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}

def wall_us(tag):
    log = OUT / f"{tag}.log"
    txt = log.read_text(errors="replace") if log.is_file() else ""
    for line in txt.splitlines():
        if "core0.veccore0" in line:
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums:
                return float(nums[0])
    for p in sorted((OUT / f"opsim_{tag}").rglob("msprof.stdout.log")):
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
    if "AssertionError" in t or "RuntimeError" in t:
        return "FAIL"
    return "UNKNOWN"

def latest_csv(tag):
    hits = sorted((OUT / f"opsim_{tag}").rglob("core0.veccore0_instr_exe.csv"))
    return hits[-1] if hits else None

def harvest(tag):
    st = status(tag)
    us = wall_us(tag)
    csv_path = latest_csv(tag)
    row = {"tag": tag, "status": st, "us": us, "csv": str(csv_path) if csv_path else None,
           "rvecex_cyc": None, "rvecld_cyc": None, "rvecst_cyc": None, "rvec_all_cyc": None,
           "membar_cc": None, "membar_cyc": None, "exipc": None, "vf_wrapper_cyc": None,
           "body_instr": None, "ex_instr": None}
    if not csv_path:
        return row
    rows = list(csv.DictReader(csv_path.open()))
    by_pipe_y = defaultdict(float)
    by_pipe_c = defaultdict(int)
    by_instr_y = defaultdict(float)
    by_instr_c = defaultdict(int)
    body = 0
    ex = 0
    for r in rows:
        instr = (r.get("instr") or "").strip()
        pipe = (r.get("pipe") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = float(r.get("cycles") or 0)
        except ValueError:
            continue
        by_pipe_y[pipe] += cyc
        by_pipe_c[pipe] += cc
        by_instr_y[instr] += cyc
        by_instr_c[instr] += cc
        if instr.lower() not in FRAMING:
            body += cc
        if "EX" in pipe.upper() or instr.lower().startswith("simt_"):
            ex += cc
    row["rvecex_cyc"] = int(by_pipe_y.get("RVECEX", 0))
    row["rvecld_cyc"] = int(by_pipe_y.get("RVECLD", 0))
    row["rvecst_cyc"] = int(by_pipe_y.get("RVECST", 0))
    rvec_pipes = ("RVECEX", "RVECLD", "RVECST", "RVECSU", "RVECLP")
    row["rvec_all_cyc"] = int(sum(by_pipe_y.get(p, 0) for p in rvec_pipes))
    row["body_instr"] = body
    row["ex_instr"] = ex
    # mem_bar opcodes
    for k, c in by_instr_c.items():
        if "MEM" in k.upper() and "BAR" in k.upper() or k.upper() in ("MEM_BAR", "MEMBAR", "PIPE_BARRIER", "BARRIER"):
            row["membar_cc"] = (row["membar_cc"] or 0) + c
            row["membar_cyc"] = (row["membar_cyc"] or 0) + int(by_instr_y[k])
    # also catch PIPE_BARRIER-ish
    for k, c in by_instr_c.items():
        ku = k.upper()
        if "BARRIER" in ku or ku in ("PIPE_BARRIER", "MEM_BAR"):
            if row["membar_cc"] is None:
                row["membar_cc"] = 0
                row["membar_cyc"] = 0
            # avoid double count if already counted
            if not (("MEM" in ku and "BAR" in ku) or ku in ("MEM_BAR", "MEMBAR", "PIPE_BARRIER", "BARRIER")):
                pass
    # VF wrapper / EXIPC
    for name in ("VF_SIMT", "VF", "VF_SIMD", "EXIPC", "IFU"):
        key = next((k for k in by_instr_c if k.upper() == name), None)
        if key:
            if name == "EXIPC":
                row["exipc"] = by_instr_c[key] / by_instr_y[key] if by_instr_y[key] else None
            if name in ("VF_SIMT", "VF", "VF_SIMD"):
                row["vf_wrapper_cyc"] = int(by_instr_y[key])
    # list barrier-like instrs for honesty
    bars = [(k, by_instr_c[k], int(by_instr_y[k])) for k in by_instr_c if "BAR" in k.upper() or "MEM" in k.upper()]
    row["bar_instrs"] = bars
    return row

tags = []
for C in (32, 128, 256):
    for arm in ("scale_keep_stream", "full_reload", "scale_ub_membar"):
        tags.append(f"sv2d_r32_c{C}_t32_{arm}")

print(f"{'tag':<42} {'st':<10} {'us':>6} {'RVECEX':>8} {'RVECLD':>8} {'RVECST':>8} {'RVEC*':>8} {'body':>6} {'ex':>6} {'VF_w':>6} {'EXIPC':>6} bars")
rows = []
for t in tags:
    r = harvest(t)
    rows.append(r)
    print(f"{r['tag']:<42} {r['status']:<10} {str(r['us']):>6} {str(r['rvecex_cyc']):>8} {str(r['rvecld_cyc']):>8} {str(r['rvecst_cyc']):>8} {str(r['rvec_all_cyc']):>8} {str(r['body_instr']):>6} {str(r['ex_instr']):>6} {str(r['vf_wrapper_cyc']):>6} {str(r['exipc']):>6} {r.get('bar_instrs')}")

# write machine-readable
import json
(OUT / "sv2d_membar_harvest.json").write_text(json.dumps(rows, indent=2))
print("WROTE", OUT / "sv2d_membar_harvest.json")
PY
echo SV2D_MEMBAR_DONE
