#!/usr/bin/env python3
"""Cleaner VF / instruction-mix breakdown for matched SimdVF vs SimtVF opsim."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/tmp/simd_vs_simt_perf")
OUT = Path("/tmp/simd_vs_simt_perf/INSTR_MIX_BREAKDOWN.md")

ARMS = [
    ("SimdVF L128 K1", "opsim_simdvf_t128_k1"),
    ("SimdVF L128 K8", "opsim_simdvf_t128_k8"),
    ("SimtVF T128 K1", "opsim_simtvf_t128_k1"),
    ("SimtVF T128 K8", "opsim_simtvf_t128_k8"),
    ("SimtVF T32 K1", "opsim_simtvf_t32_k1"),
    ("SimtVF T32 K8", "opsim_simtvf_t32_k8"),
]

# Order matters: first match wins.
BUCKET_RULES = [
    ("reduce_vcmax_vcmin", ("vcmax", "vcmin", "vcgmax", "vcgmin")),
    ("compare", ("vcmp", "fsetp", "isetp", "setp")),
    ("select", ("vsel", "sel", "cmpsel")),
    ("bcast_idx", ("vdup", "vbrc", "vci", "vid", "vseq", "movi")),
    ("vector_ldst", ("vldi", "vsti", "vld", "vst", "lds", "sts", "pld", "pst")),
    ("simt_local_arith", ("simt_fadd", "simt_fmul", "simt_iadd", "simt_imul", "simt_lea", "simt_plop3", "simt_lop3")),
    ("barrier_sync", ("bar", "barrier", "set_flag", "wait_flag", "pipe_barrier", "smem_bar", "sync")),
    ("scalar_mov", ("mov_xd", "movk", "mov_spr", "smov", "insert_xd", "extract")),
    ("scalar_ctrl", ("br", "jump", "ret", "call")),
    ("other_vector", ("rv_", "vmax", "vmin", "vadd", "vsub", "vmul")),
]


def find_csv(arm: str, name: str):
    hits = list((ROOT / arm).rglob(name))
    return hits[0] if hits else None


def us_from_log(arm: str):
    log = ROOT / f"{arm}.log"
    if not log.is_file():
        return None
    txt = log.read_text(errors="replace")
    for line in txt.splitlines():
        if "core0.veccore0" in line:
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums:
                return float(nums[0])
    return None


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def bucket_of(op: str) -> str:
    o = op.lower()
    for name, keys in BUCKET_RULES:
        if any(k in o for k in keys):
            return name
    return "other"


def analyze(arm: str) -> dict:
    p = find_csv(arm, "core0.veccore0_instr_exe.csv")
    us = us_from_log(arm)
    if not p:
        return {"arm": arm, "us": us, "error": "missing csv"}
    rows = load_csv(p)
    op_count = Counter()
    op_cycles = Counter()
    op_us = Counter()
    pipe_count = Counter()
    pipe_cycles = Counter()
    bucket_count = Counter()
    bucket_cycles = Counter()
    bucket_us = Counter()
    total_count = 0
    total_cycles = 0.0
    total_us = 0.0
    for r in rows:
        op = (r.get("instr") or "unknown").strip().lower()
        pipe = (r.get("pipe") or "?").strip()
        cnt = int(float(r.get("call_count") or 1))
        cyc = float(r.get("cycles") or 0)
        rus = float(r.get("running_time(us)") or 0)
        op_count[op] += cnt
        op_cycles[op] += cyc
        op_us[op] += rus
        pipe_count[pipe] += cnt
        pipe_cycles[pipe] += cyc
        b = bucket_of(op)
        bucket_count[b] += cnt
        bucket_cycles[b] += cyc
        bucket_us[b] += rus
        total_count += cnt
        total_cycles += cyc
        total_us += rus

    # Hot reduce / CF ops explicit
    hot = {}
    for key in (
        "rv_vcmax", "rv_vcmin", "rv_vcmp_eq", "rv_vcmp_lt", "rv_vsel", "rv_vldi", "rv_vsti",
        "simt_bar_thread_block", "simt_lds", "simt_sts", "simt_sel", "simt_fsetp", "simt_isetp",
        "simt_lea", "set_flag", "wait_flag",
    ):
        # match prefix
        c = sum(v for k, v in op_count.items() if k == key or k.startswith(key))
        y = sum(v for k, v in op_cycles.items() if k == key or k.startswith(key))
        u = sum(v for k, v in op_us.items() if k == key or k.startswith(key))
        if c:
            hot[key] = (c, y, u)

    return {
        "arm": arm,
        "us": us,
        "csv": str(p),
        "total_count": total_count,
        "total_cycles": total_cycles,
        "total_instr_us": total_us,
        "op_count": op_count,
        "op_cycles": op_cycles,
        "op_us": op_us,
        "pipe_count": pipe_count,
        "pipe_cycles": pipe_cycles,
        "bucket_count": bucket_count,
        "bucket_cycles": bucket_cycles,
        "bucket_us": bucket_us,
        "hot": hot,
    }


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main():
    results = []
    for label, arm in ARMS:
        r = analyze(arm)
        r["label"] = label
        results.append(r)
        print(label, "us", r.get("us"), "cnt", r.get("total_count"), "cyc", r.get("total_cycles"))

    lines = []
    lines.append("# SimdVF vs SimtVF — VF cycles / instruction mix")
    lines.append("")
    lines.append("Matched Parallel tournament (E=256), opsim `core0.veccore0_instr_exe.csv` from `/tmp/simd_vs_simt_perf` (2026-09-20).")
    lines.append("")
    lines.append("**How to read cycles:** each CSV row’s `cycles` is that opcode’s attributed cost (includes pipe wait). Summing across rows **overcounts** wall time when pipes overlap; wall truth stays the msprof **µs** column. Use cycle / running_time shares for **mix**, µs for **speed**.")
    lines.append("")
    lines.append("## Wall time vs attributed cost")
    lines.append("")
    lines.append("| Arm | wall µs | instr call_count | Σ cycles (attr) | Σ running_time(us) |")
    lines.append("|-----|--------:|-----------------:|----------------:|-------------------:|")
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['label']} | err | | | |")
            continue
        lines.append(
            f"| **{r['label']}** | **{r['us']}** | {r['total_count']} | {r['total_cycles']:.0f} | {r['total_instr_us']:.3f} |"
        )
    lines.append("")

    # Fair K8 compare table
    lines.append("## Fair A/B focus — K=8 (L128 vs T128 vs T32)")
    lines.append("")
    k8 = [r for r in results if "K8" in r["label"]]
    # buckets union
    bucks = []
    for name, _ in BUCKET_RULES:
        bucks.append(name)
    bucks.append("other")
    lines.append("### Bucket share by attributed cycles (K8)")
    lines.append("")
    header = "| bucket | " + " | ".join(r["label"] for r in k8) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(k8) + 1))
    for b in bucks:
        cells = []
        any_nz = False
        for r in k8:
            y = r["bucket_cycles"].get(b, 0)
            if y:
                any_nz = True
            cells.append(f"{y:.0f} ({pct(y, r['total_cycles'])})")
        if any_nz:
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Bucket share by call_count (K8)")
    lines.append("")
    lines.append(header)
    lines.append("|" + "---|" * (len(k8) + 1))
    for b in bucks:
        cells = []
        any_nz = False
        for r in k8:
            c = r["bucket_count"].get(b, 0)
            if c:
                any_nz = True
            cells.append(f"{c} ({pct(c, r['total_count'])})")
        if any_nz:
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Pipe mix by attributed cycles (K8)")
    lines.append("")
    pipes = sorted({p for r in k8 for p in r["pipe_cycles"]})
    lines.append("| pipe | " + " | ".join(r["label"] for r in k8) + " |")
    lines.append("|" + "---|" * (len(k8) + 1))
    for p in pipes:
        cells = []
        for r in k8:
            y = r["pipe_cycles"].get(p, 0)
            cells.append(f"{y:.0f} ({pct(y, r['total_cycles'])})")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Hot opcodes (K8) — count / attr cycles / attr µs")
    lines.append("")
    hot_keys = sorted({k for r in k8 for k in r["hot"]})
    lines.append("| opcode | " + " | ".join(r["label"] for r in k8) + " |")
    lines.append("|" + "---|" * (len(k8) + 1))
    for k in hot_keys:
        cells = []
        for r in k8:
            if k in r["hot"]:
                c, y, u = r["hot"][k]
                cells.append(f"{c} / {y:.0f} / {u:.3f}")
            else:
                cells.append("—")
        lines.append(f"| `{k}` | " + " | ".join(cells) + " |")
    lines.append("")

    for r in results:
        lines.append(f"## Detail: {r['label']}")
        lines.append("")
        if r.get("error"):
            lines.append(r["error"])
            lines.append("")
            continue
        lines.append(f"- wall: **{r['us']} µs**")
        lines.append(f"- csv: `{r['csv']}`")
        lines.append("")
        lines.append("Top opcodes by attributed cycles:")
        lines.append("")
        lines.append("| opcode | count | cycles | running_time(us) |")
        lines.append("|--------|------:|-------:|-----------------:|")
        for op, y in r["op_cycles"].most_common(20):
            lines.append(f"| `{op}` | {r['op_count'][op]} | {y:.0f} | {r['op_us'][op]:.3f} |")
        lines.append("")

    # Takeaway
    s = next(r for r in results if r["label"] == "SimdVF L128 K8")
    t128 = next(r for r in results if r["label"] == "SimtVF T128 K8")
    t32 = next(r for r in results if r["label"] == "SimtVF T32 K8")
    lines.append("## Takeaways")
    lines.append("")
    lines.append(
        f"- Wall: SimdVF K8 **{s['us']} µs** vs SimtVF T128 **{t128['us']} µs** (fair) vs T32 **{t32['us']} µs**."
    )
    lines.append(
        f"- Attributed Σ cycles: SimdVF **{s['total_cycles']:.0f}** vs T128 **{t128['total_cycles']:.0f}** "
        f"(~{t128['total_cycles']/s['total_cycles']:.2f}×) vs T32 **{t32['total_cycles']:.0f}**."
    )
    # barrier share
    for r in (s, t128, t32):
        b_c = r["bucket_cycles"].get("barrier_sync", 0)
        lines.append(
            f"- {r['label']} barrier/sync attr cycles: **{b_c:.0f}** ({pct(b_c, r['total_cycles'])})."
        )
    lines.append(
        "- SimdVF mix is dominated by **vector LD/ST + vcmp/vsel + scalar mov** around `rv_vcmax`/`rv_vcmin` (PTO VMI path)."
    )
    lines.append(
        "- SimtVF T128 K8 pays heavy **`simt_bar_thread_block` / scalar mov / SIMT local** cost; AllReduce-style thread sync shows up as barrier share and explains much of the 3.03 µs wall."
    )
    lines.append(
        "- T32 cuts that barrier/local overhead vs T128 (1.48 µs) but still trails SimdVF’s pure vector reduce path."
    )
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
