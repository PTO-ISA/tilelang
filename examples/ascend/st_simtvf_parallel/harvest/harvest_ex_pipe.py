#!/usr/bin/env python3
"""RVECEX / EX-pipe instruction breakdown for matched SimdVF vs SimtVF."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path("/tmp/simd_vs_simt_perf")
OUT = Path("/tmp/simd_vs_simt_perf/EX_INSTR_BREAKDOWN.md")

ARMS = [
    ("SimdVF L128 K1", "opsim_simdvf_t128_k1"),
    ("SimdVF L128 K8", "opsim_simdvf_t128_k8"),
    ("SimtVF T128 K1", "opsim_simtvf_t128_k1"),
    ("SimtVF T128 K8", "opsim_simtvf_t128_k8"),
    ("SimtVF T32 K1", "opsim_simtvf_t32_k1"),
    ("SimtVF T32 K8", "opsim_simtvf_t32_k8"),
]

# Pipes that count as "EX" for this report
EX_PIPES = {"RVECEX", "VECTOR", "RVECSU", "RVECLP", "FLOWCTRL"}


def find_csv(arm: str):
    hits = list((ROOT / arm).rglob("core0.veccore0_instr_exe.csv"))
    return hits[0] if hits else None


def analyze(arm: str) -> dict:
    p = find_csv(arm)
    if not p:
        return {"error": "missing"}
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8", errors="replace")))
    all_pipe_c = Counter()
    all_pipe_y = Counter()
    ex_op_c = Counter()
    ex_op_y = Counter()
    ex_op_u = Counter()
    ex_pipe_c = Counter()
    ex_pipe_y = Counter()
    # also pure RVECEX only
    rvecex_op_c = Counter()
    rvecex_op_y = Counter()
    rvecex_op_u = Counter()
    total_c = total_y = 0.0
    ex_c = ex_y = ex_u = 0.0
    rvecex_c = rvecex_y = rvecex_u = 0.0
    for r in rows:
        op = (r.get("instr") or "?").strip().lower()
        pipe = (r.get("pipe") or "?").strip().upper()
        cnt = int(float(r.get("call_count") or 1))
        cyc = float(r.get("cycles") or 0)
        us = float(r.get("running_time(us)") or 0)
        all_pipe_c[pipe] += cnt
        all_pipe_y[pipe] += cyc
        total_c += cnt
        total_y += cyc
        if pipe == "RVECEX":
            rvecex_op_c[op] += cnt
            rvecex_op_y[op] += cyc
            rvecex_op_u[op] += us
            rvecex_c += cnt
            rvecex_y += cyc
            rvecex_u += us
        if pipe in EX_PIPES:
            ex_op_c[op] += cnt
            ex_op_y[op] += cyc
            ex_op_u[op] += us
            ex_pipe_c[pipe] += cnt
            ex_pipe_y[pipe] += cyc
            ex_c += cnt
            ex_y += cyc
            ex_u += us
    return {
        "csv": str(p),
        "total_c": int(total_c),
        "total_y": total_y,
        "all_pipe_c": all_pipe_c,
        "all_pipe_y": all_pipe_y,
        "ex_c": int(ex_c),
        "ex_y": ex_y,
        "ex_u": ex_u,
        "ex_op_c": ex_op_c,
        "ex_op_y": ex_op_y,
        "ex_op_u": ex_op_u,
        "ex_pipe_c": ex_pipe_c,
        "ex_pipe_y": ex_pipe_y,
        "rvecex_c": int(rvecex_c),
        "rvecex_y": rvecex_y,
        "rvecex_u": rvecex_u,
        "rvecex_op_c": rvecex_op_c,
        "rvecex_op_y": rvecex_op_y,
        "rvecex_op_u": rvecex_op_u,
    }


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main():
    results = []
    for label, arm in ARMS:
        r = analyze(arm)
        r["label"] = label
        results.append(r)
        print(label, "RVECEX", r.get("rvecex_c"), r.get("rvecex_y"), "EX*", r.get("ex_c"), r.get("ex_y"))

    k8 = [r for r in results if "K8" in r["label"]]
    lines = []
    lines.append("# EX instruction breakdown (RVECEX + related)")
    lines.append("")
    lines.append("Source: matched A/B `core0.veccore0_instr_exe.csv` under `/tmp/simd_vs_simt_perf` (2026-09-20, E=256).")
    lines.append("")
    lines.append("- **RVECEX** = pure execute pipe rows (`pipe=RVECEX`).")
    lines.append("- **EX\\*** = RVECEX + VECTOR + RVECSU + RVECLP + FLOWCTRL (broader “execute-ish” attribution; includes `set_flag`/`wait_flag` often tagged VECTOR).")
    lines.append("- Shares use attributed cycles; wall µs still from prior summary.")
    lines.append("")

    lines.append("## Pipe totals (all arms, attributed cycles)")
    lines.append("")
    lines.append("| Arm | all Σcyc | RVECEX cyc | RVECEX % | EX* cyc | EX* % | RVECLD | RVECST | MTE2 | MTE3 | SCALAR | PUSHQ |")
    lines.append("|-----|--------:|-----------:|---------:|--------:|------:|-------:|-------:|-----:|-----:|-------:|------:|")
    for r in results:
        py = r["all_pipe_y"]
        lines.append(
            f"| {r['label']} | {r['total_y']:.0f} | {r['rvecex_y']:.0f} | {pct(r['rvecex_y'], r['total_y'])} | "
            f"{r['ex_y']:.0f} | {pct(r['ex_y'], r['total_y'])} | "
            f"{py.get('RVECLD',0):.0f} | {py.get('RVECST',0):.0f} | {py.get('MTE2',0):.0f} | "
            f"{py.get('MTE3',0):.0f} | {py.get('SCALAR',0):.0f} | {py.get('PUSHQ',0):.0f} |"
        )
    lines.append("")

    lines.append("## K8 — RVECEX opcode mix (count / attr cycles / attr µs)")
    lines.append("")
    # union of top opcodes by cycles across arms
    tops = set()
    for r in k8:
        for op, _ in r["rvecex_op_y"].most_common(30):
            tops.add(op)
    # sort by max cycles across arms
    def score(op):
        return max(r["rvecex_op_y"].get(op, 0) for r in k8)
    tops = sorted(tops, key=score, reverse=True)

    lines.append("| opcode | " + " | ".join(r["label"] for r in k8) + " |")
    lines.append("|" + "---|" * (len(k8) + 1))
    for op in tops:
        cells = []
        any_nz = False
        for r in k8:
            c = r["rvecex_op_c"].get(op, 0)
            y = r["rvecex_op_y"].get(op, 0)
            u = r["rvecex_op_u"].get(op, 0)
            if c or y:
                any_nz = True
                cells.append(f"{c} / {y:.0f} / {u:.3f}")
            else:
                cells.append("—")
        if any_nz:
            lines.append(f"| `{op}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### K8 RVECEX totals")
    lines.append("")
    lines.append("| Arm | RVECEX count | RVECEX Σcyc | RVECEX Σµs | % of all cyc |")
    lines.append("|-----|-------------:|------------:|-----------:|-------------:|")
    for r in k8:
        lines.append(
            f"| {r['label']} | {r['rvecex_c']} | {r['rvecex_y']:.0f} | {r['rvecex_u']:.3f} | {pct(r['rvecex_y'], r['total_y'])} |"
        )
    lines.append("")

    # Functional buckets within RVECEX only
    RULES = [
        ("reduce", ("vcmax", "vcmin", "vmax", "vmin", "redux", "fmnmx", "imnmx", "cgmax", "cgmin")),
        ("compare", ("vcmp", "fsetp", "isetp", "setp")),
        ("select", ("vsel", "sel")),
        ("bcast_idx", ("vdup", "vbrc", "vci", "vid", "movi", "s2r")),
        ("arith_logic", ("vadd", "vsub", "vmul", "iadd", "imul", "plop3", "lop3", "lea", "shf", "and", "or", "xor")),
        ("branch_cf", ("branch", "br", "end_dvg", "diverg")),
        ("barrier", ("bar",)),
        ("misc_mov", ("mov", "smov", "cvt", "conv", "pack")),
    ]

    def buck(op):
        for name, keys in RULES:
            if any(k in op for k in keys):
                return name
        return "other"

    lines.append("## K8 — RVECEX functional buckets (attr cycles)")
    lines.append("")
    bucks = [n for n, _ in RULES] + ["other"]
    lines.append("| bucket | " + " | ".join(r["label"] for r in k8) + " |")
    lines.append("|" + "---|" * (len(k8) + 1))
    for b in bucks:
        cells = []
        any_nz = False
        for r in k8:
            y = sum(cy for op, cy in r["rvecex_op_y"].items() if buck(op) == b)
            c = sum(cn for op, cn in r["rvecex_op_c"].items() if buck(op) == b)
            if y or c:
                any_nz = True
            cells.append(f"{c} cnt / {y:.0f} cyc ({pct(y, r['rvecex_y'])})")
        if any_nz:
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-arm full RVECEX lists
    for r in results:
        lines.append(f"## Detail RVECEX: {r['label']}")
        lines.append("")
        lines.append(
            f"RVECEX {r['rvecex_c']} instr / {r['rvecex_y']:.0f} cyc / {r['rvecex_u']:.3f} µs "
            f"({pct(r['rvecex_y'], r['total_y'])} of all attr cycles)"
        )
        lines.append("")
        lines.append("| opcode | count | cycles | µs | % of RVECEX cyc |")
        lines.append("|--------|------:|-------:|---:|----------------:|")
        for op, y in r["rvecex_op_y"].most_common(40):
            lines.append(
                f"| `{op}` | {r['rvecex_op_c'][op]} | {y:.0f} | {r['rvecex_op_u'][op]:.3f} | {pct(y, r['rvecex_y'])} |"
            )
        lines.append("")

    # Takeaways
    s = next(r for r in results if r["label"] == "SimdVF L128 K8")
    t128 = next(r for r in results if r["label"] == "SimtVF T128 K8")
    t32 = next(r for r in results if r["label"] == "SimtVF T32 K8")
    lines.append("## Takeaways (EX)")
    lines.append("")
    lines.append(
        f"- RVECEX attributed cycles K8: SimdVF **{s['rvecex_y']:.0f}** vs Simt T128 **{t128['rvecex_y']:.0f}** "
        f"(~{t128['rvecex_y']/max(s['rvecex_y'],1):.2f}×) vs T32 **{t32['rvecex_y']:.0f}**."
    )
    lines.append(
        "- SimdVF EX is mostly **`rv_vcmp`/`rv_vsel`/`rv_vmax|vmin`/`rv_vcmax|vcmin`/`rv_vdups`** — tournament math on vector lanes."
    )
    lines.append(
        "- SimtVF T128 EX is mostly **`simt_redux` / `simt_plop3` / `simt_sel` / `simt_isetp|fsetp` / `simt_lea` / branches** — thread-local + AllReduce-style reduce."
    )
    lines.append(
        "- T32 shrinks EX vs T128 (fewer threads → less redux/plop3 traffic) but still has no `rv_vcmax` path."
    )
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
