#!/usr/bin/env python3
"""Harvest the SV5/SV5V post-reduce scale-residency arm sweep into RESULT.md.

Usage: harvest_sv5_arms.py [OUT_DIR]   (default /tmp/st_simtvf_parallel)

Reads ``$OUT/opsim_<tag>.log`` + the opsim instr CSV for every
``sv5[v]_r*_c*_g*_t*_{keep_reg,ub_reload}`` tag found under ``$OUT/so`` or
``$OUT/opsim_*.log`` and writes:
  - ``$OUT/RESULT.md``        — per-shape keep_reg vs ub_reload winner table
  - ``$OUT/sv5_arms.tsv``     — TAG/PASS/us/IPC_proxy/maxabs rows (PASS-table feed)
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/st_simtvf_parallel")
TAG_RE = re.compile(r"^(sv5v?)_r(\d+)_c(\d+)_g(\d+)_t(\d+)_(keep_reg|ub_reload)$")
OUTSIDE = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}


def discover():
    tags = set()
    for p in (OUT / "so").glob("*.so"):
        tags.add(p.stem)
    for p in OUT.glob("opsim_*.log"):
        tags.add(p.name[len("opsim_") : -len(".log")])
    return sorted(t for t in tags if TAG_RE.match(t))


def log_text(tag):
    p = OUT / f"opsim_{tag}.log"
    return p.read_text(errors="replace") if p.is_file() else ""


def wall_us(txt):
    for line in txt.splitlines():
        if "core0.veccore0" in line or "duration_time" in line:
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums:
                return float(nums[0])
    return None


def verdict(txt):
    for line in txt.splitlines():
        if line.startswith("PASS"):
            return "PASS"
        if line.startswith("FAIL"):
            return "FAIL"
    return "MISSING"


def maxabs(txt):
    m = re.search(r"maxabs=([0-9eE.+-]+)", txt)
    return m.group(1) if m else "-"


def membar_lines(txt):
    hits = [ln.strip() for ln in txt.splitlines() if re.search(r"membar|MTE[0-9]?|barrier|set_flag|wait_flag", ln)]
    return hits[:6]


def instr_stats(tag):
    root = OUT / f"opsim_{tag}"
    hits = list(root.rglob("core0.veccore0_instr_exe.csv")) if root.is_dir() else []
    if not hits:
        return None, None, None, ""
    by_instr = Counter()
    vf_cyc = None
    vf_us = None
    body = 0
    for r in csv.DictReader(hits[0].open(newline="", encoding="utf-8", errors="replace")):
        instr = (r.get("instr") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = int(float(r.get("cycles") or 0))
            us = float(r.get("running_time(us)") or 0)
        except ValueError:
            continue
        by_instr[instr] += cc
        if instr == "VF_SIMT":
            vf_cyc, vf_us = cyc, us
        elif instr not in OUTSIDE:
            body += cc
    ipc = (body / vf_cyc) if vf_cyc else None
    top = ", ".join(f"{i}:{c}" for i, c in by_instr.most_common(8))
    return ipc, vf_us, body, top


def main():
    rows = []
    for tag in discover():
        fam, R_, C, G, T_, arm = TAG_RE.match(tag).groups()
        txt = log_text(tag)
        ipc, vf_us, body, top = instr_stats(tag)
        rows.append(
            dict(
                tag=tag, fam=fam, R=int(R_), C=int(C), G=int(G), T=int(T_), arm=arm,
                pas=verdict(txt), us=wall_us(txt), ipc=ipc, vf_us=vf_us, body=body,
                maxabs=maxabs(txt), top=top, mb=membar_lines(txt),
            )
        )

    with (OUT / "sv5_arms.tsv").open("w") as fh:
        fh.write("TAG\tPASS\tus\tIPC_proxy\tmaxabs\n")
        for r in rows:
            us = f"{r['us']:.2f}" if r["us"] is not None else "-"
            ipc = f"{r['ipc']:.3f}" if r["ipc"] is not None else "-"
            fh.write(f"{r['tag']}\t{r['pas']}\t{us}\t{ipc}\t{r['maxabs']}\n")

    L = ["# SV5 / SV5V — post-reduce inverse-scale residency sweep", "",
         f"OUT: `{OUT}`", "",
         "Arms: **keep_reg** = scale stays live in fragment/RF; **ub_reload** = scale stored",
         "to shared UB and reloaded by a later Parallel loop (implicit membar).", "",
         "| tag | PASS | wall µs | IPC_proxy | VF_SIMT µs | body_instr | maxabs |",
         "|-----|------|--------:|----------:|-----------:|-----------:|--------|"]
    for r in sorted(rows, key=lambda r: (r["fam"], r["G"], r["R"], r["arm"])):
        us = f"{r['us']:.2f}" if r["us"] is not None else "-"
        ipc = f"{r['ipc']:.3f}" if r["ipc"] is not None else "-"
        vfu = f"{r['vf_us']}" if r["vf_us"] is not None else "-"
        L.append(f"| {r['tag']} | {r['pas']} | {us} | {ipc} | {vfu} | {r['body'] or '-'} | {r['maxabs']} |")

    L += ["", "## Winner per shape", "",
          "| family | R | C | G | CG | T | keep_reg µs | ub_reload µs | Δ µs | Δ % | winner |",
          "|--------|--:|--:|--:|---:|--:|------------:|-------------:|-----:|----:|--------|"]
    shapes = {}
    for r in rows:
        shapes.setdefault((r["fam"], r["R"], r["C"], r["G"], r["T"]), {})[r["arm"]] = r
    for (fam, R_, C, G, T_), d in sorted(shapes.items()):
        k, u = d.get("keep_reg"), d.get("ub_reload")
        ku = k["us"] if k else None
        uu = u["us"] if u else None
        if ku and uu:
            dus = uu - ku
            dpc = 100.0 * dus / ku
            win = "keep_reg" if dus > 0 else ("ub_reload" if dus < 0 else "tie")
            L.append(
                f"| {fam} | {R_} | {C} | {G} | {C//G} | {T_} | {ku:.2f} | {uu:.2f} | "
                f"{dus:+.2f} | {dpc:+.1f}% | **{win}** |"
            )
        else:
            L.append(
                f"| {fam} | {R_} | {C} | {G} | {C//G} | {T_} | "
                f"{ku if ku else '-'} | {uu if uu else '-'} | - | - | incomplete |"
            )

    mb = [(r["tag"], r["mb"]) for r in rows if r["mb"]]
    if mb:
        L += ["", "## membar / MTE / flag lines seen in opsim logs", ""]
        for tag, lines in mb:
            L.append(f"- `{tag}`: " + "; ".join(f"`{x[:120]}`" for x in lines))
    L += ["", "## Top opcodes", ""]
    for r in sorted(rows, key=lambda r: r["tag"]):
        if r["top"]:
            L.append(f"- `{r['tag']}`: {r['top']}")
    txt = "\n".join(L) + "\n"
    (OUT / "RESULT.md").write_text(txt)
    print(txt)
    print("WROTE", OUT / "RESULT.md", OUT / "sv5_arms.tsv")


if __name__ == "__main__":
    main()
