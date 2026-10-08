#!/usr/bin/env python3
"""Harvest the SV5/SV6/SV7 (+VMI twins) reduce-group RF-capacity ladder into RESULT.md.

Usage: harvest_sv567_rfladder.py [OUT_DIR]   (default /tmp/st_simtvf_parallel)

Reads ``$OUT/opsim_<tag>.log`` + the opsim instr CSV for every ladder tag found
under ``$OUT/so`` or ``$OUT/opsim_*.log`` and writes:
  - ``$OUT/RESULT.md``            — per-rung keep vs stream/reload/multiwarp table
  - ``$OUT/sv567_rfladder.tsv``   — TAG/PASS/us/IPC_proxy/maxabs rows (PASS-table feed)

Arms per rung:
  sv5  G=16  keep_in_rf   vs ub_stream   (group inputs in RF vs streamed from UB)
  sv6  G=32  keep_in_warp vs reload      (one-warp keep vs UB reload)
  sv7  G≥64  multiwarp_ub vs reload      (UB partial allreduce vs single stream)
Historical arms keep_reg / ub_reload (post-reduce *scale* residency) are listed
in the raw table when present but are not paired as winners.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import harvest_vf_theory as TH  # VF cycle theory model (additive)
except Exception:  # pragma: no cover
    TH = None

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/st_simtvf_parallel")
OUT.mkdir(parents=True, exist_ok=True)
ARMS = "keep_in_rf|ub_stream|keep_in_warp|reload|multiwarp_ub|keep_reg|ub_reload"
TAG_RE = re.compile(rf"^(sv[567]v?)_r(\d+)_c(\d+)_g(\d+)_t(\d+)_({ARMS})$")
PAIRS = {"sv5": ("keep_in_rf", "ub_stream"),
         "sv6": ("keep_in_warp", "reload"),
         "sv7": ("multiwarp_ub", "reload")}
OUTSIDE = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}


def discover():
    tags = set()
    for p in (OUT / "so").glob("*.so"):
        tags.add(p.stem)
    for p in OUT.glob("opsim_*.log"):
        tags.add(p.name[len("opsim_"):-len(".log")])
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


def verdict(tag, txt):
    for line in txt.splitlines():
        if line.startswith("PASS"):
            return "PASS"
        if line.startswith("FAIL"):
            return "FAIL"
    cl = OUT / "logs" / f"compile_{tag}.log"
    if not (OUT / "so" / f"{tag}.so").is_file() and cl.is_file():
        return "COMPILE_FAIL"
    return "MISSING"


def compile_reason(tag):
    cl = OUT / "logs" / f"compile_{tag}.log"
    if not cl.is_file():
        return ""
    hits = [ln.strip() for ln in cl.read_text(errors="replace").splitlines()
            if re.search(r"Error|error:|Unsupported|Verify|InternalError|AttributeError|RuntimeError|local\.var|mask_and", ln)]
    return hits[-1][:300] if hits else ""


def maxabs(txt):
    m = re.search(r"maxabs=([0-9eE.+-]+)", txt)
    return m.group(1) if m else "-"


def instr_stats(tag):
    root = OUT / f"opsim_{tag}"
    hits = list(root.rglob("core0.veccore0_instr_exe.csv")) if root.is_dir() else []
    if not hits:
        return None, None, None, ""
    by_instr = Counter()
    vf_cyc = vf_us = None
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
        if instr in ("VF_SIMT", "VF_SIMD"):
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
        rows.append(dict(tag=tag, fam=fam, R=int(R_), C=int(C), G=int(G), T=int(T_),
                         arm=arm, pas=verdict(tag, txt), us=wall_us(txt), ipc=ipc,
                         vf_us=vf_us, body=body, maxabs=maxabs(txt), top=top,
                         why=compile_reason(tag)))

    with (OUT / "sv567_rfladder.tsv").open("w") as fh:
        fh.write("TAG\tPASS\tus\tIPC_proxy\tmaxabs\n")
        for r in rows:
            us = f"{r['us']:.2f}" if r["us"] is not None else "-"
            ipc = f"{r['ipc']:.3f}" if r["ipc"] is not None else "-"
            fh.write(f"{r['tag']}\t{r['pas']}\t{us}\t{ipc}\t{r['maxabs']}\n")

    L = ["# SV5–SV7 (+VMI twins) — reduce-group RF-capacity ladder", "",
         f"OUT: `{OUT}`", "",
         "Primary sensitivity: **group-input RF capacity during absmax** "
         "(not post-reduce scale residency).", "",
         "| rung | arms |", "|------|------|",
         "| sv5 G=16 | `keep_in_rf` (inputs preloaded into fragment/RF) vs `ub_stream` (stream from UB) |",
         "| sv6 G=32 | `keep_in_warp` (one-warp RF keep, no UB partials) vs `reload` (stream from UB) |",
         "| sv7 G=64/128 | `multiwarp_ub` (NW=4 chunk partials via UB allreduce) vs `reload` (single stream) |",
         "", "## All tags", "",
         "| tag | PASS | wall µs | IPC_proxy | VF µs | body_instr | maxabs |",
         "|-----|------|--------:|----------:|------:|-----------:|--------|"]
    for r in sorted(rows, key=lambda r: (r["fam"], r["G"], r["R"], r["arm"])):
        us = f"{r['us']:.2f}" if r["us"] is not None else "-"
        ipc = f"{r['ipc']:.3f}" if r["ipc"] is not None else "-"
        vfu = f"{r['vf_us']}" if r["vf_us"] is not None else "-"
        L.append(f"| {r['tag']} | {r['pas']} | {us} | {ipc} | {vfu} | {r['body'] or '-'} | {r['maxabs']} |")

    L += ["", "## Winner per rung / shape", "",
          "| family | R | C | G | CG | T | keep arm | keep µs | foil arm | foil µs | Δ µs | Δ % | winner |",
          "|--------|--:|--:|--:|---:|--:|----------|--------:|----------|--------:|-----:|----:|--------|"]
    shapes = {}
    for r in rows:
        shapes.setdefault((r["fam"], r["R"], r["C"], r["G"], r["T"]), {})[r["arm"]] = r
    for (fam, R_, C, G, T_), d in sorted(shapes.items()):
        ka, fa = PAIRS[fam.rstrip("v")]
        k, f = d.get(ka), d.get(fa)
        ku = k["us"] if k and k["pas"] == "PASS" else None
        fu = f["us"] if f and f["pas"] == "PASS" else None
        if ku and fu:
            dus = fu - ku
            dpc = 100.0 * dus / ku
            win = ka if dus > 0 else (fa if dus < 0 else "tie")
            L.append(f"| {fam} | {R_} | {C} | {G} | {C//G} | {T_} | {ka} | {ku:.2f} | {fa} | "
                     f"{fu:.2f} | {dus:+.2f} | {dpc:+.1f}% | **{win}** |")
        else:
            L.append(f"| {fam} | {R_} | {C} | {G} | {C//G} | {T_} | {ka} | "
                     f"{f'{ku:.2f}' if ku else (k['pas'] if k else '-')} | {fa} | "
                     f"{f'{fu:.2f}' if fu else (f['pas'] if f else '-')} | - | - | incomplete |")

    if TH is not None:
        L += ["", "## VF cycle theory vs measured", "",
              f"`C_pred = O_arm + ceil((W/T) / I_assum)` with I_assum(Simt)={TH.IPC_SIMT}, "
              f"I_assum(VMI)={TH.IPC_VMI}, O_base={TH.O_BASE:.0f}, O_loop={TH.O_LOOP:.0f}, "
              f"O_ub_roundtrip={TH.O_UBRT:.0f}, freq={TH.FREQ_HZ/1e9:.2f} GHz. "
              "Assumptions and the a5/a6 re-use plan: `reports/ST_VF_THEORY.md`.", ""]
        trows = TH.rows_for(OUT, [r["tag"] for r in rows])
        L += TH.md_table(trows)
        fi = [t["freq_implied"] for t in trows if t["freq_implied"]]
        if fi:
            L += ["", f"Implied camodel frequency (VF cycles / VF running_time): "
                      f"{min(fi)/1e9:.3f}-{max(fi)/1e9:.3f} GHz."]

    bad = [r for r in rows if r["pas"] != "PASS"]
    if bad:
        L += ["", "## Non-PASS tags (reasons, no invented µs)", ""]
        for r in bad:
            L.append(f"- `{r['tag']}` → **{r['pas']}** {('— ' + r['why']) if r['why'] else ''}")
    L += ["", "## Top opcodes", ""]
    for r in sorted(rows, key=lambda r: r["tag"]):
        if r["top"]:
            L.append(f"- `{r['tag']}`: {r['top']}")
    txt = "\n".join(L) + "\n"
    (OUT / "RESULT.md").write_text(txt)
    print(txt)
    print("WROTE", OUT / "RESULT.md", OUT / "sv567_rfladder.tsv")


if __name__ == "__main__":
    main()
