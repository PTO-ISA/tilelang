#!/usr/bin/env python3
"""Harvest opsim µs + instr mix + layout notes into reports/ and SUMMARY.md."""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

# OUT/REPORTS are overridable so a side job (e.g. the SV5 arm sweep under
# /tmp/pr272_sv5_arms_*) can harvest into its own dir without clobbering the
# shared /tmp/t_parallel_st_suite run or the suite reports/.
OUT = Path(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))
REPORTS = Path(os.environ.get("ST_SIMTVF_REPORTS", str(Path(__file__).resolve().parent / "reports")))
REPORTS.mkdir(parents=True, exist_ok=True)

PRIOR_SV2 = {
    ("sv2", 256, 1, 32): 1.17,
    ("sv2", 256, 8, 32): 1.48,
    ("sv2", 256, 1, 128): 1.18,
    ("sv2", 256, 8, 128): 3.03,
}


def us_from_log(tag: str):
    log = OUT / f"opsim_{tag}.log"
    if not log.is_file():
        # also search opsim dir msprof
        for p in (OUT / f"opsim_{tag}").rglob("msprof.stdout.log"):
            log = p
            break
        else:
            return None
    txt = log.read_text(errors="replace")
    for line in txt.splitlines():
        if "core0.veccore0" in line or "duration_time" in line:
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums:
                return float(nums[0])
    return None


def pass_from_result(tag: str):
    p = OUT / f"opsim_{tag}_result.txt"
    if not p.is_file():
        # grep log
        log = OUT / f"opsim_{tag}.log"
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("PASS"):
                    return "PASS"
                if line.startswith("FAIL"):
                    return "FAIL"
        return "MISSING"
    t = p.read_text()
    if "ok=True" in t or t.startswith("PASS") or " ok=True" in t:
        return "PASS"
    if "ok=False" in t:
        return "FAIL"
    if "PASS" in t:
        return "PASS"
    if "FAIL" in t:
        return "FAIL"
    return "UNKNOWN"


def find_instr_csv(tag: str):
    root = OUT / f"opsim_{tag}"
    if not root.is_dir():
        return None
    hits = list(root.rglob("core0.veccore0_instr_exe.csv"))
    return hits[0] if hits else None


def layout_notes(tag: str) -> str:
    src = OUT / "sources" / f"{tag}_source.txt"
    if not src.is_file():
        return "(no source dump)"
    txt = src.read_text(errors="replace")
    notes = []
    m = re.search(r"__launch_bounds__\((\d+)\)", txt)
    if m:
        notes.append(f"launch_bounds={m.group(1)}")
    m = re.search(r"asc_vf_call\([^)]*dim3\((\d+)\)", txt)
    if m:
        notes.append(f"asc_vf_threads={m.group(1)}")
    # fragment local arrays like scores[8] / scores[2]
    for name in ("scores", "idxs", "idx_cand", "t1", "v", "tile", "row_max", "x", "amax", "acc", "acc_c", "Acc"):
        mm = re.findall(rf"\b{name}\[(\d+)\]", txt)
        if mm:
            notes.append(f"{name}[] sizes={sorted(set(mm), key=lambda s: int(s))[:6]}")
    # also catch float Acc[...] declarations in CCE
    for mm in re.finditer(r"\b(?:float|half|__fp16|uint)\s+(Acc|acc|acc_c)\[(\d+)\]", txt):
        notes.append(f"{mm.group(1)}[{mm.group(2)}]")
    tir = OUT / "sources" / f"{tag}_tir.txt"
    if tir.is_file():
        notes.append(f"tir={tir.name}")
    if (OUT / "sources" / f"{tag}_source.txt").is_file():
        notes.append(f"src={tag}_source.txt")
    if "AscendAllReduce" in txt:
        notes.append("has AscendAllReduce")
    if "simt_redux" in txt or "AllReduce" in txt:
        notes.append("cross-thread reduce path")
    # elems/thread hint from tag
    m = re.search(r"_e(\d+).*_t(\d+)", tag) or re.search(r"_n(\d+)_t(\d+)", tag)
    if m:
        n, t = int(m.group(1)), int(m.group(2))
        if t:
            notes.append(f"elems_per_thread≈{n/t:.2f}")
    return "; ".join(notes) if notes else "(parse sparse)"


def instr_top(tag: str, k: int = 12) -> str:
    csv_path = find_instr_csv(tag)
    if not csv_path:
        return "(no instr csv)"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8", errors="replace")))
    by_instr = Counter()
    by_pipe = Counter()
    vf_us = vf_cyc = None
    body_instr = 0
    body_cyc = 0
    outside = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}
    for r in rows:
        instr = (r.get("instr") or "").strip()
        pipe = (r.get("pipe") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = int(float(r.get("cycles") or 0))
            us = float(r.get("running_time(us)") or 0)
        except ValueError:
            continue
        by_instr[instr] += cc
        by_pipe[pipe] += cc
        if instr == "VF_SIMT":
            vf_us, vf_cyc = us, cyc
        elif instr not in outside:
            body_instr += cc
            body_cyc += cyc
    lines = []
    if vf_us is not None:
        ipc = (body_instr / vf_cyc) if vf_cyc else 0
        lines.append(f"VF_SIMT us={vf_us} cyc={vf_cyc} body_instr={body_instr} IPC_proxy={ipc:.3f}")
    lines.append("top_opcodes: " + ", ".join(f"{i}:{c}" for i, c in by_instr.most_common(k)))
    lines.append("pipes: " + ", ".join(f"{p}:{c}" for p, c in by_pipe.most_common(8)))
    return " | ".join(lines)


def discover_tags():
    tags = set()
    for p in (OUT / "so").glob("*.so"):
        tags.add(p.stem)
    for p in OUT.glob("opsim_*_result.txt"):
        tags.add(p.name[len("opsim_") : -len("_result.txt")])
    for p in OUT.glob("compile_*.log"):
        # compile_sv2_e256_k8_t32.log
        tags.add(p.stem[len("compile_") :])
    return sorted(tags)


def main():
    rows = []
    for tag in discover_tags():
        status = pass_from_result(tag)
        us = us_from_log(tag)
        layout = layout_notes(tag)
        mix = instr_top(tag)
        # prior cite
        prior = ""
        m = re.match(r"(sv\d+)_e(\d+)_k(\d+)_t(\d+)", tag)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            if key in PRIOR_SV2:
                prior = f"prior_AB={PRIOR_SV2[key]}us"
        # per-tag report
        rep = REPORTS / f"{tag}.md"
        rep.write_text(
            f"# {tag}\n\n"
            f"- status: **{status}**\n"
            f"- wall_us: {us}\n"
            f"- {prior}\n"
            f"- layout: {layout}\n"
            f"- instr: {mix}\n"
        )
        rows.append((tag, status, us, prior, layout, mix))

    # SUMMARY table
    lines = [
        "# SimtVF Parallel ST suite — SUMMARY",
        "",
        f"Remote OUT: `{OUT}`",
        "",
        "| tag | status | µs | prior/notes | layout |",
        "|-----|--------|----|-------------|--------|",
    ]
    for tag, status, us, prior, layout, mix in rows:
        us_s = f"{us:.3f}" if isinstance(us, float) else "-"
        note = prior or mix[:60].replace("|", "/")
        lines.append(f"| {tag} | {status} | {us_s} | {note} | {layout[:80]} |")
    lines += [
        "",
        "## Priority matrix (SV2)",
        "",
        "Expect T32 K8 ≈ 1.48µs, T128 K8 ≈ 3.03µs (prior green AB).",
        "",
        "## Per-tag reports",
        "",
    ]
    for tag, *_ in rows:
        lines.append(f"- [`reports/{tag}.md`]({tag}.md)")
    summary = "\n".join(lines) + "\n"
    (REPORTS / "SUMMARY.md").write_text(summary)
    # also mirror under OUT
    (OUT / "SUMMARY.md").write_text(summary)
    print(summary)
    print("WROTE", REPORTS / "SUMMARY.md")


if __name__ == "__main__":
    main()
