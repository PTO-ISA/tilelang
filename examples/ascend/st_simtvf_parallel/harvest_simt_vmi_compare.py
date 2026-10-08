#!/usr/bin/env python3
"""Harvest paired Simt vs VMI twin opsim into reports/ST_VMI_COMPARE.md.

Reads /tmp/st_simtvf_parallel opsim dumps (instr_exe.csv + logs) for matched
tag pairs and emits cycles / instr# / IPC / EX-pipe side-by-side tables.

IPC recipes
-----------
- Simt: IPC_proxy = body_instr / VF_SIMT.cycles
  body = all opcodes except framing (set_flag/wait_flag/end*/nop/push_pb/dcci)
  and the VF wrapper itself.
- VMI: prefer EXIPC or IFU if those instr rows exist (call_count/cycles used
  as IPC = EXIPC.call_count/EXIPC.cycles when both present; else IFU proxy).
  Fallback: IPC_proxy with VF wrapper row name "VF" (SimdVF).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/tmp/st_simtvf_parallel")
SUITE = Path(__file__).resolve().parent
REPORTS = SUITE / "reports"

# (simt_tag, vmi_tag, mode_label, short_label)
PAIRS = [
    ("sv1_e256_t32", "sv1v_e256_t64", "1", "sv1_e256_t32 ↔ sv1v_e256_t64"),
    ("sv1_e2048_t32", "sv1v_e2048_t64", "1", "sv1_e2048_t32 ↔ sv1v_e2048_t64"),
    ("sv2_r32_c32_t32_frag_live", "sv2v_r32_c32_t64_frag_live", "2", "sv2_frag_live ↔ sv2v_frag_live"),
    ("sv2_r32_c32_t32_reload", "sv2v_r32_c32_t64_reload", "1", "sv2_reload ↔ sv2v_reload"),
    ("sv3_m24_vl64_k16_t32_keep", "sv3v_m24_vl64_k16_t64_keep", "3", "sv3_m24_keep ↔ sv3v_m24_keep"),
    ("sv3_m32_vl64_k16_t32_keep", "sv3v_m32_vl64_k16_t64_keep", "3", "sv3_m32_keep ↔ sv3v_m32_keep"),
    ("sv3_m32_vl64_k16_t32_split_cm16", "sv3v_m32_vl64_k16_t64_split_cm16", "3", "sv3_split_cm16 ↔ sv3v_split_cm16"),
    ("sv4_e256_b8_t32_keep_idx", "sv4v_e256_b8_t64_keep_idx", "3", "sv4_b8_keep_idx ↔ sv4v_b8_keep_idx"),
    ("sv4_e256_b8_t32_remat_idx", "sv4v_e256_b8_t64_remat_idx", "1+3", "sv4_b8_remat_idx ↔ sv4v_b8_remat_idx"),
    ("sv4_e256_b16_t32_keep_idx", "sv4v_e256_b16_t64_keep_idx", "3", "sv4_b16_keep_idx ↔ sv4v_b16_keep_idx"),
    ("sv4_e256_b16_t32_remat_idx", "sv4v_e256_b16_t64_remat_idx", "1+3", "sv4_b16_remat_idx ↔ sv4v_b16_remat_idx"),
    ("sv5_r64_c128_g16_t32", "sv5v_r64_c128_g16_t64", "2", "sv5_g16 ↔ sv5v_g16"),
    ("sv6_r64_c128_g32_t32", "sv6v_r64_c128_g32_t64", "2", "sv6_g32 ↔ sv6v_g32"),
    ("sv7_r64_c128_g64_t32", "sv7v_r64_c128_g64_t64", "2", "sv7_g64 ↔ sv7v_g64"),
    ("sv7_r64_c128_g128_t32", "sv7v_r64_c128_g128_t64", "2", "sv7_g128 ↔ sv7v_g128"),
    ("sv8_r64_c128_g16_t32_live", "sv8v_r64_c128_g16_t64_live", "3", "sv8_live ↔ sv8v_live"),
    ("sv8_r64_c128_g16_t32_spill_dist", "sv8v_r64_c128_g16_t64_spill_dist", "1", "sv8_spill_dist ↔ sv8v_spill_dist"),
    ("sv9_e256_k1_t32_keep", "sv9v_e256_k1_t64_keep", "3", "sv9_k1_keep ↔ sv9v_k1_keep"),
    ("sv9_e256_k8_t32_keep", "sv9v_e256_k8_t64_keep", "3", "sv9_k8_keep ↔ sv9v_k8_keep"),
    ("sv9_e256_k8_t32_remat_scores", "sv9v_e256_k8_t64_remat_scores", "1+3", "sv9_k8_remat_scores ↔ sv9v_k8_remat_scores"),
    ("sv9_e256_k8_t32_remat_idx", "sv9v_e256_k8_t64_remat_idx", "3", "sv9_k8_remat_idx ↔ sv9v_k8_remat_idx"),
    # CF1–CF6 ↔ CF1V–CF6V (mask/select twins)
    ("cf1_e256_k1_t32_keep", "cf1v_e256_k1_t64_keep", "3", "cf1_k1_keep ↔ cf1v_k1_keep"),
    ("cf1_e256_k8_t32_keep", "cf1v_e256_k8_t64_keep", "3", "cf1_k8_keep ↔ cf1v_k8_keep"),
    ("cf2_e256_k8_t32_remat", "cf2v_e256_k8_t64_remat", "1", "cf2_remat ↔ cf2v_remat"),
    ("cf3_e256_k8_t32_remat_idx", "cf3v_e256_k8_t64_remat_idx", "3", "cf3_remat_idx ↔ cf3v_remat_idx"),
    ("cf4_e256_t32_pfat5", "cf4v_e256_t64_pfat5", "2", "cf4_pfat5 ↔ cf4v_pfat5"),
    ("cf4_e256_t32_pfat25", "cf4v_e256_t64_pfat25", "2", "cf4_pfat25 ↔ cf4v_pfat25"),
    ("cf5_e256_t32_pnear5", "cf5v_e256_t64_pnear5", "2", "cf5_pnear5 ↔ cf5v_pnear5"),
    ("cf5_e256_t32_pnear25", "cf5v_e256_t64_pnear25", "2", "cf5_pnear25 ↔ cf5v_pnear25"),
    ("cf6_e256_t32_pnear5", "cf6v_e256_t64_pnear5", "2", "cf6_pnear5 ↔ cf6v_pnear5"),
    ("cf6_e256_t32_pnear25", "cf6v_e256_t64_pnear25", "2", "cf6_pnear25 ↔ cf6v_pnear25"),
]

FRAMING = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}
EX_PIPE_SUBSTR = ("EX", "RVECEX")  # pipe contains these (case-insensitive match via upper)


def us_from_log(tag: str):
    log = OUT / f"opsim_{tag}.log"
    paths = []
    if log.is_file():
        paths.append(log)
    root = OUT / f"opsim_{tag}"
    if root.is_dir():
        paths.extend(root.rglob("msprof.stdout.log"))
    for p in paths:
        txt = p.read_text(errors="replace")
        for line in txt.splitlines():
            if "core0.veccore0" in line:
                nums = re.findall(r"([0-9]+\.[0-9]+)", line)
                if nums:
                    return float(nums[0])
    return None


def pass_status(tag: str) -> str:
    p = OUT / f"opsim_{tag}_result.txt"
    if p.is_file():
        t = p.read_text(errors="replace")
        if "ok=True" in t or t.strip().startswith("PASS") or "PASS" in t:
            return "PASS"
        if "ok=False" in t or "FAIL" in t:
            return "FAIL"
    log = OUT / f"opsim_{tag}.log"
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("PASS"):
                return "PASS"
            if line.startswith("FAIL"):
                return "FAIL"
    return "MISSING"


def find_instr_csv(tag: str):
    root = OUT / f"opsim_{tag}"
    if not root.is_dir():
        return None
    hits = list(root.rglob("core0.veccore0_instr_exe.csv"))
    return hits[0] if hits else None


def load_rows(tag: str):
    csv_path = find_instr_csv(tag)
    if not csv_path:
        return None, None
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8", errors="replace")))
    return rows, csv_path


def is_ex_row(instr: str, pipe: str) -> bool:
    pu = (pipe or "").upper()
    il = (instr or "").lower()
    if any(s in pu for s in EX_PIPE_SUBSTR):
        return True
    # Simt EX-ish even when pipe tags differ
    if il.startswith("simt_"):
        return True
    return False


def analyze(tag: str, side: str) -> dict:
    """side: 'simt' | 'vmi'."""
    rows, csv_path = load_rows(tag)
    us = us_from_log(tag)
    status = pass_status(tag)
    out = {
        "tag": tag,
        "side": side,
        "status": status,
        "us": us,
        "csv": str(csv_path) if csv_path else None,
        "vf_cycles": None,
        "vf_us": None,
        "body_instr": None,
        "ipc": None,
        "ipc_recipe": None,
        "ex_count": Counter(),  # opcode -> call_count
        "ex_cycles": Counter(),  # opcode -> cycles
        "ex_pipe": {},  # opcode -> pipe label
        "total_ex_count": 0,
        "total_ex_cycles": 0.0,
    }
    if not rows:
        return out

    by_instr_c = Counter()
    by_instr_y = Counter()
    by_instr_u = Counter()
    by_instr_pipe = {}
    for r in rows:
        instr = (r.get("instr") or "").strip()
        pipe = (r.get("pipe") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = float(r.get("cycles") or 0)
            ru = float(r.get("running_time(us)") or 0)
        except ValueError:
            continue
        key = instr
        by_instr_c[key] += cc
        by_instr_y[key] += cyc
        by_instr_u[key] += ru
        by_instr_pipe[key] = pipe
        if is_ex_row(instr, pipe):
            out["ex_count"][key] += cc
            out["ex_cycles"][key] += cyc
            out["ex_pipe"][key] = pipe or ("simt_*" if instr.lower().startswith("simt_") else "?")

    out["total_ex_count"] = int(sum(out["ex_count"].values()))
    out["total_ex_cycles"] = float(sum(out["ex_cycles"].values()))

    # VF wrapper + body
    vf_names = []
    if side == "simt":
        vf_names = ["VF_SIMT", "vf_simt"]
    else:
        vf_names = ["VF", "vf", "VF_SIMD", "vf_simd"]

    vf_key = None
    for name in vf_names:
        for k in by_instr_c:
            if k.lower() == name.lower():
                vf_key = k
                break
        if vf_key:
            break

    body_instr = 0
    for k, c in by_instr_c.items():
        kl = k.lower()
        if kl in FRAMING:
            continue
        if vf_key and k == vf_key:
            continue
        body_instr += c
    out["body_instr"] = body_instr

    if vf_key:
        out["vf_cycles"] = by_instr_y[vf_key]
        out["vf_us"] = by_instr_u[vf_key]

    # IPC recipe
    if side == "vmi":
        # Prefer EXIPC / IFU native counters
        exipc = next((k for k in by_instr_c if k.upper() == "EXIPC"), None)
        ifu = next((k for k in by_instr_c if k.upper() == "IFU"), None)
        if exipc and by_instr_y[exipc] > 0:
            out["ipc"] = by_instr_c[exipc] / by_instr_y[exipc]
            out["ipc_recipe"] = "EXIPC"
            out["vf_cycles"] = out["vf_cycles"] or by_instr_y[exipc]
        elif ifu and by_instr_y[ifu] > 0:
            out["ipc"] = body_instr / by_instr_y[ifu]
            out["ipc_recipe"] = "IFU"
            out["vf_cycles"] = out["vf_cycles"] or by_instr_y[ifu]
        elif out["vf_cycles"]:
            out["ipc"] = body_instr / out["vf_cycles"]
            out["ipc_recipe"] = "IPC_proxy(VF)"
        else:
            out["ipc_recipe"] = "n/a"
    else:
        if out["vf_cycles"]:
            out["ipc"] = body_instr / out["vf_cycles"]
            out["ipc_recipe"] = "IPC_proxy"
        else:
            out["ipc_recipe"] = "n/a"

    return out


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def delta(a, b):
    if a is None or b is None:
        return None
    return b - a


def ex_table(simt: dict, vmi: dict, top_n: int = 40) -> tuple[list[str], list[str]]:
    """Return (markdown lines, callout lines)."""
    ops = set(simt["ex_count"]) | set(vmi["ex_count"])
    # sort by max |Δcycles| then max cycles
    def score(op):
        sc = simt["ex_cycles"].get(op, 0)
        vc = vmi["ex_cycles"].get(op, 0)
        return (abs(vc - sc), max(sc, vc))

    ordered = sorted(ops, key=score, reverse=True)[:top_n]
    lines = [
        "| opcode | pipe | Simt call_count | Simt cycles | VMI call_count | VMI cycles | Δcount | Δcycles |",
        "|--------|------|----------------:|------------:|---------------:|-----------:|-------:|--------:|",
    ]
    callouts = []
    for op in ordered:
        sp = simt["ex_pipe"].get(op) or vmi["ex_pipe"].get(op) or "?"
        sc = simt["ex_count"].get(op, 0)
        sy = simt["ex_cycles"].get(op, 0.0)
        vc = vmi["ex_count"].get(op, 0)
        vy = vmi["ex_cycles"].get(op, 0.0)
        dc = vc - sc
        dy = vy - sy
        pipe_note = sp
        if op.lower().startswith("simt_"):
            pipe_note = f"{sp} (Simt)"
        elif sp.upper() in ("RVECEX", "EX") or "EX" in sp.upper():
            pipe_note = f"{sp} (VMI/EX)"
        lines.append(
            f"| `{op}` | {pipe_note} | {sc} | {sy:.0f} | {vc} | {vy:.0f} | {dc:+d} | {dy:+.0f} |"
        )
        if sc == 0 and vc > 0:
            callouts.append(f"- **new on VMI:** `{op}` count={vc} cycles={vy:.0f}")
        elif vc == 0 and sc > 0:
            callouts.append(f"- **removed on VMI:** `{op}` was count={sc} cycles={sy:.0f}")
        elif abs(dy) >= max(50.0, 0.2 * max(sy, vy, 1.0)):
            callouts.append(f"- **large Δcycles:** `{op}` {sy:.0f} → {vy:.0f} (Δ={dy:+.0f})")

    if not ordered:
        lines.append("| *(no EX rows)* | | | | | | | |")
    if not callouts:
        callouts.append("- *(no material EX diffs, or one side missing)*")
    return lines, callouts[:12]


def render(analyses: dict[str, dict], out_path: Path) -> str:
    lines = [
        "# Simt ↔ VMI twin compare report",
        "",
        f"**Generated:** harvest_simt_vmi_compare.py from `{OUT}`",
        f"**Suite reports dir:** `{REPORTS}`",
        "",
        "## Purpose",
        "",
        "Paired Simt vs VMI twin of the **same** ST. Mapping modes from [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md).",
        "Perf deltas = translation cost, not schedule cheating.",
        "",
        "**IPC recipes:** Simt = `IPC_proxy` (body_instr / VF_SIMT.cycles). "
        "VMI = `EXIPC` or `IFU` if present, else `IPC_proxy(VF)`.",
        "",
        "## Performance table (cycles / instr / IPC)",
        "",
        "| Tag pair | Mode | Simt µs | VMI µs | Δµs | Simt cycles | VMI cycles | Δcycles | Simt instr# | VMI instr# | Δinstr | Simt IPC | VMI IPC | ΔIPC | Simt st | VMI st |",
        "|----------|------|--------:|-------:|----:|------------:|-----------:|--------:|------------:|-----------:|-------:|---------:|--------:|-----:|---------|--------|",
    ]

    filled = 0
    for simt_tag, vmi_tag, mode, label in PAIRS:
        s = analyses.get(simt_tag) or analyze(simt_tag, "simt")
        v = analyses.get(vmi_tag) or analyze(vmi_tag, "vmi")
        analyses[simt_tag] = s
        analyses[vmi_tag] = v
        d_us = delta(s["us"], v["us"])
        d_cy = delta(s["vf_cycles"], v["vf_cycles"])
        d_in = delta(s["body_instr"], v["body_instr"])
        d_ip = delta(s["ipc"], v["ipc"])
        if s["status"] == "PASS" and v["status"] == "PASS" and s["csv"] and v["csv"]:
            filled += 1
        sipc = fmt(s["ipc"]) if s["ipc"] is not None else "—"
        vipc = fmt(v["ipc"]) if v["ipc"] is not None else "—"
        if v["ipc"] is not None and v["ipc_recipe"]:
            vipc = f"{vipc} ({v['ipc_recipe']})"
        lines.append(
            f"| {label} | {mode} | {fmt(s['us'])} | {fmt(v['us'])} | {fmt(d_us)} | "
            f"{fmt(s['vf_cycles'], 0)} | {fmt(v['vf_cycles'], 0)} | {fmt(d_cy, 0)} | "
            f"{fmt(s['body_instr'], 0)} | {fmt(v['body_instr'], 0)} | {fmt(d_in, 0)} | "
            f"{sipc} | {vipc} | {fmt(d_ip)} | {s['status']} | {v['status']} |"
        )

    lines += [
        "",
        f"**Filled pairs (both PASS + csv):** {filled} / {len(PAIRS)}",
        "",
        "## EX highlight (per pair)",
        "",
        "Filter: `pipe` contains `EX`/`RVECEX`, plus Simt `simt_*` opcodes. Δ = VMI − Simt.",
        "",
    ]

    priority = {
        "sv5_r64_c128_g16_t32",
        "sv6_r64_c128_g32_t32",
        "sv7_r64_c128_g64_t32",
        "sv7_r64_c128_g128_t32",
        "sv8_r64_c128_g16_t32_live",
        "sv8_r64_c128_g16_t32_spill_dist",
        "sv9_e256_k8_t32_keep",
        "sv9_e256_k8_t32_remat_scores",
        "sv9_e256_k8_t32_remat_idx",
        "sv9_e256_k1_t32_keep",
    }

    for simt_tag, vmi_tag, mode, label in PAIRS:
        s = analyses[simt_tag]
        v = analyses[vmi_tag]
        if simt_tag not in priority and not (s["csv"] and v["csv"]):
            continue
        lines.append(f"### {label} (mode {mode})")
        lines.append("")
        if not s["csv"] or not v["csv"]:
            lines.append("*(empty — awaiting opsim / instr_exe.csv on one or both sides)*")
            lines.append("")
            continue
        tbl, callouts = ex_table(s, v)
        lines.extend(tbl)
        lines.append("")
        lines.append(f"EX totals: Simt count={s['total_ex_count']} cycles={s['total_ex_cycles']:.0f}; "
                     f"VMI count={v['total_ex_count']} cycles={v['total_ex_cycles']:.0f}; "
                     f"Δcount={v['total_ex_count']-s['total_ex_count']:+d} "
                     f"Δcycles={v['total_ex_cycles']-s['total_ex_cycles']:+.0f}.")
        lines.append("")
        lines.append("**Top diffs:**")
        lines.extend(callouts)
        lines.append("")

    lines += [
        "## Mapping-mode reminder",
        "",
        "See [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md). Do not bolt AABBCC / token_tile / vf_fuse onto a twin.",
        "",
        "## Regenerator",
        "",
        "```bash",
        "python3 harvest_simt_vmi_compare.py --out reports/ST_VMI_COMPARE.md",
        "```",
        "",
    ]
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    # also mirror under OUT/reports if writable
    try:
        mirror = OUT / "reports" / "ST_VMI_COMPARE.md"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(text)
    except OSError:
        pass
    return text


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPORTS / "ST_VMI_COMPARE.md")
    ap.add_argument("--root", type=Path, default=Path("/tmp/st_simtvf_parallel"),
                    help="opsim root (default /tmp/st_simtvf_parallel)")
    args = ap.parse_args()
    OUT = args.root
    analyses = {}
    for simt_tag, vmi_tag, _, _ in PAIRS:
        analyses[simt_tag] = analyze(simt_tag, "simt")
        analyses[vmi_tag] = analyze(vmi_tag, "vmi")
        print(f"{simt_tag}: {analyses[simt_tag]['status']} us={analyses[simt_tag]['us']} "
              f"ipc={analyses[simt_tag]['ipc']} csv={bool(analyses[simt_tag]['csv'])}")
        print(f"{vmi_tag}: {analyses[vmi_tag]['status']} us={analyses[vmi_tag]['us']} "
              f"ipc={analyses[vmi_tag]['ipc']} recipe={analyses[vmi_tag]['ipc_recipe']} "
              f"csv={bool(analyses[vmi_tag]['csv'])}")
    text = render(analyses, args.out)
    print("WROTE", args.out, "bytes", len(text))


if __name__ == "__main__":
    main()
