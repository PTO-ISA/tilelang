#!/usr/bin/env python3
"""VF-body instruction count, wall cycles, IPC, 4-warp rooftop estimate."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/tmp/simd_vs_simt_perf")

ARMS = [
    ("SimtVF T32 K8", "opsim_simtvf_t32_k8", "VF_SIMT"),
    ("SimtVF T32 K1", "opsim_simtvf_t32_k1", "VF_SIMT"),
    ("SimtVF T128 K8", "opsim_simtvf_t128_k8", "VF_SIMT"),
    ("SimdVF L128 K8", "opsim_simdvf_t128_k8", "VF"),
    ("SimdVF L128 K1", "opsim_simdvf_t128_k1", "VF"),
]

# Ops that are outside the SIMT/SIMD VF body (host/MTE framing around asc_vf_call)
OUTSIDE = {
    "set_flag", "wait_flag", "mov_src_to_dst_alignv2", "dc_preload_xn_imm",
    "ldp_xi_xj_xn", "st_xd_xn_imm", "end_label", "end", "nop", "dcci",
    "push_pb",
}


def us_from_log(arm: str):
    log = ROOT / f"{arm}.log"
    if not log.is_file():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if "core0.veccore0" in line:
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums:
                return float(nums[0])
    return None


def load_rows(arm: str):
    p = next((ROOT / arm).rglob("core0.veccore0_instr_exe.csv"))
    return list(csv.DictReader(p.open(newline="", encoding="utf-8", errors="replace"))), str(p)


def analyze(label, arm, vf_name):
    rows, csvp = load_rows(arm)
    wall_us = us_from_log(arm)
    op_c, op_y, op_u = Counter(), Counter(), Counter()
    for r in rows:
        op = r["instr"].strip().lower()
        c = int(float(r["call_count"] or 1))
        y = float(r["cycles"] or 0)
        u = float(r["running_time(us)"] or 0)
        op_c[op] += c
        op_y[op] += y
        op_u[op] += u

    vf_key = vf_name.lower()
    vf_cyc = op_y.get(vf_key, 0.0)
    vf_us = op_u.get(vf_key, 0.0)
    # freq from VF row if both present
    freq_ghz = (vf_cyc / vf_us / 1e3) if vf_us > 0 else None  # cycles/us / 1000 = GHz

    total_c = sum(op_c.values())
    # VF-body instr: everything not in OUTSIDE and not the VF wrapper itself
    body_c = 0
    body_ops = Counter()
    for op, c in op_c.items():
        if op in OUTSIDE or op == vf_key:
            continue
        body_c += c
        body_ops[op] += c

    # also "simt_* only" for SIMT
    simt_c = sum(c for op, c in op_c.items() if op.startswith("simt_"))
    rv_c = sum(c for op, c in op_c.items() if op.startswith("rv_"))

    wall_cyc = (wall_us * freq_ghz * 1e3) if (wall_us and freq_ghz) else None
    ipc_vf = (body_c / vf_cyc) if vf_cyc else None
    ipc_wall = (body_c / wall_cyc) if wall_cyc else None
    # alternate: all non-outside including scalar mov that may be in VF
    ipc_simt_only = (simt_c / vf_cyc) if vf_cyc and simt_c else None

    return {
        "label": label,
        "csv": csvp,
        "wall_us": wall_us,
        "vf_name": vf_name,
        "vf_cyc": vf_cyc,
        "vf_us": vf_us,
        "freq_ghz": freq_ghz,
        "wall_cyc": wall_cyc,
        "total_instr": total_c,
        "body_instr": body_c,
        "simt_instr": simt_c,
        "rv_instr": rv_c,
        "ipc_body_over_vf": ipc_vf,
        "ipc_body_over_wall": ipc_wall,
        "ipc_simt_over_vf": ipc_simt_only,
        "top_body": body_ops.most_common(12),
        "outside_c": sum(op_c[o] for o in OUTSIDE if o in op_c) + op_c.get(vf_key, 0),
    }


def main():
    results = [analyze(*a) for a in ARMS]
    lines = []
    lines.append("# SimtVF / SimdVF — instruction count, VF wall cycles, IPC, 4-warp rooftop")
    lines.append("")
    lines.append("VF cycles/µs from `instr_exe` row `VF_SIMT` or `VF`. Body instr = all opcodes except MTE/flag framing + the VF wrapper itself.")
    lines.append("")
    lines.append("| Arm | wall µs | VF µs | VF cyc | freq GHz | body instr | simt_/rv_ | IPC body/VF | IPC body/wall |")
    lines.append("|-----|--------:|------:|-------:|---------:|-----------:|----------:|------------:|--------------:|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['wall_us']} | {r['vf_us']:.3f} | {r['vf_cyc']:.0f} | "
            f"{r['freq_ghz']:.3f} | {r['body_instr']} | {r['simt_instr'] or r['rv_instr']} | "
            f"{r['ipc_body_over_vf']:.3f} | {r['ipc_body_over_wall']:.3f} |"
        )
    lines.append("")

    # Focus T32 K8 rooftop
    t32 = next(r for r in results if r["label"] == "SimtVF T32 K8")
    simd = next(r for r in results if r["label"] == "SimdVF L128 K8")
    lines.append("## T32 K8 numbers (primary)")
    lines.append("")
    lines.append(f"- wall: **{t32['wall_us']} µs** ≈ **{t32['wall_cyc']:.0f} cycles** @ {t32['freq_ghz']:.3f} GHz")
    lines.append(f"- `VF_SIMT` wall: **{t32['vf_us']:.3f} µs** = **{t32['vf_cyc']:.0f} cycles**")
    lines.append(f"- instructions in VF body (excl. MTE/flags/VF op): **{t32['body_instr']}**")
    lines.append(f"- of which `simt_*`: **{t32['simt_instr']}**")
    lines.append(f"- **IPC ≈ body_instr / VF_cyc = {t32['body_instr']} / {t32['vf_cyc']:.0f} = {t32['ipc_body_over_vf']:.3f}**")
    lines.append(f"- if count only `simt_*`: {t32['simt_instr']} / {t32['vf_cyc']:.0f} = {t32['ipc_simt_over_vf']:.3f}")
    lines.append("")
    lines.append("Top body opcodes:")
    for op, c in t32["top_body"]:
        lines.append(f"- `{op}` × {c}")
    lines.append("")

    lines.append("## 4-warp rooftop estimate (independent tokens)")
    lines.append("")
    lines.append("Assumption: 4 tokens × `SimtVF(32)` = 4 independent warps; same per-token body; perfect compute overlap; no UB conflict.")
    lines.append("")
    cyc = t32["vf_cyc"]
    n = t32["body_instr"]
    ipc = t32["ipc_body_over_vf"]
    lines.append(f"- Per token today: **{n} instr / {cyc:.0f} VF cyc** → IPC **{ipc:.3f}** (issue slots largely idle / dependency-bound)")
    lines.append(f"- If 4 warps fill issue to IPC≈1.0 on same VF cyc budget: need ~{cyc:.0f} issued/cyc×1 ≈ {cyc:.0f} instr in flight — we only have {n} per token, so **headroom from IPC alone ≈ {1.0/ipc:.2f}×** on this warp’s issue")
    lines.append(f"- Occupancy rooftop on **throughput**: 4 tokens in ~same {cyc:.0f} cyc window → **~{cyc/4:.0f} cyc/token** ≈ **{t32['vf_us']/4:.3f} µs/token** VF-only")
    lines.append(f"- Plus framing: wall today {t32['wall_us']} → ideal 4-way **~{t32['wall_us']/4:.3f} µs/token** throughput")
    lines.append(f"- Compare SimdVF single-token: wall **{simd['wall_us']} µs**, VF **{simd['vf_us']:.3f} µs** / **{simd['vf_cyc']:.0f} cyc**, body instr **{simd['body_instr']}**, IPC body/VF **{simd['ipc_body_over_vf']:.3f}**")
    lines.append("")
    lines.append("Caveat: raising IPC toward 1 on **one** T32 warp needs ILP inside that warp (independent ops), not just 4 warps — 4 warps help **machine throughput**, each warp can still sit at IPC~0.4.")
    lines.append("T128 single-token is NOT this rooftop (cooperative AllReduce).")
    lines.append("")

    out = ROOT / "VF_IPC_ROOFTOP.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("WROTE", out)


if __name__ == "__main__":
    main()
