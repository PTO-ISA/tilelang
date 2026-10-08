#!/usr/bin/env python3
"""Harvest Simt CF1–CF6 + VMI CF1V–CF6V opsim metrics into a summary table.

Reuses recipes from harvest_simt_vmi_compare.py:
  e2e µs      = core0.veccore0 duration from opsim log
  VF cycles   = VF_SIMT.cycles
  body_instr  = all opcodes except framing + VF_SIMT wrapper
  IPC_proxy   = body_instr / VF_SIMT.cycles
  EX instr#   = sum call_count where pipe has EX/RVECEX or instr starts with simt_
  EXIPC       = EXIPC/IFU row if present; else — (do not invent)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
OUT = Path("/tmp/st_simtvf_parallel")
REPORTS = SUITE / "reports"

# Import analyze helpers from harvest_simt_vmi_compare.py
_spec = importlib.util.spec_from_file_location(
    "harvest_simt_vmi_compare", SUITE / "harvest_simt_vmi_compare.py"
)
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

TAGS = [
    # Simt (threads=32)
    ("CF1", "cf1_e256_k1_t32_keep", "One Parallel if vs scalar thresh; scores KEEP (K=1)", "simt"),
    ("CF1", "cf1_e256_k8_t32_keep", "One Parallel if vs scalar thresh; scores KEEP (K=8)", "simt"),
    ("CF2", "cf2_e256_k8_t32_remat", "Same if; remat + kill-in-shared", "simt"),
    ("CF3", "cf3_e256_k8_t32_remat_idx", "remat_idx only (kill/select by rematted i)", "simt"),
    ("CF4", "cf4_e256_t32_pfat5", "Nested if (2-level); p_fat=5% skew", "simt"),
    ("CF4", "cf4_e256_t32_pfat25", "Nested if (2-level); p_fat=25% skew", "simt"),
    ("CF5", "cf5_e256_t32_pnear5", "Div + near-0 range branch (ULP intent); pnear=5%", "simt"),
    ("CF5", "cf5_e256_t32_pnear25", "Div + near-0 range branch (ULP intent); pnear=25%", "simt"),
    ("CF6", "cf6_e256_t32_pnear5", "Newton recip-style + near-0 / extra-iter; pnear=5%", "simt"),
    ("CF6", "cf6_e256_t32_pnear25", "Newton recip-style + near-0 / extra-iter; pnear=25%", "simt"),
    # VMI twins (lanes=64, mask/select)
    ("CF1V", "cf1v_e256_k1_t64_keep", "VMI twin: mask select thresh kill KEEP (K=1)", "vmi"),
    ("CF1V", "cf1v_e256_k8_t64_keep", "VMI twin: mask select thresh kill KEEP (K=8)", "vmi"),
    ("CF2V", "cf2v_e256_k8_t64_remat", "VMI twin: remat + kill-in-shared (mask select)", "vmi"),
    ("CF3V", "cf3v_e256_k8_t64_remat_idx", "VMI twin: remat_idx mask select", "vmi"),
    ("CF4V", "cf4v_e256_t64_pfat5", "VMI twin: nested mask select; p_fat=5%", "vmi"),
    ("CF4V", "cf4v_e256_t64_pfat25", "VMI twin: nested mask select; p_fat=25%", "vmi"),
    ("CF5V", "cf5v_e256_t64_pnear5", "VMI twin: div + near-0 mask select; pnear=5%", "vmi"),
    ("CF5V", "cf5v_e256_t64_pnear25", "VMI twin: div + near-0 mask select; pnear=25%", "vmi"),
    ("CF6V", "cf6v_e256_t64_pnear5", "VMI twin: Newton + near-0 mask select; pnear=5%", "vmi"),
    ("CF6V", "cf6v_e256_t64_pnear25", "VMI twin: Newton + near-0 mask select; pnear=25%", "vmi"),
]


def native_exipc(tag: str):
    """Return (value, recipe) if EXIPC or IFU row exists with cycles>0; else (None, None)."""
    rows, _ = _h.load_rows(tag)
    if not rows:
        return None, None
    by_c = {}
    by_y = {}
    for r in rows:
        instr = (r.get("instr") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = float(r.get("cycles") or 0)
        except ValueError:
            continue
        by_c[instr] = by_c.get(instr, 0) + cc
        by_y[instr] = by_y.get(instr, 0.0) + cyc
    for name in ("EXIPC", "IFU"):
        key = next((k for k in by_c if k.upper() == name), None)
        if key and by_y.get(key, 0) > 0:
            return by_c[key] / by_y[key], name
    return None, None


def fmt_num(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def fmt_int(v):
    if v is None:
        return "—"
    return str(int(v))


def harvest():
    rows_out = []
    for case, tag, purpose, side in TAGS:
        a = _h.analyze(tag, side)
        exipc_val, exipc_recipe = native_exipc(tag)
        rows_out.append(
            {
                "case": case,
                "tag": tag,
                "purpose": purpose,
                "status": a["status"],
                "us": a["us"],
                "vf_cycles": a["vf_cycles"],
                "body_instr": a["body_instr"],
                "ipc": a["ipc"],
                "ipc_recipe": a["ipc_recipe"],
                "ex_instr": a["total_ex_count"],
                "exipc": exipc_val,
                "exipc_recipe": exipc_recipe or "n/a",
                "csv": a["csv"],
            }
        )
    return rows_out


def md_table(rows):
    lines = []
    lines.append(
        "| case | tag | what to test | status | e2e µs | VF cyc | instr# | IPC | EX instr# | EXIPC |"
    )
    lines.append(
        "|------|-----|--------------|--------|-------:|-------:|-------:|----:|----------:|------:|"
    )
    for r in rows:
        ipc_s = fmt_num(r["ipc"], 3)
        if r["ipc"] is not None and r["ipc_recipe"]:
            ipc_s = f"{ipc_s}"
        exipc_s = fmt_num(r["exipc"], 3) if r["exipc"] is not None else "—"
        lines.append(
            "| {case} | `{tag}` | {purpose} | {status} | {us} | {vf} | {bi} | {ipc} | {ex} | {exipc} |".format(
                case=r["case"],
                tag=r["tag"],
                purpose=r["purpose"],
                status=r["status"],
                us=fmt_num(r["us"], 2),
                vf=fmt_int(r["vf_cycles"]) if r["vf_cycles"] is not None else "—",
                bi=fmt_int(r["body_instr"]) if r["body_instr"] is not None else "—",
                ipc=ipc_s,
                ex=fmt_int(r["ex_instr"]),
                exipc=exipc_s,
            )
        )
    return "\n".join(lines)


def main():
    rows = harvest()
    print("=== CF1–CF6 + CF1V–CF6V metrics ===")
    for r in rows:
        print(
            f"{r['tag']}: status={r['status']} us={r['us']} vf={r['vf_cycles']} "
            f"body={r['body_instr']} ipc={r['ipc']} ex={r['ex_instr']} "
            f"exipc={r['exipc']}({r['exipc_recipe']}) csv={r['csv']}"
        )
    print()
    print(md_table(rows))
    out_json = OUT / "cf_summary_metrics.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nWrote {out_json}")
    md_path = OUT / "cf_summary_table.md"
    note = (
        "\n\n**Metric recipes:** e2e = `core0.veccore0` wall µs; VF cyc = `VF_SIMT.cycles` (Simt) or `VF`/`EXIPC`/`IFU` (VMI); "
        "instr# = body opcodes (excl. framing `{set_flag,wait_flag,end_label,end,nop,push_pb,dcci}` + VF_SIMT); "
        "IPC = `IPC_proxy` = body_instr / VF_SIMT.cycles; EX instr# = pipe contains EX/RVECEX or `simt_*`; "
        "EXIPC = native EXIPC/IFU row if present, else `—` (SimtVF dumps usually lack EXIPC — same as ST_VMI_COMPARE).\n"
    )
    md_path.write_text(md_table(rows) + note)
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
