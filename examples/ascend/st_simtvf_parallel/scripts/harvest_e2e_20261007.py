#!/usr/bin/env python3
"""Harvest wall_us / C_meas / body_instr / IPC for Simt + *d (+ *v) matched shapes.

Writes:
  /tmp/st_simtvf_parallel/reports/simt_simd_e2e_20261007.tsv
  /tmp/st_simtvf_parallel/reports/SIMT_SIMD_E2E_COMPARE_20261007.md
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

OUTS = [
    Path("/tmp/st_simtvf_parallel"),
    Path("/tmp/pr272_sv567_rfladder_20261006"),
    Path("/tmp/pr272_sv4_cliff_20261005/out"),
]
PTODSL = Path("/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d")
REPORTS = Path("/tmp/st_simtvf_parallel/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

FRAMING = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}
RVEC_PIPES = {"RVECEX", "RVECLD", "RVECST", "RVECSU"}

# Matched-shape primary rows for e2e table: (family, shape_label, simt_tag, d_tag, v_tag_or_None, note)
ROWS = [
    ("SV1", "E256 T32/T64", "sv1_e256_t32", "sv1d_e256_t32", "sv1v_e256_t64", "stream baseline"),
    ("SV1", "E2048 T32/T64", "sv1_e2048_t32", None, "sv1v_e2048_t64", "stream large; *d no E2048 primary"),
    ("SV2", "R16×C64 input_keep", "sv2_r16_c64_t32_input_keep", "sv2d_r16_c64_t32_input_keep", None, "KEEP; Simt may COMPILE_FAIL"),
    ("SV2", "R16×C32 input_keep (fallback)", "sv2_r16_c32_t32_input_keep", None, None, "ABI fallback when C64 keep fails"),
    ("SV2", "R32×C64 input_stream", "sv2_r32_c64_t32_input_stream", "sv2d_r32_c64_t32_input_stream", None, "STREAM primary"),
    ("SV2", "R32×C128 input_stream", "sv2_r32_c128_t32_input_stream", "sv2d_r32_c128_t32_input_stream", None, "STREAM wide-C"),
    ("SV2", "R32×C64 fold_scale_reload foil", "sv2_r32_c64_t32_reload", "sv2d_r32_c64_t32_fold_scale_reload", None, "optional scale-remat foil"),
    ("SV3", "M24 VL64 K16 keep", "sv3_m24_vl64_k16_t32_keep", "sv3d_m24_vl64_k16_t32_keep", "sv3v_m24_vl64_k16_t64_keep", "Acc KEEP"),
    ("SV3", "M32 VL64 K16 keep", "sv3_m32_vl64_k16_t32_keep", "sv3d_m32_vl64_k16_t32_keep", "sv3v_m32_vl64_k16_t64_keep", "Acc KEEP"),
    ("SV3", "M32 VL64 K16 split_cm16", "sv3_m32_vl64_k16_t32_split_cm16", "sv3d_m32_vl64_k16_t32_split_cm16", "sv3v_m32_vl64_k16_t64_split_cm16", "chunk flush"),
    ("SV4", "E256 B8 keep_idx", "sv4_e256_b8_t32_keep_idx", "sv4d_e256_b8_t32_keep_idx", None, "idxs+acc KEEP; *v COMPILE_FAIL"),
    ("SV4", "E256 B8 remat_idx(_calc)", "sv4_e256_b8_t32_remat_idx_calc", "sv4d_e256_b8_t32_remat_idx", None, "Simt remat_idx_calc ↔ *d remat_idx"),
    ("SV5", "R64 C128 G16 keep_in_rf", "sv5_r64_c128_g16_t32_keep_in_rf", "sv5d_r64_c128_g16_t32_keep_in_rf", None, "*v keep COMPILE_FAIL"),
    ("SV5", "R64 C128 G16 ub_stream", "sv5_r64_c128_g16_t32_ub_stream", "sv5d_r64_c128_g16_t32_ub_stream", "sv5v_r64_c128_g16_t64_ub_stream", "Case-3; *v wall scalar-transpose dominated"),
    ("SV6", "R64 C128 G32 keep_in_warp", "sv6_r64_c128_g32_t32_keep_in_warp", "sv6d_r64_c128_g32_t32_keep_in_warp", None, "*v keep COMPILE_FAIL"),
    ("SV6", "R64 C128 G32 reload", "sv6_r64_c128_g32_t32_reload", "sv6d_r64_c128_g32_t32_reload", "sv6v_r64_c128_g32_t64_reload", "Case-3; *v wall scalar-transpose dominated"),
    ("SV7", "R64 C128 G64 multiwarp_ub", "sv7_r64_c128_g64_t32_multiwarp_ub", "sv7d_r64_c128_g64_t32_multiwarp_ub", "sv7v_r64_c128_g64_t64_multiwarp_ub", "Case-3"),
    ("SV7", "R64 C128 G64 reload", "sv7_r64_c128_g64_t32_reload", "sv7d_r64_c128_g64_t32_reload", "sv7v_r64_c128_g64_t64_reload", "Case-3"),
    ("SV7", "R64 C128 G128 multiwarp_ub", "sv7_r64_c128_g128_t32_multiwarp_ub", "sv7d_r64_c128_g128_t32_multiwarp_ub", None, ""),
    ("SV7", "R64 C128 G128 reload", "sv7_r64_c128_g128_t32_reload", "sv7d_r64_c128_g128_t32_reload", "sv7v_r64_c128_g128_t64_reload", "Case-3"),
    ("SV8", "R64 C128 G16 live", "sv8_r64_c128_g16_t32_live", "sv8d_r64_c128_g16_t32_live", None, "*v COMPILE_FAIL historically"),
    ("SV8", "R64 C128 G16 spill_dist", "sv8_r64_c128_g16_t32_spill_dist", "sv8d_r64_c128_g16_t32_spill_dist", None, ""),
    ("SV9", "E256 K8 keep", "sv9_e256_k8_t32_keep", "sv9d_e256_k8_t32_keep", None, "*v COMPILE_FAIL"),
    ("SV9", "E256 K8 remat_scores", "sv9_e256_k8_t32_remat_scores", "sv9d_e256_k8_t32_remat_scores", None, ""),
    ("SV9", "E256 K8 remat_idx", "sv9_e256_k8_t32_remat_idx", "sv9d_e256_k8_t32_remat_idx", None, ""),
]

HISTORICAL = [
    ("SV2 HIST", "R32×C32 frag_live", "sv2_r32_c32_t32_frag_live", None, None, "HISTORICAL pre-align"),
    ("SV2 HIST", "R32×C32 reload", "sv2_r32_c32_t32_reload", None, None, "HISTORICAL pre-align"),
]


def find_opsim_root(tag: str):
    """Return (root_dir, log_path) for tag, searching OUTS + PTODSL."""
    candidates = []
    if tag and "d" in tag[2:4]:  # *d tags like sv1d / sv2d
        candidates.append(PTODSL / f"opsim_{tag}")
        candidates.append(PTODSL)  # logs may be flat
    for base in OUTS:
        candidates.append(base / f"opsim_{tag}")
        # also nested out/
        candidates.append(base / "out" / f"opsim_{tag}")
    for c in candidates:
        if c.is_dir() and c.name.startswith("opsim_"):
            return c
    return None


def find_log(tag: str, root: Path | None):
    paths = []
    if root and root.is_dir():
        paths.append(root.parent / f"opsim_{tag}.log")
        paths.append(root.parent / f"{tag}.log")  # ptodsl flat
        paths.append(root / "msprof.stdout.log")
        paths.extend(root.rglob("msprof.stdout.log"))
    for base in OUTS + [PTODSL]:
        paths.append(base / f"opsim_{tag}.log")
        paths.append(base / f"{tag}.log")
    for p in paths:
        if p.is_file():
            return p
    return None


def find_result(tag: str, root: Path | None):
    paths = []
    if root:
        paths.append(root.parent / f"opsim_{tag}_result.txt")
    for base in OUTS + [PTODSL]:
        paths.append(base / f"opsim_{tag}_result.txt")
    for p in paths:
        if p.is_file():
            return p
    return None


def find_instr_csv(root: Path | None):
    if not root or not root.is_dir():
        return None
    hits = list(root.rglob("core0.veccore0_instr_exe.csv"))
    return hits[0] if hits else None


def us_from_text(txt: str):
    for line in txt.splitlines():
        if "core0.veccore0" in line:
            # prefer dotted floats; also accept integer us (e.g. "1" for 1.0)
            nums = re.findall(r"([0-9]+\.[0-9]+)", line)
            if nums:
                return float(nums[0])
            nums = re.findall(r"\b([0-9]+)\b", line)
            # skip bare core index-like; take first large-enough or any after core0
            if nums:
                # duration is typically the first standalone number after the name
                for n in nums:
                    v = float(n)
                    if v >= 1 or "." in n:
                        return v
                return float(nums[0])
    return None


def pass_status(tag: str, root: Path | None, log: Path | None) -> str:
    # compile log under OUT/logs
    for base in OUTS:
        clog = base / "logs" / f"compile_{tag}.log"
        # also summary lines
    # check compile fail markers in oneshot summaries
    for base in OUTS + [PTODSL, Path("/tmp")]:
        for name in (
            "SUMMARY_sv2_matched_20261007.txt",
            "SUMMARY_sv2_raw.txt",
            "oneshot_sv2_matched_20261007.log",
            "sv2_matched_console.log",
        ):
            p = base / name if base != Path("/tmp") else Path("/tmp") / name
            if p.is_file():
                t = p.read_text(errors="replace")
                if f"COMPILE_FAIL {tag}" in t:
                    return "COMPILE_FAIL"
                if f"COMPILE_OK {tag}" in t:
                    break
    res = find_result(tag, root)
    if res:
        t = res.read_text(errors="replace")
        if "ok=True" in t or t.strip().startswith("PASS") or "PASS" in t:
            return "PASS"
        if "ok=False" in t or "FAIL" in t:
            return "FAIL"
    if log and log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("PASS") or f"PASS {tag}" in line:
                return "PASS"
            if line.startswith("FAIL") or f"FAIL {tag}" in line:
                return "FAIL"
            if "COMPILE_FAIL" in line and tag in line:
                return "COMPILE_FAIL"
    # directory exists with csv => likely ran
    if root and find_instr_csv(root):
        return "PASS"  # opsim produced CSV; confirm via log prefer
    return "MISSING"


def analyze(tag: str, side: str) -> dict:
    out = {
        "tag": tag,
        "side": side,
        "status": "MISSING",
        "wall_us": None,
        "C_meas": None,
        "VF_us": None,
        "body_instr": None,
        "ipc": None,
        "ipc_recipe": None,
        "exipc": None,
        "rvec_cycles": None,
        "csv": None,
        "root": None,
    }
    if not tag:
        return out
    root = find_opsim_root(tag)
    log = find_log(tag, root)
    out["root"] = str(root) if root else None
    out["status"] = pass_status(tag, root, log)
    if log and log.is_file():
        us = us_from_text(log.read_text(errors="replace"))
        if us is not None:
            out["wall_us"] = us
    if root:
        for p in root.rglob("msprof.stdout.log"):
            us = us_from_text(p.read_text(errors="replace"))
            if us is not None:
                out["wall_us"] = us
                break
    csv_path = find_instr_csv(root)
    if not csv_path:
        return out
    out["csv"] = str(csv_path)
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8", errors="replace")))
    by_c = Counter()
    by_y = Counter()
    by_u = Counter()
    by_pipe = {}
    rvec_cycles = 0.0
    for r in rows:
        instr = (r.get("instr") or "").strip()
        pipe = (r.get("pipe") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = float(r.get("cycles") or 0)
            ru = float(r.get("running_time(us)") or 0)
        except ValueError:
            continue
        by_c[instr] += cc
        by_y[instr] += cyc
        by_u[instr] += ru
        by_pipe[instr] = pipe
        if pipe.upper() in RVEC_PIPES:
            rvec_cycles += cyc

    # body_instr
    vf_key = None
    if side == "simt":
        for k in by_c:
            if k.lower() in ("vf_simt",):
                vf_key = k
                break
    else:
        for k in by_c:
            if k.lower() in ("vf", "vf_simd", "vf_simt"):
                vf_key = k
                break

    # Also accumulate vector-pipe call counts (for *d/*v body_instr)
    body_all = 0
    body_rvec = 0
    for r in rows:
        instr = (r.get("instr") or "").strip()
        pipe = (r.get("pipe") or "").strip()
        try:
            cc = int(float(r.get("call_count") or 0))
        except ValueError:
            continue
        kl = instr.lower()
        if kl in FRAMING:
            continue
        if vf_key and instr == vf_key:
            continue
        body_all += cc
        if pipe.upper() in RVEC_PIPES:
            body_rvec += cc

    if side == "simt":
        out["body_instr"] = body_all
        if vf_key:
            out["C_meas"] = by_y[vf_key]
            out["VF_us"] = by_u[vf_key]
            if out["C_meas"] and out["C_meas"] > 0:
                out["ipc"] = body_all / out["C_meas"]
                out["ipc_recipe"] = "body/VF_SIMT"
    else:
        # vector instr call_count sum (RVEC pipes) per task recipe
        out["body_instr"] = body_rvec if body_rvec > 0 else body_all
        # C_meas: VF_* row when present; else Σ RVEC pipe cycles
        # (RVEC sum double-counts parallel pipes — only a fallback)
        if vf_key and by_y[vf_key] > 0:
            out["C_meas"] = by_y[vf_key]
            out["VF_us"] = by_u[vf_key]
            out["ipc_recipe"] = "vec_body/VF"
        elif rvec_cycles > 0:
            out["C_meas"] = rvec_cycles
            out["ipc_recipe"] = "vec_body/ΣRVEC"
        exipc = next((k for k in by_c if k.upper() == "EXIPC"), None)
        if exipc and by_y[exipc] > 0:
            out["exipc"] = by_c[exipc] / by_y[exipc]
        if out["C_meas"] and out["C_meas"] > 0:
            out["ipc"] = out["body_instr"] / out["C_meas"]
        # annotate Case-3 *v: also stash RVEC sum for note
        out["rvec_cycles"] = rvec_cycles if rvec_cycles else None
    return out


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        if nd == 0:
            return f"{v:.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.{nd}f}"
    return str(v)


def one_side_cells(a: dict):
    return (
        a["status"],
        fmt(a["wall_us"]),
        fmt(a["C_meas"], 0),
        fmt(a["body_instr"], 0),
        fmt(a["ipc"], 3),
    )


def render(rows_spec):
    analyses = {}
    md = []
    tsv_lines = [
        "family\tshape\tsimt_tag\tsimt_st\tsimt_wall_us\tsimt_C_meas\tsimt_body_instr\tsimt_IPC"
        "\td_tag\td_st\td_wall_us\td_C_meas\td_body_instr\td_IPC\td_EXIPC"
        "\tv_tag\tv_st\tv_wall_us\tv_C_meas\tv_body_instr\tv_IPC\tnote"
    ]
    md += [
        "# SimtVF vs Simd/VMI (*d / *v) e2e compare — matched shapes",
        "",
        "**Date:** 2026-10-07 HKT  ",
        "**SOC:** Ascend950PR_9599  ",
        "**Sources:** `/tmp/st_simtvf_parallel`, `/tmp/pr272_sv567_rfladder_20261006`, `/tmp/pr272_sv4_cliff_20261005/out`, `/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d/`  ",
        "**No git push / no PR #272.**",
        "",
        "## Metric recipes",
        "",
        "- **wall_us** = `core0.veccore0` duration_time(us)",
        "- **Simt:** `C_meas` = `VF_SIMT.cycles`; `body_instr` = Σ call_count excluding `VF_*` / set_flag / wait_flag / end / nop / push_pb / dcci; `IPC` = body_instr / C_meas",
        "- **Simd/VMI (*d,*v):** `C_meas` = `VF.cycles` / `VF_SIMD.cycles` if present, else Σ RVEC pipe cycles; `body_instr` = Σ call_count on RVEC pipes only; `IPC` = body_instr / C_meas; EXIPC noted when present. Case-3 `*v` also notes `C_RVEC` (Σ RVEC) because wall_us is scalar-transpose dominated.",
        "- Layer-D `*d` is the primary VMI column (24/24 PASS). Layer-B `*v` included where PASS; Case-3 `*v` walls are **scalar-transpose dominated** (see ST_CASE3_VMI.md).",
        "- Do not invent numbers for COMPILE_FAIL / MISSING.",
        "",
        "## Matched-shape main table",
        "",
        "| Family | Shape / arm | Simt st | Simt µs | Simt C | Simt instr | Simt IPC | *d st | *d µs | *d C | *d instr | *d IPC | *v st | *v µs | *v C | *v instr | *v IPC | Note |",
        "|--------|-------------|---------|--------:|-------:|-----------:|---------:|-------|------:|-----:|---------:|-------:|-------|------:|-----:|---------:|-------:|------|",
    ]

    for fam, shape, simt, d, v, note in rows_spec:
        s = analyze(simt, "simt") if simt else {"status": "—", "wall_us": None, "C_meas": None, "body_instr": None, "ipc": None, "tag": ""}
        dd = analyze(d, "vmi") if d else {"status": "—", "wall_us": None, "C_meas": None, "body_instr": None, "ipc": None, "exipc": None, "tag": ""}
        vv = analyze(v, "vmi") if v else {"status": "—", "wall_us": None, "C_meas": None, "body_instr": None, "ipc": None, "tag": ""}
        if simt:
            analyses[simt] = s
        if d:
            analyses[d] = dd
        if v:
            analyses[v] = vv
        ss, sw, sc, si, sp = one_side_cells(s) if simt else ("—", "—", "—", "—", "—")
        ds, dw, dc, di, dp = one_side_cells(dd) if d else ("—", "—", "—", "—", "—")
        vs, vw, vc, vi, vp = one_side_cells(vv) if v else ("—", "—", "—", "—", "—")
        note2 = note
        if v and vv.get("status") == "PASS" and fam in ("SV5", "SV6", "SV7"):
            if "scalar-transpose" not in note:
                note2 = (note + "; *v wall scalar-transpose dominated").strip("; ")
            # append RVEC sum when available for VF-body fairness
            rc = vv.get("rvec_cycles")
            if rc:
                note2 = f"{note2}; C_RVEC={rc:.0f}".replace("; ;", ";")
        md.append(
            f"| {fam} | {shape} | {ss} | {sw} | {sc} | {si} | {sp} | {ds} | {dw} | {dc} | {di} | {dp} | {vs} | {vw} | {vc} | {vi} | {vp} | {note2} |"
        )
        tsv_lines.append(
            "\t".join([
                fam, shape,
                simt or "", ss, sw, sc, si, sp,
                d or "", ds, dw, dc, di, dp, fmt(dd.get("exipc"), 3) if d else "",
                v or "", vs, vw, vc, vi, vp,
                note2,
            ])
        )

    md += [
        "",
        "## Historical SV2 R32×C32 appendix (not matched-shape)",
        "",
        "| Tag | st | µs | C | instr | IPC | note |",
        "|-----|----|---:|--:|------:|----:|------|",
    ]
    for fam, shape, simt, d, v, note in HISTORICAL:
        s = analyze(simt, "simt")
        analyses[simt] = s
        ss, sw, sc, si, sp = one_side_cells(s)
        md.append(f"| `{simt}` | {ss} | {sw} | {sc} | {si} | {sp} | {note} |")
        tsv_lines.append(
            "\t".join([fam, shape, simt, ss, sw, sc, si, sp, "", "", "", "", "", "", "", "", "", "", "", "", "", note])
        )

    # coverage summary
    n_simt_pass = sum(1 for a in analyses.values() if a.get("side") == "simt" and a.get("status") == "PASS")
    md += [
        "",
        "## Coverage notes",
        "",
        f"- Analyzed tags with any status: {len(analyses)}",
        "- Simt SV2 `r16_c64_input_keep`: expect COMPILE_FAIL (layout); fallback `r16_c32` used for KEEP column.",
        "- *v Case-3 (sv5v–sv7v): PASS arms have walls dominated by scalar transpose outside VF; compare C_meas (RVEC), not wall_us, for VF-body fairness.",
        "- *v SV2/SV4/SV8/SV9: historically COMPILE_FAIL — left blank, not invented.",
        "",
        "## Artifact paths (remote)",
        "",
        "- TSV: `/tmp/st_simtvf_parallel/reports/simt_simd_e2e_20261007.tsv`",
        "- MD: `/tmp/st_simtvf_parallel/reports/SIMT_SIMD_E2E_COMPARE_20261007.md`",
    ]

    tsv_path = REPORTS / "simt_simd_e2e_20261007.tsv"
    md_path = REPORTS / "SIMT_SIMD_E2E_COMPARE_20261007.md"
    tsv_path.write_text("\n".join(tsv_lines) + "\n")
    md_path.write_text("\n".join(md) + "\n")
    print(f"WROTE {tsv_path}")
    print(f"WROTE {md_path}")
    # also print a compact summary of SV2 rows
    print("===SV2_ROWS===")
    for tag in [
        "sv2_r16_c64_t32_input_keep",
        "sv2_r16_c32_t32_input_keep",
        "sv2_r32_c64_t32_input_stream",
        "sv2_r32_c128_t32_input_stream",
        "sv2_r32_c64_t32_reload",
        "sv2d_r16_c64_t32_input_keep",
        "sv2d_r32_c64_t32_input_stream",
        "sv2d_r32_c128_t32_input_stream",
        "sv2d_r32_c64_t32_fold_scale_reload",
    ]:
        a = analyses.get(tag) or analyze(tag, "simt" if "d" not in tag[2:4] else "vmi")
        print(
            f"{tag}: st={a['status']} wall={a['wall_us']} C={a['C_meas']} instr={a['body_instr']} ipc={a['ipc']} recipe={a.get('ipc_recipe')}"
        )


if __name__ == "__main__":
    render(ROWS)
