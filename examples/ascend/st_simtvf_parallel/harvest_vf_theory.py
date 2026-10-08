#!/usr/bin/env python3
"""VF cycle THEORY model + measured comparison for the SV5–SV7 RF-capacity ladder.

Usable two ways:
  * module  — ``import harvest_vf_theory as TH`` then ``TH.theory(fam, R, C, G, T, arm)``
              and ``TH.compare(th, c_meas, body_instr, us_meas)``
  * CLI     — ``harvest_vf_theory.py [OUT_DIR]`` writes ``$OUT/VF_THEORY.md`` +
              ``$OUT/vf_theory.tsv`` for every ladder tag found in that OUT dir.

See ``reports/ST_VF_THEORY.md`` for the formula and the assumption tables; every
default here is overridable by env var so a5 vs a6 uarch studies can re-run the
same W with different IPC / overhead / frequency.

Model
-----
    W        = useful work ops in the VF body (whole-kernel, all lanes)
    W_thr    = W / T                      (SIMT issue is per-thread; the camodel
                                           instr CSV counts per-thread too)
    C_pred   = O_arm + ceil(W_thr / I_assum)
    us_pred  = C_pred / FREQ_HZ
    ratio    = C_meas / C_pred            (OUTLIER if >2.0 or <0.5)
    IPC_meas = body_instr / C_meas        (compare against I_assum)

W terms (per arm), with E = R*C elements, N = R*CG groups, NW = 4 chunks:
    W_arith = 3E    cast f16→f32 (1) + abs as max(v,-v) (1) + running max (1)
    W_mem   = 1E    ub_stream / reload          (one UB load per element)
            = 2E    keep_in_rf / keep_in_warp   (RF store in preload + RF load in reduce)
            = 1E + 3*N*NW  multiwarp_ub         (UB load + partial st/ld + merge max)
    W_emit  = 3N    clamp to 1e-6, reciprocal, store sf_inv
    legacy keep_reg / ub_reload additionally carry the sink'd subsequent
    compute: + (2E + 1E) = 3E (reload + |x|·s FFMA) and + 1N sink store.
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ---- tunable assumptions (a5 vs a6: re-run with different values) ----
IPC_SIMT = float(os.environ.get("ST_VF_IPC_SIMT", "0.20"))
# SIMT bodies whose inner serial loop is NOT unrolled run a real branch loop
# (ISETP_I + BRANCH + LEA + IADD per element) and issue much worse.
IPC_SIMT_BRANCHY = float(os.environ.get("ST_VF_IPC_SIMT_BRANCHY", "0.10"))
UNROLL_MAX = int(os.environ.get("ST_VF_UNROLL_MAX", "16"))  # serial trips <= this are unrolled
W_LOOP_OPS = float(os.environ.get("ST_VF_W_LOOP_OPS", "4"))  # idx/cmp/branch ops per elem pass
IPC_VMI = float(os.environ.get("ST_VF_IPC_VMI", "0.85"))
FREQ_HZ = float(os.environ.get("ST_VF_FREQ_HZ", "1.8e9"))  # camodel Ascend950PR_9599
O_BASE = float(os.environ.get("ST_VF_O_BASE", "300"))      # VF enter/exit + UB prologue
O_LOOP = float(os.environ.get("ST_VF_O_LOOP", "250"))      # each extra Parallel loop
O_UBRT = float(os.environ.get("ST_VF_O_UBRT", "400"))      # UB round-trip + membar/flag
NW = 4

ARMS = "keep_in_rf|ub_stream|keep_in_warp|reload|multiwarp_ub|keep_reg|ub_reload"
TAG_RE = re.compile(rf"^(sv[567]v?)_r(\d+)_c(\d+)_g(\d+)_t(\d+)_({ARMS})$")
OUTSIDE = {"set_flag", "wait_flag", "end_label", "end", "nop", "push_pb", "dcci"}

# arm → (n_extra_parallel_loops, n_ub_roundtrips)
ARM_OVERHEAD = {
    "ub_stream": (0, 0),
    "reload": (0, 0),
    "keep_in_rf": (1, 0),
    "keep_in_warp": (1, 0),
    "multiwarp_ub": (1, 1),
    "keep_reg": (1, 0),
    "ub_reload": (1, 1),
}


def theory(fam: str, R: int, C: int, G: int, T: int, arm: str) -> dict:
    """Predicted VF cycles for one case/arm. Pure function of shape + arm."""
    E = R * C
    N = R * (C // G)
    w_arith = 3 * E
    if arm in ("keep_in_rf", "keep_in_warp"):
        w_mem = 2 * E
    elif arm == "multiwarp_ub":
        w_mem = 1 * E + 3 * N * NW
    else:
        w_mem = 1 * E
    w_emit = 3 * N
    W = w_arith + w_mem + w_emit
    if arm in ("keep_reg", "ub_reload"):   # legacy arms keep the subsequent compute
        W += 3 * E + 1 * N
    is_vmi = fam.endswith("v")
    # inner serial trip count + how many times the group is walked element-wise
    if arm == "multiwarp_ub":
        inner_trip, passes = G // NW, 1
    elif arm in ("keep_in_rf", "keep_in_warp"):
        inner_trip, passes = G, 2          # preload pass + reduce pass
    elif arm in ("keep_reg", "ub_reload"):
        inner_trip, passes = G, 2          # reduce pass + subsequent compute pass
    else:
        inner_trip, passes = G, 1
    # On SimtVF a serial loop longer than the unroll budget stays a branch loop:
    # it adds index/compare/branch ops that W does not otherwise count, and the
    # body issues at the much lower "branchy" IPC. On SimdVF/VMI the serial loop
    # wraps a *vector* Parallel, so its loop control is amortised over the lanes
    # and no penalty applies.
    branchy = (not is_vmi) and inner_trip > UNROLL_MAX
    w_loop = int(W_LOOP_OPS * E * passes) if branchy else 0
    W += w_loop
    ipc = IPC_VMI if is_vmi else (IPC_SIMT_BRANCHY if branchy else IPC_SIMT)
    # VMI/SimdVF issues per lane-group, not per SIMT thread: the VF body runs the
    # whole domain vectorised, so divide by lanes (tag `t` field) the same way.
    w_thr = W / float(T)
    nl, nrt = ARM_OVERHEAD.get(arm, (0, 0))
    O = O_BASE + O_LOOP * nl + O_UBRT * nrt
    c_pred = O + math.ceil(w_thr / ipc)
    return dict(W=W, W_thr=w_thr, I_assum=ipc, O=O, C_pred=c_pred,
                us_pred=c_pred / FREQ_HZ * 1e6, freq_hz=FREQ_HZ,
                w_loop=w_loop, branchy=branchy, inner_trip=inner_trip,
                backend=("VMI" if is_vmi else "Simt"))


def compare(th: dict, c_meas, body_instr, us_meas=None) -> dict:
    """Measured vs predicted. Never fabricates a measurement."""
    out = dict(C_meas=c_meas, body_instr=body_instr, us_meas=us_meas,
               ratio=None, IPC_meas=None, freq_implied=None, verdict="NO_MEAS")
    if not c_meas:
        return out
    out["ratio"] = c_meas / th["C_pred"]
    if body_instr:
        out["IPC_meas"] = body_instr / c_meas
    if us_meas:
        out["freq_implied"] = c_meas / (us_meas * 1e-6)
    r = out["ratio"]
    out["verdict"] = "OK" if 0.5 <= r <= 2.0 else ("OUTLIER_SLOW" if r > 2.0 else "OUTLIER_FAST")
    return out


# ---------------- measurement side (camodel instr CSV) ----------------
def vf_measured(out: Path, tag: str):
    """→ (vf_cycles, vf_us, body_instr, top_opcodes, source).

    SimtVF traces carry a single ``VF_SIMT`` row = the whole VF region, so that
    row is the measurement. SimdVF/VMI traces have no ``VF_*`` row (and this
    PTODSL emits no ``EXIPC``/``IFU`` counters either), so the VF body is taken
    as the sum over the **vector pipes** (``RVEC*``): that deliberately excludes
    the scalar transpose prologue and the scalar reciprocal epilogue, which the
    theory ``W`` does not model.
    """
    root = out / f"opsim_{tag}"
    hits = list(root.rglob("core0.veccore0_instr_exe.csv")) if root.is_dir() else []
    if not hits:
        return None, None, None, "", "none"
    by = Counter()
    vf_cyc = vf_us = None
    body = 0
    rvec_cyc = rvec_us = 0.0
    rvec_cnt = 0
    for r in csv.DictReader(hits[0].open(newline="", encoding="utf-8", errors="replace")):
        instr = (r.get("instr") or "").strip()
        pipe = (r.get("pipe") or "").strip().upper()
        try:
            cc = int(float(r.get("call_count") or 0))
            cyc = int(float(r.get("cycles") or 0))
            us = float(r.get("running_time(us)") or 0)
        except ValueError:
            continue
        by[instr] += cc
        if pipe.startswith("RVEC"):
            rvec_cyc += cyc
            rvec_us += us
            rvec_cnt += cc
        if instr in ("VF_SIMT", "VF_SIMD"):
            vf_cyc = (vf_cyc or 0) + cyc
            vf_us = (vf_us or 0.0) + us
        elif instr.lower() not in OUTSIDE:
            body += cc
    top = ", ".join(f"{i}:{c}" for i, c in by.most_common(8))
    if vf_cyc:
        return vf_cyc, vf_us, body, top, "VF_SIMT"
    if rvec_cnt:
        return int(rvec_cyc), rvec_us, rvec_cnt, top, "RVEC pipes"
    return None, None, None, top, "none"


def rows_for(out: Path, tags=None):
    if tags is None:
        tags = set()
        for p in (out / "so").glob("*.so"):
            tags.add(p.stem)
        for p in out.glob("opsim_*.log"):
            tags.add(p.name[len("opsim_"):-len(".log")])
        tags = sorted(t for t in tags if TAG_RE.match(t))
    rows = []
    for tag in tags:
        m = TAG_RE.match(tag)
        if not m:
            continue
        fam, R_, C_, G_, T_, arm = m.groups()
        th = theory(fam, int(R_), int(C_), int(G_), int(T_), arm)
        cyc, us, body, top, src = vf_measured(out, tag)
        cmp_ = compare(th, cyc, body, us)
        rows.append(dict(tag=tag, fam=fam, R=int(R_), C=int(C_), G=int(G_),
                         T=int(T_), arm=arm, top=top, meas_src=src, **th, **cmp_))
    return rows


def md_table(rows) -> list[str]:
    L = ["| tag | arm | W | W_loop | W/T | I_assum | O | C_pred | µs_pred | C_meas | meas_src | ratio | IPC_meas | verdict |",
         "|-----|-----|--:|------:|----:|--------:|--:|-------:|--------:|-------:|----------|------:|---------:|---------|"]
    for r in sorted(rows, key=lambda r: (r["fam"], r["G"], r["arm"])):
        f = lambda v, p=2: ("-" if v is None else f"{v:.{p}f}")
        L.append(
            f"| {r['tag']} | {r['arm']} | {r['W']} | {r['w_loop']} | {r['W_thr']:.0f} | {r['I_assum']:.2f} | "
            f"{r['O']:.0f} | {r['C_pred']} | {r['us_pred']:.2f} | "
            f"{r['C_meas'] if r['C_meas'] else '-'} | {r.get('meas_src', '-')} | "
            f"{f(r['ratio'])} | {f(r['IPC_meas'], 3)} | {r['verdict']} |"
        )
    return L


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/st_simtvf_parallel")
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_for(out)
    with (out / "vf_theory.tsv").open("w") as fh:
        fh.write("TAG\tarm\tW\tW_loop\tW_thr\tI_assum\tO\tC_pred\tus_pred\tC_meas\tmeas_src\tus_meas\tratio\tIPC_meas\tfreq_implied\tverdict\n")
        for r in rows:
            g = lambda k, p=4: ("" if r[k] is None else f"{r[k]:.{p}g}")
            fh.write(f"{r['tag']}\t{r['arm']}\t{r['W']}\t{r['w_loop']}\t{r['W_thr']:.1f}\t{r['I_assum']}\t{r['O']:.0f}\t"
                     f"{r['C_pred']}\t{r['us_pred']:.3f}\t{r['C_meas'] or ''}\t{r.get('meas_src','')}\t{g('us_meas')}\t"
                     f"{g('ratio')}\t{g('IPC_meas')}\t{g('freq_implied')}\t{r['verdict']}\n")
    L = ["# VF cycle theory vs camodel measurement — SV5–SV7 RF-capacity ladder", "",
         f"OUT: `{out}`", "",
         f"Assumptions: I_assum(Simt)={IPC_SIMT} (branchy {IPC_SIMT_BRANCHY}, "
         f"unroll<= {UNROLL_MAX}), I_assum(VMI)={IPC_VMI}, "
         f"O_base={O_BASE:.0f}, O_loop={O_LOOP:.0f}, O_ub_roundtrip={O_UBRT:.0f}, "
         f"freq={FREQ_HZ/1e9:.2f} GHz.", "",
         "`C_pred = O_arm + ceil((W/T) / I_assum)`; see `reports/ST_VF_THEORY.md`.", ""]
    L += md_table(rows)
    fi = [r["freq_implied"] for r in rows if r["freq_implied"]]
    if fi:
        L += ["", f"Implied camodel frequency from VF rows (cycles/running_time): "
                  f"{min(fi)/1e9:.3f}–{max(fi)/1e9:.3f} GHz "
                  f"(mean {sum(fi)/len(fi)/1e9:.3f} GHz) — used as the µs conversion basis."]
    txt = "\n".join(L) + "\n"
    (out / "VF_THEORY.md").write_text(txt)
    print(txt)
    print("WROTE", out / "VF_THEORY.md", out / "vf_theory.tsv")


if __name__ == "__main__":
    main()
