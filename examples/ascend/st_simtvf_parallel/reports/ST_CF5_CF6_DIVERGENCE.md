# When does Simt divergence really win? (CF5 / CF6)

**Date:** 2026-10-07 ~10:44 HKT  
**Audience:** Lok / CF-bench  
**Inputs:** Simt CF5/CF6, Layer-B CF5v/CF6v, Layer-D CF5d/CF6d + pnear∈{5,25}

## Bottom line (honest)

1. **Vs Layer-B `*v`:** the Simt “win” (1.18 / 1.37 vs 6.23 / 12.35 µs) is **mostly ABI tax avoidance**, not divergence. TileLang `SimdVF` Parallel whitelist rejects Div/Sub/Max; CF5v/CF6v hoist Div/Newton to Kernel-serial **outside** VF → huge remat/publish tax.
2. **Vs Layer-D `*d` (vector rooftop):** Simt does **NOT** win on µs at E=256. CF5d/CF6d keep Div/Newton in vregs (`vdiv`/`vmul`/`vsub` + `vcmp`/`vsel`) → **0.89 / 0.92 µs**, slightly **faster** than Simt.
3. **Skew knobs (`pnear`, `pfat`) do not move wall** on any of Simt / `*v` / `*d` at E=256 for these micros — lockstep Select (and SimtVF at this size) still pays both arms.

So: **call the Simt CF5/CF6 advantage “ABI escape vs Layer-B”, not “divergence beats vector.”** Divergence would need a workload the suite does not yet measure.

## Numbers (Ascend950PR_9599, core0.veccore0 µs)

| Twin | path | pnear5 | pnear25 | source |
|------|------|-------:|--------:|--------|
| CF5 Simt | `T.SimtVF` Parallel if abs&lt;eps else a/b | **1.18** | **1.18** | ST_CF_IMPLEMENT 2026-09-28 |
| CF5v | Div Kernel-serial **outside** SimdVF; VF Select only | **6.23** | **6.23** | same |
| CF5d | `vdiv`+`vcmp`/`vsel` in RF | **0.89** | **0.89** | opsim 2026-10-07 |
| CF6 Simt | Parallel near-0 + `serial(N_FAST=3)` Newton | **1.37** | **1.37** | ST_CF_IMPLEMENT |
| CF6v | Newton Kernel-serial outside VF | **12.35** | **12.35** | same |
| CF6d | seed `vdiv` + 3× Newton in vregs | **0.92** | **0.93** | opsim 2026-10-07 |

Related nested-if (not div, but same skew lesson): CF4 Simt 1.23/1.25, CF4v 0.93/0.93, CF4d **0.89/0.89** for pfat5/25 — flat.

## What the suite currently measures

| Layer | What it really stresses |
|-------|-------------------------|
| Simt CF5/CF6 | Predicated Parallel + short Newton loop; **no proven per-lane early-exit savings** at E=256 (pnear flat) |
| CF5v/CF6v | **ABI tax** of computing Div/Newton outside SimdVF + remat into VF Select |
| CF5d/CF6d | Pure vector ISA rooftop: always evaluate quot/Newton for all lanes, then `vsel` near-0 → 0 |

Host `pnear` only changes **input skew** (fraction of near-0 lanes). Kernel binaries and lockstep schedules are identical across pnear tags.

## When divergence *would* matter (not yet shown)

| Knob | Why it could help Simt vs lockstep `*d` |
|------|----------------------------------------|
| **Fatter near-0 path** | Today near-0 is “store 0” — cheaper than Div. If near-0 did heavy work *or* far path was skippable with true lane exit, skew would show. |
| **More Newton iters / adaptive iters** | Suite fixes `N_FAST=3` for all far lanes. Per-lane early Newton exit (or extra iters only on hard lanes) favors divergence. |
| **Larger E / multi-block** | Divergence amortization / warp occupancy effects invisible at E=256 single Kernel. |
| **True per-lane early exit** | Need ISA/runtime that skips dead lanes’ Div/Newton, not mask-Select that still issues ops. Ascend vector `vsel` is lockstep; SimtVF at E=256 behaved like flat wall too. |
| **Asymmetric arm cost** | Nested CF4 fat path is one mul — too thin. A fat arm with many FMAs/divs + low `pfat` is the classic Simt story. |

## Framing for CF-bench writeups

- Prefer: “Simt CF5/CF6 beat CF5v/CF6v because **Layer-B cannot put Div/Newton in Parallel**.”
- Avoid: “Simt divergence beats vector masking on CF5/CF6” — **false vs Layer-D** at current size.
- Layer-D is the fair vector baseline; Layer-B is the TileLang ABI baseline.
- To claim a real divergence win later: add a micro with **fat near-0-or-far arm**, optional **adaptive Newton**, and sweep `pnear`/`pfat` until Simt µs drops while `*d` stays flat (lockstep).

## Files

- Kernels: `kernels_ptodsl/cf5d_div_ulp_branch.py`, `cf6d_newton_branch.py`
- Prior note: `reports/ST_CF5D_CF6D.md`
- Full CF1–CF4d: `reports/ST_CF2D_CF4D.md`
- Oneshot: `oneshot_ptodsl_cf5d_cf6d.sh`, `oneshot_ptodsl_cf1d_cf6d.sh`
