# CF6b (Simt) vs CF6bd (PTO-DSL) — fat-hard / thin-easy divergence micro

**Date:** 2026-10-07 ~14:50 HKT  
**SOC:** Ascend950PR_9599 (opsim login, pto-b10)  
**Work dir (remote):** `/tmp/pr272_cf6b_cf6bd_20261007/`  
**Out (remote):** `/tmp/st_simtvf_parallel/cf6b_cf6bd/`  
**Box sources:** `/workspace/st_simtvf_parallel/kernels/cf6b_fat_hard_newton.py`, `kernels_ptodsl/cf6bd_fat_hard_newton.py`  
**Oneshot:** `oneshot_cf6b_cf6bd.sh`  
**Harvest:** log-truth (`PASS` + `core0.veccore0`), not SUMMARY.tsv rc  
**Deps:** `.camodel_deps_vmi018` (NOT bare PTOAS-vmi/ptodsl)  
**No PR #272 push.**

## Intent

Asymmetric arm costs so Simt **can** drop µs as `phard` falls, while CF6bd stays flat at the fat lockstep cost.

| Arm | Cost |
|-----|------|
| Near-0 (`\|x\| < EPS`) | `y=0` |
| Hard (`EPS ≤ \|x\| < HARD_HI`) | seed + `N_HARD=12` Newton + 8 Horner FMAs |
| Easy (`\|x\| ≥ HARD_HI`) | seed + `N_FAST=2` Newton |

Host skew `phard%` → hard band; rest easy. Primary: `E=256 T=32 phard∈{5,25}`.

## Tag table (filled)

| ST | shape | arm | schedule | status | µs | maxabs |
|----|-------|-----|----------|--------|---:|--------|
| CF6b | E=256 T=32 | phard5 | SimtVF Parallel if/elif/else (fat hard / thin easy) | PASS | 2.02 | 1.192e-07 |
| CF6b | E=256 T=32 | phard25 | SimtVF Parallel if/elif/else (fat hard / thin easy) | PASS | 2.02 | 7.629e-06 |
| CF6bd | E=256 T=32 VL64 | phard5 | lockstep always fat+thin + nested vsel | PASS | **1.08** | 9.537e-07 |
| CF6bd | E=256 T=32 VL64 | phard25 | lockstep always fat+thin + nested vsel | PASS | **1.08** | 3.815e-06 |

**Bold µs** = faster twin within each phard pair. Walls from `core0.veccore0 duration_time(us)`.

## Does Simt µs drop with lower phard while CF6bd stays flat?

| Question | Answer |
|----------|--------|
| CF6bd flat across phard5→25? | **YES** — 1.08 µs both (lockstep rooftop as designed) |
| CF6b Simt drops as phard falls? | **NO** — **2.02 µs both** (identical wall) |
| At phard5, CF6b &lt; CF6bd? | **NO** — Simt **2.02** loses to *d **1.08** (~1.87× slower) |

**Verdict:** Do **not** claim Simt divergence win on CF6b vs CF6bd at this size. Success criteria from `ST_CF6_DIVERGENCE_CASE.md` (2)+(3) **fail**. Ascend SimtVF at E=256/T=32 appears to still pay lockstep/predicate cost across asymmetric arms (same µs + identical `instr_exe` row count across phard). Escalate later: `break`/while early-exit (if TileLang supports), larger E / multi-block, or document “SimtVF does not early-exit asymmetric arms at this micro.”

### instr_exe identity (proxy)

| Tag | instr_rows (`core0.veccore0_instr_exe.csv`) | note |
|-----|--------------------------------------------:|------|
| CF6b phard5 | 455 | |
| CF6b phard25 | 455 | identical to phard5 |
| CF6bd phard5 | 340 | |
| CF6bd phard25 | 340 | identical to phard5 |

emit-mlir CF6bd: **338** lines.

## Artifacts

- This dir: `SUMMARY_harvest.tsv`, per-tag `.log`, `cf6bd_e256_t32_phard5.mlir`
- Remote logs: `/tmp/st_simtvf_parallel/cf6b_cf6bd/`
- Oneshot: `/workspace/st_simtvf_parallel/oneshot_cf6b_cf6bd.sh`
- Design: `reports/ST_CF6_DIVERGENCE_CASE.md`
- Also: `reports/ST_CF6B_RESULT.md` (symlink-style copy of this table)

## Notes

- Post-PASS Simt host abort (`corrupted size` / SEGV after `PASS maxabs=…`) is a known sim_dsl teardown glitch; wall + numeric PASS are valid (same pattern as other CF micros).
- Horner polish uses `T.alloc_var("float32")` (mutable) inside Simt Parallel.
- CF6bd must **not** use `from __future__ import annotations` — stringified `pto.ptr(...)` breaks `@pto.jit` entry parse on camodel deps.
