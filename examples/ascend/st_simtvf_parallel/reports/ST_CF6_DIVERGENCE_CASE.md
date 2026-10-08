# CF6 divergence case — schedule verdict + CF6b micro

**Date:** 2026-10-07 ~11:05 HKT  
**Audience:** Lok / CF-bench  
**Box:** `/workspace/st_simtvf_parallel/`  
**Prior:** `reports/ST_CF5_CF6_DIVERGENCE.md`, `ST_CF5D_CF6D.md`, `ST_CF_IMPLEMENT.md`

## (a) Is the CURRENT CF6 Simt schedule good for a divergence claim?

**No.** Do not claim “Simt divergence beats lockstep vector” on current CF6 vs CF6d.

### What current Simt CF6 actually does

Source: `kernels/cf6_newton_branch.py`

```text
Parallel(E):
  if abs(x) < eps:     y = 0                    # thin near-0 arm
  else:
    y = 1/x
    serial(N_FAST=3):  y = y*(2 - x*y)          # fixed trip count, no break
```

- Newton lives **inside** `T.Parallel` / `T.SimtVF` (good vs CF6v ABI tax).
- Trip count is **compile-time fixed** `serial(N_FAST)` — **no** per-lane early exit / residual break.
- Near-0 arm is a single store-0 (cheaper than Div+Newton, but **too thin** to move wall).

### Evidence it still pays lockstep-ish cost

| Twin | pnear5 | pnear25 | note |
|------|-------:|--------:|------|
| CF6 Simt | **1.37 µs**, VF **961** cyc, **192** instr | **same** | `ST_CF_IMPLEMENT` / `ST_CF1_CF6_SLICES` |
| CF6v | 12.35 µs | same | ABI tax (Newton outside SimdVF) |
| CF6d | **0.92 / 0.93 µs** | flat | vector rooftop; always seed+3 Newton then `vsel` |

- **pnear flat** on Simt µs **and** VF cycles / instr# → changing near-0 fraction does not retire Newton work.
- **Simt loses to CF6d** (1.37 > 0.92). Layer-D is the fair VMI baseline.
- No CF6 `instr_log.dump` harvested on this box; cycle identity across pnear is the available proxy. (SV harvests under `reports/sv2g_harvest/` are unrelated.)

### Schedule critique (Newton-in-Parallel + serial(N_FAST))

| Mechanism | Likely on Ascend SimtVF at E=256 / T=32? |
|-----------|------------------------------------------|
| Per-lane **early exit** of Newton | **No** — fixed `serial(N_FAST)`, no `break` / residual guard |
| Warp skips thin near-0 when siblings take far | Possible in principle, but near-0 is store-0; wall stays Newton-dominated → **pnear flat** |
| Paying both arms (predicated / lockstep) | Consistent with flat pnear + identical instr# |
| Beating CF6d via divergence | **Not with this micro** — *d* is already cheaper doing lockstep Newton in vregs |

**Verdict:** Current schedule is a fine **ABI-escape** demo vs Layer-B `*v`, and a fine Newton-in-SimtVF smoke test. It is **not** a divergence-win demo vs Layer-D.

Design intent in `ST_CF1_CF6_SLICES.md` (“near-0 / bad seed takes **more** iters; healthy lanes exit early”) was **inverted** in the landed kernel (near-0 → 0; far → fixed 3 iters).

---

## (b) ONE concrete CF6-divergent micro: **CF6b** (`cf6b_fat_hard`)

**Goal:** asymmetric arm costs so Simt **can** drop µs as `phard` falls, while CF6bd stays flat at the fat lockstep cost.

### Semantics (gold)

Constants: `EPS=1e-4`, `HARD_HI=1e-1`, `N_FAST=2`, `N_HARD=12`, `N_POLY=8` (poly polish on hard arm only).

For each lane `x`:

1. `ax = abs(x)`
2. **Near-0** (`ax < EPS`): `y = 0`
3. **Hard** (`EPS ≤ ax < HARD_HI`): seed `y=1/x`, then `N_HARD` Newton iters `y←y*(2-x*y)`, then `N_POLY` Horner-style FMAs  
   `y ← ((…(c_{n}·x + c_{n-1})·x + …)·x + c_0) + y` (fixed coeffs; gold must match)
4. **Easy** (`ax ≥ HARD_HI`): seed `y=1/x`, then `N_FAST` Newton only (no poly)

Host skew knob **`phard`**: force first `round(E*phard/100)` lanes into the hard band `[EPS, HARD_HI)`; remaining lanes into easy (`|x| ≥ HARD_HI`). Keep a tiny `pnear` (e.g. 0 or 5) optional — primary sweep is **phard∈{5,25,50}**.

Tags: `cf6b_e{E}_t{T}_phard{P}` — primary `cf6b_e256_t32_phard{5,25}`.  
Layer-D twin tags: `cf6bd_e256_t32_phard{P}`.

### Why Simt SHOULD beat *d here

| Side | Cost model |
|------|------------|
| **CF6bd (*d)** | Lockstep: per VL chunk always run **hard pipeline** (or both arms + `vsel`). Wall ≈ fat cost, **flat vs phard**. |
| **CF6b Simt** | `if/elif/else` inside `SimtVF` Parallel. At **low phard**, many warps are all-easy → pay only `N_FAST` seed+Newton. Mixed warps serialize arms but still beat always-fat *d if fat≫thin. |

Cost ratio target: hard arm ≈ **6–10×** easy arm (12 Newton + 8 FMAs vs 2 Newton). CF4’s “fat” was one mul — too thin; this is the fix.

### Simt schedule sketch (required shape)

```python
with T.SimtVF(threads=threads):
    for i in T.Parallel(E):
        ax = T.max(x[i], -x[i])
        if ax < eps:
            y[i] = T.float32(0.0)
        elif ax < hard_hi:
            y[i] = T.float32(1.0) / x[i]
            for _t in T.serial(N_HARD):          # fat
                y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
            # N_POLY Horner polish (fat tail) — see draft kernel
            ...
        else:
            y[i] = T.float32(1.0) / x[i]
            for _t in T.serial(N_FAST):          # thin
                y[i] = y[i] * (T.float32(2.0) - x[i] * y[i])
```

**Keep work inside SimtVF Parallel** — do **not** hoist Newton/poly to Kernel-serial (that recreates CF6v).

### *d twin sketch (lockstep rooftop)

Per VL chunk: build `near` / `hard` masks; `safe_x = vsel(near, 1, x)`; always run seed + **N_HARD** Newton in vregs + **N_POLY** FMAs; then  
`y = vsel(near, 0, vsel(hard, y_hard, y_easy))`  
where `y_easy` is either recomputed with N_FAST or taken as “first N_FAST of the hard pipeline” (prefer **recompute thin** or **slice** so gold matches; simplest fair rooftop = always full hard pipeline then `vsel` near→0 and hard-vs-easy select from precomputed easy path).  
Honest lockstep upper bound: **always pay hard**.

### Stretch (not required for CF6b v1): adaptive break

If TileLang grows `break` / `while` inside SimtVF Parallel:

```text
serial(N_MAX=16): Newton step; if abs(1 - x*y) < tol: break
```

Then *d must issue N_MAX always; Simt can retire easy lanes early. **Unknown** whether Ascend SimtVF + TileLang support this today — **no suite example**. Prefer fat-hard / thin-easy first (no break dependency).

### What schedule change is required on Simt side?

| Change | Need? |
|--------|------:|
| Fatter hard arm (`N_HARD≫N_FAST` + poly) | **Yes** |
| Asymmetric `if/elif/else` (not near-0→0 only) | **Yes** |
| Keep Newton/poly **inside** SimtVF Parallel | **Yes** |
| `break` / residual early exit | Optional stretch |
| Separate Parallel phases / remat outside VF | **No** (hurts; CF6v path) |
| Larger E / multi-block | Optional later |

---

## (c) Files / paths

| Item | Path |
|------|------|
| Current Simt CF6 | `kernels/cf6_newton_branch.py` |
| Current CF6v (ABI tax) | `kernels/cf6v_newton_branch.py` |
| Current CF6d (rooftop) | `kernels_ptodsl/cf6d_newton_branch.py` |
| Prior verdict | `reports/ST_CF5_CF6_DIVERGENCE.md` |
| **This design** | `reports/ST_CF6_DIVERGENCE_CASE.md` |
| **Draft Simt CF6b** | `kernels/cf6b_fat_hard_newton.py` |
| **Draft *d CF6bd** | `kernels_ptodsl/cf6bd_fat_hard_newton.py` |

**Opsim done 2026-10-07 ~14:50 HKT:** see `reports/cf6b_cf6bd_20261007/RESULT.md` / `ST_CF6B_RESULT.md`. CF6b Simt **flat 2.02 µs** phard5=phard25; CF6bd **flat 1.08 µs**; Simt loses to *d. Divergence-win criteria (2)+(3) **fail**. No PR #272 push.

### Success criteria for a later opsim pass

Claim Simt divergence win only if:

1. CF6bd µs **flat** across `phard∈{5,25}` (lockstep).
2. CF6b Simt µs **drops** as `phard` falls (e.g. phard5 ≪ phard50), and  
3. At **phard5**, CF6b Simt **&lt;** CF6bd µs by a clear margin (not noise).

If (2) fails (Simt also flat), SimtVF is still lockstep/predicate-paying both arms → escalate to break/while or larger E, or document “Ascend SimtVF does not early-exit asymmetric arms at this size.”

---

## One-liner for Lok

**Current CF6 Simt schedule is not good for a divergence claim** (loses to CF6d; pnear flat; fixed `serial(3)`). **CF6b** = fat hard-band Newton+poly vs thin easy Newton; sweep `phard`; *d always pays fat. Drafts under `kernels/cf6b_fat_hard_newton.py` + `kernels_ptodsl/cf6bd_fat_hard_newton.py`.
