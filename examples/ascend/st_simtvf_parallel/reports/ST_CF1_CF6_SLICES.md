# CF control-flow slices — Simt vs Simd sensitivity

**Date:** 2026-09-25 ~11:45 HKT  
**Suite:** `examples/ascend/st_simtvf_parallel/`  
**Status:** Simt CF1–CF6 + VMI CF1V–CF6V landed; oneshots `oneshot_cf1_cf6.sh` / `oneshot_cf1v_cf6v.sh`  
**Updated:** 2026-09-28 ~10:45 HKT  
**Parent:** [`ST_CF_TOPK_GATE_DESIGN.md`](ST_CF_TOPK_GATE_DESIGN.md) · [`ST_LIST_DESIGN.md`](ST_LIST_DESIGN.md)

---

## Summary table (Simt oneshot 2026-09-25)

Harvested from `/tmp/st_simtvf_parallel/opsim_cf*` via `harvest_cf_summary.py`.

| case | tag | what to test | status | e2e µs | VF cyc | instr# | IPC | EX instr# | EXIPC |
|------|-----|--------------|--------|-------:|-------:|-------:|----:|----------:|------:|
| CF1 | `cf1_e256_k1_t32_keep` | One Parallel if vs scalar thresh; scores KEEP (K=1) | PASS | 1.15 | 576 | 137 | 0.238 | 40 | — |
| CF1 | `cf1_e256_k8_t32_keep` | One Parallel if vs scalar thresh; scores KEEP (K=8) | PASS | 0.97 | 309 | 256 | 0.828 | 159 | — |
| CF2 | `cf2_e256_k8_t32_remat` | Same if; remat + kill-in-shared | PASS | 1.95 | 2041 | 362 | 0.177 | 265 | — |
| CF3 | `cf3_e256_k8_t32_remat_idx` | remat_idx only (kill/select by rematted i) | PASS | 1.33 | 851 | 272 | 0.320 | 175 | — |
| CF4 | `cf4_e256_t32_pfat5` | Nested if (2-level); p_fat=5% skew | PASS | 1.23 | 758 | 212 | 0.280 | 110 | — |
| CF4 | `cf4_e256_t32_pfat25` | Nested if (2-level); p_fat=25% skew | PASS | 1.25 | 781 | 212 | 0.271 | 110 | — |
| CF5 | `cf5_e256_t32_pnear5` | Div + near-0 range branch (ULP intent); pnear=5% | PASS | 1.18 | 657 | 149 | 0.227 | 56 | — |
| CF5 | `cf5_e256_t32_pnear25` | Div + near-0 range branch (ULP intent); pnear=25% | PASS | 1.18 | 656 | 149 | 0.227 | 56 | — |
| CF6 | `cf6_e256_t32_pnear5` | Newton recip-style + near-0 / extra-iter; pnear=5% | PASS | 1.37 | 961 | 192 | 0.200 | 102 | — |
| CF6 | `cf6_e256_t32_pnear25` | Newton recip-style + near-0 / extra-iter; pnear=25% | PASS | 1.37 | 961 | 192 | 0.200 | 102 | — |

**Metric recipes** (same as `harvest_simt_vmi_compare.py` / `harvest_cf_summary.py`):
- **e2e µs** = `core0.veccore0` wall duration from opsim log
- **VF cyc** = `VF_SIMT.cycles` in `core0.veccore0_instr_exe.csv`
- **instr#** = body opcodes excl. framing `{set_flag,wait_flag,end_label,end,nop,push_pb,dcci}` and the VF_SIMT wrapper
- **IPC** = `IPC_proxy` = body_instr / VF_SIMT.cycles
- **EX instr#** = sum `call_count` where pipe contains EX/RVECEX, or opcode starts with `simt_`
- **EXIPC** = native EXIPC/IFU row if present; SimtVF dumps usually lack it → `—` (do not invent; same practice as ST_VMI_COMPARE)


See also [`ST_CF_IMPLEMENT.md`](ST_CF_IMPLEMENT.md).

---

## Product cut (keep it simple)

**Primary product cut (2026-10-07):** **CF1–CF3 + CF6** only.  
CF4 / CF5 / CF6b stay on disk as **appendix / historical divergence experiments** — not primary compare narrative. See `SIMT_SIMD_E2E_COMPARE_CF_20261007.md` / `ST_CF_FULL_COMPARE.md`.

| ST | Role | One-line purpose | Twin |
|----|------|------------------|------|
| **CF1** | **primary** | One Parallel `if` vs scalar thresh; scores **KEEP** | Simt + `*v` + `*d` (K=8) |
| **CF2** | **primary** | Same `if`; remat + **kill-in-shared** | Simt + `*v` + `*d` — SIMD beats Simt |
| **CF3** | **primary** | **`remat_idx` only** — kill / select by rematted `i` | Simt + `*v` + `*d` — SIMD beats Simt |
| **CF6** | **primary** | **Newton** + near-0 / fixed N_FAST (`pnear5`; pnear25 flat) | Simt + `*d` rooftop; `*v` = ABI tax |
| CF4 | appendix | Nested if (2-level) — divergence probe | files kept; not primary |
| CF5 | appendix | Div + ULP / near-0 range branch | files kept; not primary |
| CF6b | appendix | Fat-hard / thin-easy Newton+Horner | files kept; not primary |

**Dropped from primary cut:** CF4, CF5, CF6b (appendix only); AABBCC / `vf_fuse` schedule micros (revisit later as CF-S*). Old “KEEP idxs” CF3 arm — SV4 already covers KEEP vs remat; CF3 is remat_idx-only for a simple user-facing CF.

**Why CF6 stays primary (vs CF4/CF5):** clearest numeric CF with a clean `*d` rooftop vs Simt and an explicit `*v` ABI-tax annotation. CF4/CF5/CF6b did not earn a primary divergence-win story at E=256.

---

## Coverage vs SV (short)

| Axis | Owner |
|------|-------|
| RF / remat pressure (no CF) | SV2–SV4, SV8 |
| Packed topk 2-if e2e | SV9 |
| Pure one-if KEEP / shared kill | **CF1 / CF2** |
| Remat index in a CF predicate | **CF3** (SV4 = gather+psum, no CF) |
| Nested / divergent numeric CF | **CF6** primary; CF4/CF5/CF6b appendix |

---

## CF1 — thresh kill KEEP

```python
with T.SimtVF(threads=threads):
    scores = T.alloc_fragment((E,), "float32")
    for i in T.Parallel(E):
        scores[i] = s[i]
    for k in T.serial(K):
        thr = thresh[k]
        for i in T.Parallel(E):
            if scores[i] > thr:
                scores[i] = NEG
```

**Tickle:** predicated rewrite density; membar≈0. Tags: `cf1_e{E}_k{K}_t{T}_keep`.

---

## CF2 — remat + kill-in-shared

```python
with T.SimtVF(threads=threads):
    scores = T.alloc_fragment((E,), "float32")
    for k in T.serial(K):
        thr = thresh[k]
        for i in T.Parallel(E):
            scores[i] = s[i]
        for i in T.Parallel(E):
            if scores[i] > thr:
                s[i] = NEG
```

**Tickle:** same CF, publish tax vs CF1. Tags: `cf2_e{E}_k{K}_t{T}_remat`.

---

## CF3 — remat_idx only

No KEEP idxs buffer. Scalar `victim[k]` (or fixed schedule). Sole Parallel CF uses rematted `i`.

```python
with T.SimtVF(threads=threads):
    scores = T.alloc_fragment((E,), "float32")
    for i in T.Parallel(E):
        scores[i] = s[i]
    for k in T.serial(K):
        v = victim[k]
        for i in T.Parallel(E):
            if i == v:                 # remat index at use
                scores[i] = NEG
```

**Tickle:** index remat inside CF predicate (not RF pressure — SV4 owns that). Tags: `cf3_e{E}_k{K}_t{T}_remat_idx`.

---

## CF4 — nested if (Simt divergence stress)

Two-level predicate. Outer: coarse class; inner: refine. Skew data so most lanes take the short path.

```python
# Illustrative: clamp / classify then refine
with T.SimtVF(threads=threads):
    x = T.alloc_fragment((E,), "float32")
    y = T.alloc_fragment((E,), "float32")
    for i in T.Parallel(E):
        x[i] = a[i]
    for i in T.Parallel(E):
        if x[i] > hi:
            y[i] = hi
        else:
            if x[i] < lo:              # nested
                y[i] = lo
            else:
                y[i] = x[i] * scale[i] # fat path — minority of lanes
```

**Tickle:** Simt can diverge (short-path lanes finish); Simd often pays fat path for the warp/VL.  
**Knobs:** fraction of lanes in fat path (`p_fat` ∈ {0.05, 0.25, 0.5}), E, T/lanes.  
**Tags:** `cf4_e{E}_t{T}_pfat{…}` · twin `cf4v_…`.  
**Compare:** Simt µs vs VMI µs — expect Simt win when `p_fat` small.

---

## CF5 — div with ULP 0.5 + range branch

Division (or recip) with a **near-zero / special** branch; main path aims ~0.5 ULP class behavior (gold = high-prec ref). Branch avoids blow-up or uses a different formula.

```python
# Illustrative — exact ISA/API TBD at impl
with T.SimtVF(threads=threads):
    x = T.alloc_fragment((E,), "float32")
    y = T.alloc_fragment((E,), "float32")
    for i in T.Parallel(E):
        x[i] = a[i]
    for i in T.Parallel(E):
        ax = T.max(x[i], -x[i])
        if ax < eps:                   # near 0 — special path
            y[i] = safe_div_near0(x[i], b[i])
        else:
            y[i] = x[i] / b[i]         # main path; gold checks ~0.5 ULP
```

**Tickle:** range branch + div; Simt wins if few lanes hit near-0.  
**Knobs:** `eps`, `p_near0`, E. Gold: ULP vs fp64 or correctly-rounded ref on main path.  
**Tags:** `cf5_e{E}_t{T}_eps{…}_pnear{…}`.

---

## CF6 — Newton (or similar) with range branch / extra iter

Iterative refine; **near-0 (or bad seed) takes more iters / other formula**; healthy lanes exit early (Simt) vs lockstep max-iters (Simd).

```python
# Illustrative Newton recip / sqrt-style — N_MAX small
with T.SimtVF(threads=threads):
    x = T.alloc_fragment((E,), "float32")
    y = T.alloc_fragment((E,), "float32")
    for i in T.Parallel(E):
        x[i] = a[i]
    for i in T.Parallel(E):
        ax = T.max(x[i], -x[i])
        if ax < eps:
            # other branch: series / scaled Newton / more iters
            y[i] = newton_near0(x[i], n_extra)
        else:
            y[i] = seed_recip(x[i])
            for t in T.serial(N_FAST):  # early-done friendly on Simt
                y[i] = y[i] * (2.0 - x[i] * y[i])
```

**Tickle:** iteration × branch — classic Simt divergence win vs Simd.  
**Knobs:** `eps`, `p_near0`, `N_FAST` vs `n_extra`, E.  
**Tags:** `cf6_e{E}_t{T}_eps{…}_pnear{…}_n{…}`.

---

## Simt vs Simd compare contract (CF4–CF6)

For each tag, run **Simt** (`target=ascend`, deps) and **VMI twin** (`target=pto`, overlay):

| Field | Required |
|-------|----------|
| µs / cycles | both |
| IPC_proxy / EXIPC | both |
| EX opcode highlight | both |
| Winner | Simt / Simd / tie + Δ% |
| Data skew | `p_fat` / `p_near0` used |

Expect: **low skew → Simt ≥ Simd**; **p≈0.5 → gap shrinks**. Document cliffs in `ST_VMI_COMPARE.md` style rows.

---

## First-pass matrix

| Wave | Tags | Question |
|------|------|----------|
| A | `cf1_e256_k{1,8}_t32_keep` | Pure CF KEEP |
| A | `cf2_e256_k8_t32_remat` | Kill publish vs CF1 |
| A | `cf3_e256_k8_t32_remat_idx` | Remat `i` in CF |
| B | `cf4_…_pfat{5,25,50}` | Nested if; Simt win at low pfat? |
| B | `cf5_…_pnear{5,25}` | Div+ULP + near0 branch |
| B | `cf6_…_pnear{5,25}` | Newton + extra iter near0 |

---

## Relation map

```text
Simple user CF
  CF1  thresh if KEEP
  CF2  thresh if + shared kill
  CF3  remat_idx only

Simt-divergence CF (vs Simd twin)
  CF4  nested if
  CF5  div ~0.5 ULP + range branch
  CF6  Newton + near0 / extra iter

Later (not this cut)
  CF-S  AABBCC / vf_fuse schedule
```

## Out of scope now

- AABBCC / vf_fuse (schedule)
- VL128 `num_groups=2`
- Full MoE topk (SV9 / optional linked CF later)

---

## Landing note (2026-09-28 VMI twins)

| Artifact | Path |
|----------|------|
| CF1 / CF1V | `kernels/cf1_pred_thresh_keep.py` / `cf1v_pred_thresh_keep.py` |
| CF2 / CF2V | `kernels/cf2_remat_thresh_kill_shared.py` / `cf2v_…` |
| CF3 / CF3V | `kernels/cf3_remat_idx.py` / `cf3v_remat_idx.py` |
| CF4 / CF4V | `kernels/cf4_nested_if.py` / `cf4v_nested_if.py` |
| CF5 / CF5V | `kernels/cf5_div_ulp_branch.py` / `cf5v_div_ulp_branch.py` |
| CF6 / CF6V | `kernels/cf6_newton_branch.py` / `cf6v_newton_branch.py` |
| Oneshot Simt | `oneshot_cf1_cf6.sh` |
| Oneshot VMI | `oneshot_cf1v_cf6v.sh` |
| Opsim gold | `run_opsim_generic.py` (`cf1_`…`cf6_` + `cf*v_`) |

**VMI CF:** `T.Select` mask (not divergent if); lanes=64; CF5/CF6 Div/Newton hoisted outside SimdVF (ABI). Detail + full table: [`ST_CF_IMPLEMENT.md`](ST_CF_IMPLEMENT.md).

### VMI twin PASS table (2026-09-28)

| case | tag | status | e2e µs | notes |
|------|-----|--------|-------:|-------|
| CF1V | `cf1v_e256_k1_t64_keep` | PASS | 1.17 | Select KEEP |
| CF1V | `cf1v_e256_k8_t64_keep` | PASS | 1.01 | Select KEEP |
| CF2V | `cf2v_e256_k8_t64_remat` | PASS | 1.04 | remat+Select kill |
| CF3V | `cf3v_e256_k8_t64_remat_idx` | PASS | 1.02 | remat i Select |
| CF4V | `cf4v_e256_t64_pfat5` | PASS | 0.93 | nested Select |
| CF4V | `cf4v_e256_t64_pfat25` | PASS | 0.93 | nested Select |
| CF5V | `cf5v_e256_t64_pnear5` | PASS | 6.23 | Div outside VF (ABI tax) |
| CF5V | `cf5v_e256_t64_pnear25` | PASS | 6.23 | same |
| CF6V | `cf6v_e256_t64_pnear5` | PASS | 12.35 | Newton outside VF |
| CF6V | `cf6v_e256_t64_pnear25` | PASS | 12.35 | same |

Simt wins CF5/CF6 (divergence path); VMI competitive on CF1–CF4.
