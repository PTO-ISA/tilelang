# ST RF KEEP vs STREAM/RELOAD audit

**Date:** 2026-09-29 ~10:30 HKT  
**Owner feedback (Lok):** TileLang Simt `T.Parallel` hides RF/UB; VMI/ptodsl must make **programmer choice** of what stays in RF vs what streams/reloads **source-visible**. Benchmarks must hit *turning points* (KEEP scales across reuse; STREAM row data when RF budget / reuse pattern demands) — not “append every live value into `rows[]` until TILE fits”.

**Contract**
| Verb | Meaning |
|------|---------|
| **KEEP** | Live set stays in RF/vregs across the reuse that needs it |
| **STREAM** | Load → use → drop; if needed again, **reload from UB** |
| **REMAT/RELOAD** | Explicit re-materialize from shared/UB (scale, index, scores, …) |

## Summary table

| Case | Layer | SHOULD KEEP | SHOULD STREAM/RELOAD | Turning point | Source OK? | Rewrite? |
|------|-------|-------------|----------------------|---------------|------------|----------|
| SV1 | A/B/D | nothing loop-carried | A,B chunks stream ld→add→st | baseline stream | **OK** | N |
| SV2 | A | fold `scale[R,LANES]` fragment ALWAYS | x KEEP@R16 / STREAM@R≥32 | input KEEP vs STREAM by R | **OK** (2026-09-29) | **Y → done** input_keep/stream |
| SV2d old `frag_live`/`TILE rows[]` | D | *(wrong)* all rows+scales in TILE | — | accidental fits-in-RF | **WRONG** | **Y → done** `scale_keep_stream` |
| SV2d `input_stream` / `fold_scale_keep` | D | scale VL ALWAYS | rows STREAM vload | scale KEEP + row STREAM | **OK** fold rewrite | N |
| SV2d `full_reload` | D | — | rows + scales from UB | remat tax | **OK** (C32 hist 1.33; **C128 primary 1.47 µs**) | N |
| SV2d `row_keep_tile` | D | rows+scales in TILE | — | *negative* example | OK as labeled NEG (C32 hist 1.06; C128 tile 1.53; **C128 row_keep_full FAIL spill**) | N (keep as foil) |
| SV3 | A/B | `Acc[M,VL]` across K | A tile / x reload each K | Acc RF cliff / split | **OK** (intent) | N (*d missing*) |
| SV4 | A/B/D | `acc[E]` always; `idxs` on keep_idx | idxs remat on remat_idx; A[b] stream | KEEP idxs vs remat while Acc live | **OK** | N |
| SV5/6/7 | A/B | reduce temps per (i,g) | X from shared serial(G) | G width stress | **OK** (stream X) | N (*d missing*) |
| SV8 | A/B | `sf_inv` on `live` | `spill_dist` expands to shared then reload | KEEP reduced vs layout-spill | **OK** | N (*d missing*) |
| SV9 | A/B/D | scores (+idxs on keep) | remat_scores reload scores; remat_idx remat i | KEEP scores/idxs vs remat | **OK** | N |
| CF1 | A/B/D | scores across K | thresh scalar stream | KEEP scores + kill | **OK** | N |
| CF2 | A/B | — (remat scores) | scores from shared each k; kill in shared | remat tax vs CF1 | **OK** | N (*d missing*) |
| CF3 | A/B | scores | remat Parallel index i | remat_idx | **OK** | N (*d missing*) |
| CF4/5/6 | A/B | working x stream/eltwise | branch divergence / ULP / Newton | CF shape not RF keep | **OK** (CF stress) | N |
| SP1 | A / D keep_pos | V/Sf across K; pos_row snapshot | Expert pad; scatter writes | KEEP QuantTensor + pos | **OK** | N (remat_pos *d missing*) |
| SP2 | A | (pos keep arm) | gather stream | dual gather+reduce | **OK** intent | N (*d missing*) |
| SP3–SP6 | A | layout/decode working set | pack/unpack stream | format/layout not RF ladder | **OK** as format STs | N |

\* CF7 not present in suite (CF1–CF6 only).

## Per-family notes

### SV — RF ladder core
- **SV1d:** pure stream — correct baseline.
- **SV2d:** was WRONG (TILE `rows[]` KEEP-everything). Rewritten 2026-09-29 to `scale_keep_stream` / `full_reload` (+ negative `row_keep_tile`).
- **SV3:** Acc KEEP / inputs reload — strong contract; no *d yet.
- **SV4d:** keep_idx vs remat_idx with Acc KEEP — **OK**.
- **SV5–7:** absmax reduce streaming X — OK; not keep/stream contrast arms.
- **SV8:** live vs spill_dist — OK keep/stream of reduced scale.
- **SV9d:** keep / remat_scores / remat_idx — **OK**.

### CF
- **CF1d:** scores KEEP — **OK**.
- **CF2:** remat twin of CF1 — OK; no *d.
- **CF3–6:** remat_idx / nested / ULP / Newton — CF sensitivity, not RF pack; OK.

### SP
- **SP1d:** V/Sf KEEP + pos snapshot + stream scatter into *exp — **OK** for keep_pos. remat_pos *d not built.
- **SP2–6:** access/format STs; RF keep secondary.

## *d rewrite queue
| Twin | Action |
|------|--------|
| SV2d | **DONE** — C128/C256 cliff: scale_keep_stream / full_reload PASS; row_keep_full FAIL spill |
| SV4d | No change |
| SV9d | No change |
| CF1d | No change |
| SP1d | No change (optional later: remat_pos arm) |
| SV3/5/6/7/8, CF2–6, SP2–6 | No *d yet — when added, follow KEEP/STREAM contract above |

## Simt A gaps (note only)
SV2 A: scale ALWAYS `alloc_fragment`; arms `input_keep` (R16, x+scale frag) vs `input_stream` (R≥32, stream x). Twin SV2v uses shared for VF working (VMI ABI) — structure matches, cannot claim fragment RF.


## SV2d C≥128 RF cliff (2026-09-29 ~12:15 HKT)

**VF stack (measured):** ptoas fatal `exceeded vf stack size (6144)` — budget **6144 B**.

| Geom | scale_keep_stream | full_reload | row_keep_full | row_keep_tile |
|------|-------------------|-------------|---------------|---------------|
| R32×C128 | PASS **1.51 µs** | PASS **1.47 µs** | **FAIL** spill 14368 | PASS 1.53 µs |
| R32×C256 | PASS **2.25 µs** | PASS **2.22 µs** | **FAIL** spill 36128 | PASS 2.22 µs |

RF math: VL=64 f32; C128 → 64 vregs×256B=16384B; C256 → 128 vregs=32768B; scale_keep ≈ (1+NCH)×256 B.

**Verdict:** turning point exposed — KEEP-scales+STREAM-rows legal; full `rows[]` KEEP illegal at C≥128. `row_keep_full` uses cooked (`x²+eps`) force-live so compiler cannot plain-remat UB loads past the cliff.


## SV2d scale_ub_membar (2026-09-29 ~13:55 HKT)

**Why KEEP lost to full_reload:** remat without `mem_bar` is under-taxed; KEEP pack-select raises RVECEX.

**Fix arm:** `scale_ub_membar` — UB publish + `mem_bar(VST_VLD)` (+ per-use bar) + reload scale; STREAM rows.

| C | KEEP wall | full_reload | scale_ub_membar | SMEM_BAR |
|--:|----------:|------------:|----------------:|----------|
| 64 | 1.18 | 1.10 | **1.79** | 33/1442 |
| 128 | **1.51** | 1.47 | **2.16** | 33/1446 |
| 256 | **2.25** | 2.22 | **2.90** | 33/1521 |

KEEP beats UB+membar remat on wall; EXIPC absent in ptodsl instr dumps (report —).

## SV2 input KEEP vs STREAM — scale ALWAYS KEEP (2026-09-29 ~14:45 HKT)

**Correction (Lok):** Scale ALWAYS KEEP in register. Do NOT use scale remat/reload as the primary turning point. Sensitivity is **input (x/rows) KEEP vs STREAM by R**.

### Contract

| Verb | What |
|------|------|
| Scale | ALWAYS KEEP — A: `alloc_fragment((R,LANES))`; D: VL vreg from vmax fold |
| Input KEEP | R≈16: x/rows live with scale (fits RF) |
| Input STREAM | R≥32 (32/64): x must STREAM/reload; scale still KEEP |

### Mapping A ↔ D

| Layer A (Simt) | Layer D (ptodsl) | Role |
|----------------|------------------|------|
| `input_keep` (R≈16) | `input_keep` (R≈16) | KEEP x/rows + KEEP scale |
| `input_stream` / `frag_live` (R≥32) | `input_stream` / `fold_scale_keep` (R≥32) | STREAM x/rows; KEEP scale |
| `reload` (demoted) | `fold_scale_reload` (demoted) | legacy scale-remat foil — not primary |

### A snippet (`input_stream` — scale fragment KEEP, x STREAM)

```python
scale = T.alloc_fragment((R, LANES), "float32")  # ALWAYS KEEP
for i, lane in T.Parallel(R, LANES):
    m = alloc_var(EPS)
    for ch in serial(NCH):
        m = max(m, abs(x_ub[i, ch*LANES+lane]))
    scale[i, lane] = max(m, EPS)
for i, lane in T.Parallel(R, LANES):
    s = scale[i, lane]            # from fragment
    for ch in serial(NCH):
        out_ub[i, j] = x_ub[i, j] / s   # STREAM x
```

`input_keep` adds `x = alloc_fragment((R,C))` filled during fold; consumer reads `x[i,j]/s`.

### RF math (D, C=64 NCH=1, VL=64 f32, stack 6144 B)

| Arm | R | live row vregs | bytes | fits? |
|-----|--:|---------------:|------:|:-----:|
| input_keep | 16 | 16 | 4096 | yes |
| input_keep | 32 | 32 | 8192 | **no** (cliff) |
| input_stream | 32/64 | ~1–2 + scale | ~768 | yes |

### Opsim targets

A: `sv2_r16_c64_t32_input_keep`, `sv2_r32_c64_t32_input_stream`, `sv2_r64_c64_t32_input_stream`
D: `sv2d_r16_c64_t32_input_keep`, `sv2d_r32_c64_t32_input_stream`, `sv2d_r64_c64_t32_input_stream` (+ optional R32 input_keep spill foil)


### Opsim Ascend950PR_9599 (2026-09-29 ~14:48 HKT)

**Layer D (ptodsl)**

| Tag | Result | wall µs | notes |
|-----|--------|--------:|-------|
| `sv2d_r16_c64_t32_input_keep` | **PASS** | **0.91** | primary small-R KEEP |
| `sv2d_r32_c64_t32_input_stream` | **PASS** | **1.01** | primary stream |
| `sv2d_r64_c64_t32_input_stream` | **PASS** | **1.38** | large-R stream |
| `sv2d_r32_c128_t32_input_stream` | **PASS** | **1.34** | C128 stream |
| `sv2d_r32_c64_t32_input_keep` | PASS | 1.05 | RF math 8192>6144 but no ptoas spill (cook may remat) |

**Layer A (Simt)**

| Tag | Result | wall µs | notes |
|-----|--------|--------:|-------|
| `sv2_r16_c32_t32_input_keep` | **PASS** | **1.16** | primary A KEEP (x+scale frag) |
| `sv2_r32_c32_t32_input_stream` | **PASS** | **2.41** | stream x; scale frag KEEP |
| `sv2_r32_c64_t32_input_stream` | **PASS** | **3.60** | C64 NCH=2 stream |
| `sv2_r64_c64_t32_input_stream` | **PASS** | **8.34** | large-R stream |
| `sv2_r16_c64_t32_input_keep` | **COMPILE_FAIL** | — | layout: no available layout (x frag R16×C64 + scale) |

**x in fragment:** only on `input_keep` (by design). `input_stream` STREAMs x from x_ub — no compile blocker; scale-only fragment KEEP works.
