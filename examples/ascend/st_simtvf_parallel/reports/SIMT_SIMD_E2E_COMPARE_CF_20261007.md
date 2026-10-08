# SimtVF vs Simd/VMI (*d / *v) e2e compare — CF sensitivity (primary cut)

**Date:** 2026-10-07 HKT  
**SOC:** Ascend950PR_9599  
**Product cut:** **CF1–CF3 + CF6** (primary). CF4 / CF5 / CF6b = appendix / historical divergence experiments — not primary.  
**Layout twin of:** `reports/SIMT_SIMD_E2E_COMPARE_20261007.md` / `simt_simd_e2e_20261007.tsv`  
**Sources:**
- Simt A + Layer-B `*v`: `reports/ST_CF_IMPLEMENT.md` (2026-09-28 harvest; VF cyc / body_instr / IPC)
- Sensitivity aims: `reports/ST_CF1_CF6_SLICES.md`
- Layer-D `*d` walls: `ST_CF1D_CF6D_SUMMARY.md`, `ST_CF2D_CF4D.md`, `ST_CF5D_CF6D.md` (2026-10-07; wall µs only)
- CF6b/CF6bd: `reports/cf6b_cf6bd_20261007/RESULT.md` + dig logs (appendix only)
- Divergence framing: `ST_CF5_CF6_DIVERGENCE.md`, `ST_CF6_DIVERGENCE_CASE.md` (appendix)

## Product framing

| Layer | Role in this table |
|-------|--------------------|
| **Simt (A)** | Control-flow sensitivity baseline |
| **`*d` (Layer-D / PTO-DSL)** | **Primary SIMD/VMI rooftop** (wall µs; C/instr/IPC often `—` this pass) |
| **`*v` (Layer-B / SimdVF)** | Shown **where PASS**; competitive on CF1–CF3; **ABI tax** on CF6 (Newton outside SimdVF) |

**When SIMD beats Simt (primary):**

| case | Winner vs Simt | Takeaway |
|------|----------------|----------|
| **CF2** remat+shared-kill | `*d` 0.90 µs and `*v` 1.04 µs ≪ Simt 1.95 µs | remat+publish tax hits Simt hardest |
| **CF3** remat_idx | `*d` 0.83 µs and `*v` 1.02 µs ≪ Simt 1.33 µs | index remat in CF predicate; SIMD wins wall |
| **CF6** Newton / near-0 | `*d` 0.92 µs ≪ Simt 1.37 µs | **vector rooftop**; not a Simt divergence win |
| **CF6 `*v`** | Simt 1.37 µs ≪ `*v` 12.35 µs | **ABI tax** (Newton hoisted outside SimdVF) — do **not** read as divergence beating vector |

CF1 KEEP is near-parity (Simt / `*v` / `*d` ≈ 0.97 / 1.01 / 0.90 µs at K=8). CF6 `pnear25` is **flat** vs `pnear5` on all layers (one row below; see appendix note).

## CF2 vs CF3 (sensitivity clarity)

| | **CF2** | **CF3** |
|--|---------|---------|
| Predicate | `scores[i] > thr` (same thresh-if as CF1) | `i == victim[k]` (index equality) |
| What remats | **scores** each K from shared | lane index **`i`** at use (no KEEP idxs) |
| Kill / write home | kill into **shared `s`** (publish tax) | update **scores** via Select / if |
| Sensitivity | remat + shared-kill cost vs CF1 KEEP | index remat *inside* CF predicate (not RF pressure — SV4) |

## Metric recipes

- **wall_us** = `core0.veccore0` duration_time(us)
- **Simt (A):** `C_meas` = `VF_SIMT.cycles`; `body_instr` = Σ call_count excl. framing + VF wrapper; `IPC` = body_instr / C_meas
- **Layer-B `*v`:** `C_meas` = `VF` / `VF_SIMD` cycles (or Σ RVEC if that is what harvest used); `IPC` = body_instr / C_meas
- **Layer-D `*d`:** `C_meas` = RVEC pipe sum when harvested; EXIPC often `—`. **This pass:** CF `*d` harvest is **wall µs only** → C_meas / body_instr / IPC = `—` (do not invent)
- Layer-D `*d` is the primary VMI rooftop column. Layer-B `*v` only where PASS.
- Never invent numbers for missing harvest / no twin.

## Primary table — CF1–CF3 + CF6

| case | sensitivity | Simt st | Simt µs | Simt C | Simt instr | Simt IPC | *d st | *d µs | *d C | *d instr | *d IPC | *v st | *v µs | *v C | *v instr | *v IPC | Note |
|------|-------------|---------|--------:|-------:|-----------:|---------:|-------|------:|-----:|---------:|-------:|-------|------:|-----:|---------:|-------:|------|
| CF1 | One Parallel if vs scalar thresh; scores KEEP (K=1) | PASS | 1.15 | 576 | 137 | 0.238 | — | — | — | — | — | PASS | 1.17 | 577 | 106 | 0.184 | KEEP; no CF1d k1 primary |
| CF1 | One Parallel if vs scalar thresh; scores KEEP (K=8) | PASS | 0.97 | 309 | 256 | 0.828 | PASS | 0.90 | — | — | — | PASS | 1.01 | 301 | 308 | 1.023 | KEEP thresh kill; *d wall-only; near-parity |
| CF2 | Same if; remat + kill-in-shared | PASS | 1.95 | 2041 | 362 | 0.177 | PASS | 0.90 | — | — | — | PASS | 1.04 | 356 | 425 | 1.194 | **SIMD beats Simt** (remat+publish) |
| CF3 | remat_idx only (kill/select by rematted i) | PASS | 1.33 | 851 | 272 | 0.320 | PASS | 0.83 | — | — | — | PASS | 1.02 | 321 | 380 | 1.184 | **SIMD beats Simt** (remat i in CF) |
| CF6 | Newton recip + near-0 / fixed N_FAST; pnear=5% | PASS | 1.37 | 961 | 192 | 0.200 | PASS | 0.92 | — | — | — | PASS | 12.35 | 463 | 6746 | 14.570 | ***d* beats Simt**; ***v* = ABI tax** (not divergence win). pnear25 flat |

## Tag key (primary)

| case | Simt tag | *v tag | *d tag |
|------|----------|--------|--------|
| CF1 K1 | `cf1_e256_k1_t32_keep` | `cf1v_e256_k1_t64_keep` | `— (no k1 *d)` |
| CF1 K8 | `cf1_e256_k8_t32_keep` | `cf1v_e256_k8_t64_keep` | `cf1d_e256_k8_t32_keep` |
| CF2 | `cf2_e256_k8_t32_remat` | `cf2v_e256_k8_t64_remat` | `cf2d_e256_k8_t32_remat` |
| CF3 | `cf3_e256_k8_t32_remat_idx` | `cf3v_e256_k8_t64_remat_idx` | `cf3d_e256_k8_t32_remat_idx` |
| CF6 pnear5 | `cf6_e256_t32_pnear5` | `cf6v_e256_t64_pnear5` | `cf6d_e256_t32_pnear5` |

## Coverage / honesty notes (primary)

- **CF1:** KEEP near-parity across layers at K=8; CF1d K=1 not in primary matrix.
- **CF2 / CF3:** clearest **SIMD beats Simt** stories (remat / remat_idx); `*d` is fastest wall.
- **CF6:** `*d` is vector rooftop vs Simt; `*v` wall is **ABI tax** (Newton outside SimdVF) — not evidence that divergence beats vector. `pnear` skew is **flat** at E=256.
- **Missing IPC for `*d`:** opsim dig on box has wall logs only for CF Layer-D. Marked `—`.

---

## Appendix — divergence experiments (not primary)

CF4 / CF5 / CF6b remain on disk (kernels + reports) as **historical / divergence probes**. They are **not** part of the product primary compare. Skew knobs (`pfat`, `pnear`, `phard`) stayed flat; CF6b failed divergence-win criteria vs CF6bd.

| case | sensitivity | Simt st | Simt µs | Simt C | Simt instr | Simt IPC | *d st | *d µs | *d C | *d instr | *d IPC | *v st | *v µs | *v C | *v instr | *v IPC | Note |
|------|-------------|---------|--------:|-------:|-----------:|---------:|-------|------:|-----:|---------:|-------:|-------|------:|-----:|---------:|-------:|------|
| CF4 | Nested if (2-level); p_fat=5% skew | PASS | 1.23 | 758 | 212 | 0.280 | PASS | 0.89 | — | — | — | PASS | 0.93 | 147 | 136 | 0.925 | appendix; pfat flat |
| CF4 | Nested if (2-level); p_fat=25% skew | PASS | 1.25 | 781 | 212 | 0.271 | PASS | 0.89 | — | — | — | PASS | 0.93 | 147 | 136 | 0.925 | appendix; pfat flat |
| CF5 | Div + near-0 range branch (ULP); pnear=5% | PASS | 1.18 | 657 | 149 | 0.227 | PASS | 0.89 | — | — | — | PASS | 6.23 | 529 | 3211 | 6.070 | appendix; *v ABI tax (Div) |
| CF5 | Div + near-0 range branch (ULP); pnear=25% | PASS | 1.18 | 656 | 149 | 0.227 | PASS | 0.89 | — | — | — | PASS | 6.23 | 529 | 3211 | 6.070 | appendix; pnear flat |
| CF6 | Newton recip + near-0 / fixed N_FAST; pnear=25% | PASS | 1.37 | 961 | 192 | 0.200 | PASS | 0.93 | — | — | — | PASS | 12.35 | 463 | 6746 | 14.570 | flat twin of primary pnear5 |
| CF6b | Fat-hard / thin-easy Newton+Horner; phard=5% | PASS | 2.02 | — | — | — | PASS | 1.08 | — | — | — | — | — | — | — | — | appendix; no *v; loses to *d |
| CF6b | Fat-hard / thin-easy Newton+Horner; phard=25% | PASS | 2.02 | — | — | — | PASS | 1.08 | — | — | — | — | — | — | — | — | appendix; phard flat; divergence fail |

**Appendix tags:** `cf4_*_pfat{5,25}`, `cf5_*_pnear{5,25}`, `cf6_*_pnear25`, `cf6b_*_phard{5,25}` / `cf6bd_*`. Detail: `ST_CF5_CF6_DIVERGENCE.md`, `ST_CF6_DIVERGENCE_CASE.md`, `ST_CF6B_RESULT.md`.

## Artifact paths

- TSV: `reports/simt_simd_e2e_cf_20261007.tsv`
- MD: `reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md`
- Also mirrored: `reports/ST_CF_FULL_COMPARE.md`
