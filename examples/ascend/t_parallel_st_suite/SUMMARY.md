# SUMMARY — Simt vs SIMD (PTO-DSL) VF results

**Scope:** SV1–SV9, CF1–CF6, SP1–SP6 (Simt) + matching SIMD (PTO-DSL) kernels.  
**Primary metric:** **VF_total** = Σ cycles over every VF launch row in `core0.veccore0_instr_exe.csv`.  
**Source:** opsim Ascend950PR_9599 on pto-b10, harvested 2026-10-07–08 from Wenbo tip `3828bbcf` (see prior `VF_CYCLES_SIMT_VS_DSL_BEST_20261008`).

### Honesty note (this PTO-ISA tip)

Most figures below are **not re-validated** on PTO-ISA/`pto-dev` tip opsim (Simt oneshots still blocked by missing CANN MicroAPI symbols). **Exception:** SP4d/SP5d SIMD VF harvested live on this tip (`fa8ffeeb`, Ascend950PR_9599, 2026-10-08 HK).

---

## SV — vector / RF

| case | shape | Simt arm | Simt VF | SIMD best arm | SIMD VF | ratio | Simt µs | SIMD µs |
|------|-------|----------|--------:|---------------|--------:|------:|--------:|--------:|
| SV1 | E256 | stream | 727 | stream | 59 | 12.32 | 1.21 | 0.89 |
| SV1 | E2048 | stream | 2,625 | — | — | — | 2.58 | — |
| SV2 | R16×C64 | input_keep | COMPILE_FAIL | input_keep | 136 | — | — | 0.92 |
| SV2 | R16×C32 | input_keep (ABI fallback) | 530 | — | — | — | 1.16 | — |
| SV2 | R32×C64 | input_stream ★ | 4,941 | input_stream | 172 | 28.73 | 3.60 | 1.01 |
| SV2 | R32×C64 | fold_scale_reload | 6,411 | input_stream | 172 | 37.27 | 4.42 | 1.01 |
| SV2 | R32×C128 | input_stream | 13,500 | input_stream | 253 | 53.36 | 8.49 | 1.34 |
| SV3 | M24 VL64 K16 | keep | 9,105 | keep | 824 | 11.05 | 6.79 | 3.14 |
| SV3 | M32 VL64 K16 | keep | 11,874 | split_cm16 | 6,469 | 1.84 | 8.63 | 6.88 |
| SV3 | M32 VL64 K16 | split_cm16 ★ | 10,384 | split_cm16 | 6,469 | 1.61 | 7.79 | 6.88 |
| SV4 | E256 B8 | keep_idx | 1,372 | keep_idx | 120 | 11.43 | 1.67 | 1.02 |
| SV4 | E256 B8 | remat_idx_calc ★ | 1,156 | keep_idx | 120 | 9.63 | 1.62 | 1.02 |
| SV5 | R64 C128 G16 | keep_in_rf | 10,641 | ub_stream | 16,846 | 0.63 | 7.04 | 13.06 |
| SV5 | R64 C128 G16 | ub_stream ★ | 5,506 | ub_stream | 16,846 | 0.33 | 4.18 | 13.06 |
| SV6 | R64 C128 G32 | keep_in_warp | 24,697 | reload | 16,522 | 1.49 | 14.83 | 11.66 |
| SV6 | R64 C128 G32 | reload ★ | 20,748 | reload | 16,522 | 1.26 | 12.58 | 11.66 |
| SV7 | R64 C128 G64 | multiwarp_ub ★ | 11,182 | reload | 16,592 | 0.67 | 7.32 | 11.1 |
| SV7 | R64 C128 G64 | reload | 24,339 | reload | 16,592 | 1.47 | 14.54 | 11.1 |
| SV7 | R64 C128 G128 | multiwarp_ub | 25,948 | reload | 16,570 | 1.57 | 15.42 | 10.79 |
| SV7 | R64 C128 G128 | reload ★ | 24,423 | reload | 16,570 | 1.47 | 14.58 | 10.79 |
| SV8 | R64 C128 G16 | live ★ | 28,402 | live | 31,843 | 0.89 | 17.04 | 19.56 |
| SV8 | R64 C128 G16 | spill_dist | 30,545 | live | 31,843 | 0.96 | 18.23 | 19.56 |
| SV9 | E256 K8 | keep ★ | 1,122 | remat_scores | 271 | 4.14 | 1.48 | 1.00 |
| SV9 | E256 K8 | remat_scores | 2,094 | remat_scores | 271 | 7.73 | 2.02 | 1.00 |
| SV9 | E256 K8 | remat_idx | 1,122 | remat_scores | 271 | 4.14 | 1.48 | 1.00 |

**Sensitivity notes**

- **SV1:** stream eltwise baseline; SIMD collapses VF work ~12× at E256.
- **SV2:** input residency (KEEP vs STREAM) + optional fold-scale reload foil; STREAM is primary matched shape.
- **SV3:** Acc KEEP vs split_cm16 chunk flush; at M32 SIMD prefers split.
- **SV4:** idxs+acc KEEP vs remat_idx_calc; remat wins slightly on Simt near spill boundary.
- **SV5–SV7 (Case-3):** RF ladder — small/mid/large G; Simt often wins small-G stream; SIMD reload competitive mid/large.
- **SV8:** quant e2e live vs spill_dist; near parity.
- **SV9:** TopK KEEP / remat_scores / remat_idx; SIMD remat_scores is best schedule.

---

## CF — control flow

| case | shape | Simt arm | Simt VF | SIMD best arm | SIMD VF | ratio | Simt µs | SIMD µs |
|------|-------|----------|--------:|---------------|--------:|------:|--------:|--------:|
| CF1 | E256 K1 | keep | 576 | — (no K1 SIMD) | — | — | 1.15 | — |
| CF1 | E256 K8 | keep | 309 | keep | 105 | 2.94 | 0.97 | 0.90 |
| CF2 | E256 K8 | remat+shared-kill | 2,041 | remat | 113 | 18.06 | 1.95 | 0.90 |
| CF3 | E256 K8 | remat_idx | 851 | remat_idx | 105 | 8.10 | 1.33 | 0.83 |
| CF4 | E256 pfat5 | nested_if | 758 | nested_if | 70 | 10.83 | 1.23 | 0.89 |
| CF4 | E256 pfat25 | nested_if | 781 | nested_if | 70 | 11.16 | 1.25 | 0.89 |
| CF5 | E256 pnear5 | div_ulp | 657 | div_ulp | 74 | 8.88 | 1.18 | 0.89 |
| CF5 | E256 pnear25 | div_ulp | 656 | div_ulp | 74 | 8.86 | 1.18 | 0.89 |
| CF6 | E256 pnear5 | newton | 961 | newton | 129 | 7.45 | 1.37 | 0.93 |
| CF6 | E256 pnear25 | newton | 961 | newton | 129 | 7.45 | 1.37 | 0.93 |

**Sensitivity notes**

- **CF1:** one Parallel if vs scalar thresh; scores KEEP — near-parity walls, SIMD still fewer VF cycles.
- **CF2:** remat scores + shared-kill publish tax hits Simt hardest (ratio ~18).
- **CF3:** index remat inside CF predicate (not RF pressure).
- **CF4:** nested if; pfat skew flat on both sides.
- **CF5:** Div + near-0 ULP branch; pnear flat.
- **CF6:** Newton recip + near-0; pnear flat; SIMD is the vector rooftop (not a Simt divergence win).

---

## SP — sparse / packing

| case | shape | Simt arm | Simt VF | SIMD best arm | SIMD VF | ratio | Simt µs | SIMD µs |
|------|-------|----------|--------:|---------------|--------:|------:|--------:|--------:|
| SP1 | T32 K2 H128 G32 | keep_pos | 19,881 | keep_pos (vscatter) | 31,424 | 0.63 | 12.63 | 19.04 |
| SP1 | T32 K2 H128 G32 | remat_pos ★ | 17,290 | keep_pos (vscatter) | 31,424 | 0.55 | 11.19 | 19.04 |
| SP2 | T32 K2 H128 G32 | sf0_w0+keep_acc ★ | 13,738 | sf0_w0+keep_acc | 837 | 16.41 | 9.19 | 2.02 |
| SP2 | T32 K2 H128 G32 | sf1_w1+keep_acc | 23,958 | sf0_w0+keep_acc | 837 | 28.62 | 14.88 | 2.02 |
| SP2 | T32 K2 H128 G32 | sf1_w1+remat_acc | 37,255 | sf0_w0+keep_acc | 837 | 44.51 | 22.27 | 2.02 |
| SP2 | T32 K2 H128 G32 | sf0_w0+remat_acc | 18,944 | sf0_w0+keep_acc | 837 | 22.63 | 12.08 | 2.02 |
| SP3 | M32 H128 G32 fp32 | fp32 | 11,158 | fp32 | 349 | 31.97 | 7.45 | 1.44 |
| SP3 | M32 H128 G32 e8m0 | e8m0 | 11,171 | e8m0 | 311 | 35.92 | 7.45 | 1.39 |
| SP4 | E64 H128 pad25 | pad25 | 13,968 | mask | 265 | 52.71 | 9.63 | 1.98 |
| SP5 | N64 H128 G32 QG32 | sideband ★ | 52,001 | sideband | 583 | 89.19 | 30.44 | 1.87 |
| SP5 | N64 H128 G32 QG32 | interleave | 54,655 | interleave | 699 | 78.19 | 31.92 | 1.94 |
| SP6 | N32 H128 G32 | unpack ★ | 7,813 | unpack+vselr ★ | 393 | 19.88 | 5.35 | 1.22 |
| SP6 | N32 H128 G32 | unpack_sf | 18,727 | unpack_sf+vselr ★ | 707 | 26.49 | 11.42 | 1.40 |

**SIMD SP6 arms (same shape):** unpack gather 534 / vselr **393**; unpack_sf gather 826 / vselr **707**.  
**SP2 SIMD note:** keep_acc ≡ remat_acc on VF (837 @ sf0, 2301 @ sf1) — Acc remat cliff is Simt-side; sf0→sf1 tax survives on SIMD (837→2301).

**Sensitivity notes**

- **SP1:** Pos KEEP vs remat under dual live (V,Sf) co-scatter. Simt remat_pos wins VF; SIMD keep_pos is vscatter path (higher VF here).
- **SP2:** Acc KEEP/remat × sf0_w0/sf1_w1 compute tax. Simt Acc remat cliffs hard; SIMD collapses Acc remat.
- **SP3:** fp32 SF load vs A5 bit-reinterpret e8m0 compose (no soft Pow2 LUT; no native e8m0 vcvt on A5).
- **SP4:** pad gather predication. Simt early-skips `Expert[p]<0`; SIMD (SP4d) walks every row and zeros holes with a full-lane `vsel` mask. SIMD VF **265** @ 1.98 µs (ratio ~53× vs Simt pad25).
- **SP5:** sideband gathers `V[slot]` and `Sf[slot]` as two tensors; interleave gathers one `Pack[slot]` then splits the scale tail. Same gold `Out[q,j]=V[slot,j]*Sf[slot,j//G]`. SIMD sideband VF **583** @ 1.87 µs (★, ratio ~89×); interleave VF **699** @ 1.94 µs (ratio ~78×).
- **SP6:** soft 4-bit e2m1 LUT unpack (± SF); SIMD ★ = **vselr** (needs VL-padded table).

---

## Coverage

| Layer | Count | Notes |
|-------|------:|-------|
| Simt kernels | 21 | SV1–9, CF1–6, SP1–6 |
| SIMD (PTO-DSL) kernels | 21 | SV1d–9d, CF1d–6d, SP1d–6d |
| Intended product | 42 | 21 Simt + 21 SIMD; SP4d/SP5d VF harvested 2026-10-08 |

Deck: [`docs/st-deck-v2.pptx`](docs/st-deck-v2.pptx).
