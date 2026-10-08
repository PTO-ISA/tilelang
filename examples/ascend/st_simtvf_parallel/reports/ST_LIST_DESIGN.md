# SimtVF Parallel ST list — design (SV1–SV9 contiguous)

**Date:** 2026-09-24 ~14:30 HKT (VMI twin rules)  
**Suite:** `/workspace/st_simtvf_parallel`  
**Status:** Contiguous **SV1–SV9** (no SV1B/SV2G aliases). Legacy micros in `kernels/_removed_legacy/`.

**Suite contract (every ST):**
1. **Purpose / sensitivity** — what RF / Parallel / remat behavior is probed.
2. **GPU vs NPU expectation** — GPU MRF vs NPU SimtVF folded RF / layout.
3. **Turning-point knobs** — parameters that cross a cliff.
4. **Artifacts** — source / TIR / thread-block RF map / µs+IPC (Ascend950PR_9599).

Harness: SimtVF = `target=ascend` + cython + deps-native. VMI twins = `target=pto` + overlay — **never mix**.

---

## Locked master map (SV1–SV9)

| New | Content | Former | Status |
|-----|---------|--------|--------|
| **SV1** | Stream Parallel eltwise | SV1 | GREEN |
| **SV2** | Live RF eltwise + scale bcast (`frag_live` / `reload`) | SV1B | GREEN |
| **SV3** | GEMV Acc KEEP / split | SV2G | GREEN |
| **SV4** | Index gather + psum (`keep_idx` / `remat_idx`) | replaced keep2 | sources ready; µs TBD |
| **SV5** | Case-3 small-G reduce→reduced eltwise (G=16) | SV5 | GREEN |
| **SV6** | Case-3 mid-G reduce→reduced eltwise (G=32) | **NEW** mid | sources ready; µs TBD |
| **SV7** | Case-3 large-G reduce→reduced eltwise (G=64, try 128) | old SV6 | GREEN (as old SV6 tags) |
| **SV8** | Case-3 quant e2e live / spill_dist | SV8 | GREEN |
| **SV9** | Topk e2e keep / remat_scores / remat_idx | planned; from legacy topk | sources ready; µs TBD |
| **SV5V/6V/7V/8V** | VMI twins of Case-3 | — | sources; opsim pending |

**Two e2e finals:** SV8 = quant / group-scale; SV9 = MoE-style topk.

---

## Tag conventions

| ST | Tag pattern |
|----|-------------|
| SV1 | `sv1_e{E}_t{T}` |
| SV2 | `sv2_r32_c32_t32_{frag_live,reload}` |
| SV3 | `sv3_m{M}_vl{VL}_k{K}_t{T}_{keep,split_cm{CM}}` |
| SV4 | `sv4_e*_b*_t*_{keep_idx,remat_idx}` (+ VMI `sv4v_*_t64_*`) |
| SV5 | `sv5_r64_c128_g16_t32` |
| SV6 | `sv6_r64_c128_g32_t32` |
| SV7 | `sv7_r64_c128_g64_t32` (+ optional g128) |
| SV8 | `sv8_r64_c128_g16_t32_{live,spill_dist}` |
| SV9 | `sv9_e256_k{1,8}_t32_{keep,remat_scores,remat_idx}` (+ optional t128 keep) |
| SP1 | `sp1_t{T}_k{K}_h{H}_g{G}_t{Thr}_{keep_pos,remat_pos}` |
| SP2 | `sp2_t{T}_k{K}_h{H}_g{G}_t{Thr}_{sf0,sf1}_{w0,w1}_{keep_pos,remat_pos}` |
| SP3 | `sp3_m{M}_h{H}_g{G}_t{Thr}_{fp32,ue8m0}` |
| SP4 | `sp4_e{Eexp}_h{H}_t{Thr}_pad{Pct}` |
| SP5 | `sp5_n{N}_h{H}_g{G}_qg{Qg}_t{Thr}_{sideband,interleave}` |
| SP6 | `sp6_n{N}_h{H}_g{G}_t{Thr}_{unpack,unpack_sf}` |

---

## ST-SV1 — Stream Parallel eltwise

**Purpose:** Baseline Parallel streak: `t3 = t1 + t2`, no carry, no bcast.

| concern | GPU | NPU SimtVF |
|---------|-----|------------|
| Live state | Temps die each elem | `t1,t2,t3` fold to `[E/T]` |
| Cliff | Large live tile if footprint grows | E↑ → elems/thread RF pressure |

**Tags:** `sv1_e{E}_t{T}`. **Measured:** e256_t32 1.21 µs; e2048_t32 2.58 µs.

---

## ST-SV2 — Live RF eltwise + scale bcast (ex-SV1B)

**Purpose:** Keep per-row `scale[R]` in RF during `out = x / scale[i]` vs shared `reload`.

| concern | GPU | NPU SimtVF |
|---------|-----|------------|
| Live scales | Regs; no LDS reload | Fragment RF VL-folded |
| Reload | LDS round-trip | Shared publish + load tax |

**Tags:** `sv2_r32_c32_t32_{frag_live,reload}`.  
**Measured (as old sv1b_*):** frag_live 4.93 µs; reload 4.24 µs (~14% faster).

**Kernel:** `kernels/sv2_eltwise_bcast_rf.py`

---

## ST-SV3 — GEMV Acc KEEP / split (ex-SV2G)

**Purpose:** Loop-carried `Acc[M,VL]` KEEP across K; A/`x` reload. `split` = Acc chunk flush.

| concern | GPU | NPU SimtVF |
|---------|-----|------------|
| Acc KEEP | MRF holds Acc | Fold `M*VL/T` elems/thread; expect KEEP @ 64 fp32/thread |
| Split | Chunk + flush | Membar O(ceil(M/CM)) not O(K) |

**Tags:** `sv3_m{24,32}_vl64_k16_t32_{keep,split_cm16}`. GREEN (as old sv2g_*).  
**Kernel:** `kernels/sv3_gemv_partial_keep.py`

---

## ST-SV4 — Index gather + partial-sum across batch B

**Purpose:** Non-identity VCI `Idx[E]` gather while `acc[E]` KEEP across `serial(B)`. Arms: KEEP idxs+acc (`keep_idx`) vs remat Idx each gather (`remat_idx`). No topk CF.

| concern | GPU | NPU SimtVF / VMI |
|---------|-----|------------------|
| Index KEEP vs remat | Keep VCI RF vs remat at use | Live idxs vs remat from shared Idx |
| Acc KEEP | Partial sum across batch | `acc[E]` live across B (mode 3) |
| vs SV9 | No CF; tensor Idx gather | Topk CF + kill lives in SV9 |

**Tags:** `sv4_e256_b{8,16}_t32_{keep_idx,remat_idx}` · VMI `sv4v_e256_b{8,16}_t64_{keep_idx,remat_idx}`. Sources ready; µs TBD.  
**Kernels:** `kernels/sv4_index_gather_psum.py` / `sv4v_index_gather_psum.py`  
**Detail:** [`ST_SV4_INDEX_GATHER_PSUM.md`](ST_SV4_INDEX_GATHER_PSUM.md) (replaced keep2).

---

## ST-SV5 / SV6 / SV7 — Case-3 reduced-only (small / mid / large G)

**Purpose:** Group absmax → `sf_inv` on `[R,CG]` only (no bcast). Contiguous G ladder.

Math: `Y[i,g] = 1 / max(absmax(X[i,g*G:(g+1)*G]), 1e-6)`.

| ST | G | CG @C=128 | Role |
|----|---|-----------|------|
| SV5 | 16 | 8 | small-G |
| SV6 | 32 | 4 | **NEW mid** |
| SV7 | 64 (try 128) | 2 (1) | large-G / CG collapse |

| concern | GPU | NPU SimtVF |
|---------|-----|------------|
| Scale RF | Regs for `sf_inv` | VL-fold |
| Reduce | Serial G | Prefer Parallel(R,CG)+serial(G); no `j//G` in frag |
| Bcast | Deferred to SV8 | Same |

**Measured:** SV5 9.34 µs; SV7(=old SV6) G64 14.44 / G128 14.69 µs; **SV6 mid TBD**.

---

## ST-SV8 — Final quant e2e (Case-3)

**Purpose:** Full group-quant pipeline: absmax → `sf_inv[R,CG]` → `Out[i,j] = X[i,j] * sf_inv[i, j//G]`.

Arms:
- **`live`** — keep `sf_inv` in frag RF; consumer Parallel(R,CG)+serial(G).
- **`spill_dist`** — layout-expand `[R,CG]→[R,C]` in shared, then VL-aligned reload.

**Tags:** `sv8_r64_c128_g16_t32_{live,spill_dist}`.  
**Measured:** live 17.04 µs / 0.141; spill_dist 18.23 µs / 0.138.

Umbrella: [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md).

---

## ST-SV9 — Final topk e2e

See [`ST_SV9_TOPK_E2E.md`](ST_SV9_TOPK_E2E.md). Arms `keep` / `remat_scores` / `remat_idx`.

---

## VMI / SimdVF twins (SV1V–SV9V)

Every Simt ST now has a VMI twin (`svNv_*` tags, `common_pto_harness`, `target=pto` + overlay).

| Twin | Mirrors | Status |
|------|---------|--------|
| SV1V–SV4V, SV9V | SV1–SV4, SV9 | **sources added 2026-09-24**; opsim deferred |
| SV5V–SV8V | Case-3 | sources prior; opsim deferred |

### VMI twin translation rules

**Contract (Lok, 2026-09-24):** a VMI twin follows the **original Simt ST** — same loops, same arms, same tag semantics. Do **not** invent VMI-only schedules (e.g. AABBCC / `token_tile` / `vf_fuse`) that the Simt case does not have.

For each case, document how Simt **fragment compute** maps to VMI using one of:

1. **ld + st + compute** — explicit loads/stores around compute (reload / remat / spill).
2. **fused with compute inline** — ld/st folded into compute (stream eltwise / fused reduce).
3. **stay-alive / forward live state in loop** — KEEP across `serial(K|B|…)`; topk-gate style predicated kill on live scores.

**Allowed ABI-only diffs:** `SimtVF(threads)`→`SimdVF(lanes)`, `alloc_fragment`→prefer `alloc_shared`, ascend/cython→pto/overlay, `T.max(v,-v)` not `T.abs`.

**Full per-case table + forbidden list:** [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md).  
Case-3 detail: [`ST_CASE3_VMI.md`](ST_CASE3_VMI.md).  
Control-flow next list (topk-gate; CF4/CF5 ≠ bolt-on to SV9V): [`ST_CF_TOPK_GATE_DESIGN.md`](ST_CF_TOPK_GATE_DESIGN.md); code slices + first-pass sensitivity: [`ST_CF1_CF6_SLICES.md`](ST_CF1_CF6_SLICES.md).

```bash
bash oneshot_sv1v_sv9v.sh
```

---

## Case-3 VMI twins (SV5V / SV6V / SV7V / SV8V)

Same math under SimdVF + overlay. Opsim pending. [`ST_CASE3_VMI.md`](ST_CASE3_VMI.md).

**Required compare:** every twin opsim must emit [`ST_VMI_COMPARE.md`](ST_VMI_COMPARE.md) (cycles / instr Δ / **EX highlight** / IPC) via `harvest_simt_vmi_compare.py`. Full twin contract: [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md).


---


---

## ST-SP1 … SP6 — Packed-structure access (MoE dual scatter/gather)

**Purpose:** TileKernels-style packed MoE / quant access missing from SV: co-move `(V,Sf)` under fused **Pos**, pad gather, sideband↔interleave, UE8M0/FP4 packs. See [`ST_SP1_TO_SP6_ACCESS.md`](ST_SP1_TO_SP6_ACCESS.md).

| ST | Content | Tags (primary) | Status |
|----|---------|----------------|--------|
| **SP1** | Dual scatter `(V,Sf)` via Pos | `sp1_t32_k2_h128_g32_t32_{keep_pos,remat_pos}` | GREEN Simt |
| **SP2** | Dual gather + wreduce | `sp2_…_sf1_w1_{keep_pos,remat_pos}` | GREEN Simt |
| **SP3** | FP32 SF vs packed UE8M0 | `sp3_m32_h128_g32_t32_{fp32,ue8m0}` | GREEN Simt (soft UE8M0) |
| **SP4** | Pad gather | `sp4_e64_h128_t32_pad25` | GREEN Simt |
| **SP5** | Sideband vs interleave | `sp5_n64_h128_g32_qg32_t32_{sideband,interleave}` | GREEN Simt |
| **SP6** | FP4 e2m1 unpack ± SF | `sp6_n32_h128_g32_t32_{unpack,unpack_sf}` | GREEN Simt |

**Oneshot:** `oneshot_sp1_sp6.sh`. VMI twins later (no bolt-on onto SV9V).

## Removed legacy (do not walk)

| archived file | reason |
|---------------|--------|
| sv2/sv3/sv4 topk micros | → SV9 arms |
| sv5_bcast_multiconsumer | → SV2 + Case-3 |
| sv6_group_scale_remat | → SV5/SV6/SV7/SV8 |
| sv7_block_reduce_128 | number reused by Case-3 large-G |
| sv8_block_reduce_32x32 | active SV8 = quant e2e |

---

## Run all Simt SV1–SV9

```bash
bash oneshot_sv1_sv9.sh
# from laptop:
powershell -File .\run_via_laptop_sv1_sv9.ps1
```

## Pointers

| Doc | Role |
|-----|------|
| This file | Canonical SV1–SV9 + GPU/NPU |
| [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md) | Quant Case-3 umbrella (SV5–SV8) |
| [`ST_SV9_TOPK_E2E.md`](ST_SV9_TOPK_E2E.md) | Topk e2e |
| [`RESULTS.md`](RESULTS.md) | Measured + archived tags |
| `kernels/_removed_legacy/` | Archived micros |

**CF oneshot:** `oneshot_cf1_cf6.sh` (Simt CF1–CF6 landed 2026-09-25).
