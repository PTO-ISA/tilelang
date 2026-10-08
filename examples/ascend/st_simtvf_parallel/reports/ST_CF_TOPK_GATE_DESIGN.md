# Control-flow ST design — first cut from topk-gate

**Date:** 2026-09-24 ~14:30 HKT (fidelity note added)  
**Suite:** `/workspace/st_simtvf_parallel`  
**Basis:** production topk-gate under `~/projects/tilelang-deepseek` + fluxdata copies  
(`example_simdvf_topk_gate.py`, `topk_gate_vl128.py`, TileKernels-vmi `topk_gate_vmi.py`, msprof AABBCC/ABCABC artifacts).  
**Status:** design only — CF kernels **not** implemented this pass (stub names proposed).

**Code slices + first-pass sensitivity:** [`ST_CF1_CF6_SLICES.md`](ST_CF1_CF6_SLICES.md).

**2026-09-25 cut (Lok):** User-facing CF = **CF1/CF2 + CF3 remat_idx only**, plus **CF4 nested-if / CF5 div+ULP / CF6 Newton+range** for Simt-vs-Simd divergence. Schedule AABBCC/vf_fuse deferred. See [`ST_CF1_CF6_SLICES.md`](ST_CF1_CF6_SLICES.md).



## Fidelity note — CF4 / CF5 are NOT free knobs on SV9V

**Rule (Lok, 2026-09-24):** VMI twins must follow the **Simt** ST. SV9’s loop is a
**single expert vector** with `serial(K)` only — **no token unroll**. Therefore:

| Claim | Verdict |
|-------|---------|
| “SV9V can do AABBCC / ABCABC because production topk-gate / VMI can” | **Forbidden.** That invents a schedule Simt SV9 does not have. |
| “SV9V can set `vf_fuse` / `token_tile=2` to match msprof” | **Forbidden** on the SV9V twin. |
| Where do AABBCC / `vf_fuse` live? | **CF4** (AABBCC vs ABCABC) and **CF5** (`vf_fuse`) — **separate CF STs**. |
| What must CF4/CF5 declare first? | A **Simt (or shared) ST** that actually has those knobs (token_tile, mte_mode, vf_fuse), then a VMI twin that mirrors **that** ST — not a bolt-on to SV9V. |

SV9 / SV9V remain the **micro e2e** for keep / remat_scores / remat_idx (mapping mode 3 stay-alive kill).  
CF6 may later combine SV9 arms **with** CF4/CF5 schedules only after CF4/CF5 have their own Simt baseline.

Canonical twin contract: [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md).

---
## Methodology contract (from ST_LIST_DESIGN)

Every CF ST must declare:
1. **Sensitivity target** — which CF / remat / schedule behavior is probed.
2. **Simt vs VMI expectation** — where folded RF / overlay path diverge.
3. **Turning-point knobs** — E, K, lanes/threads, schedule mode, vf_fuse, remat arm.
4. **Artifacts** — µs, IPC, membar / SMEM_BAR count, spill / RF map, source+TIR.

## Harvested topk-gate CF patterns

| Pattern | Where seen | Why it matters for ST |
|---------|------------|------------------------|
| **Predicated kill** | `one_k`: after winner, `vsel`/`vcmp(eq)` → write `neg_inf` into score lane (mask-out) | Hot predicated rewrite of live scores across K |
| **Scores KEEP vs remat** | In-reg chunks (`MAX_CHUNKS_INREG`) vs UB-stream reload each pass / each K | Same axis as SV9 `keep` / `remat_scores` |
| **Index KEEP vs remat (VCI)** | Comment: “do not re-VCI” when reusing `index_vec` across token1; remat via `vci(offset)` on UB-stream | Same axis as SV9 `remat_idx` / SV4 gather_psum |
| **Tie-break min-index** | `match-eq → vmin` for smallest index on equal max | Stable top-K; stresses predicated idx path |
| **Membar / phase grouping** | `mte_mode=aabbcc` (all MTE2 → VF → all MTE3) vs `abcabc` (per-token MTE2→VF→MTE3) vs `single_shot` (one 2-row copy) | Turns membar / SMEM_BAR and pipe overlap |
| **vf_fuse** | One `SimdVF` wrapping token_tile unroll vs per-token SimdVF (`vf_blocks=2`) | VF push count / schedule cliff |
| **VL64 vs VL128/g=2** | `num_groups=2` packs 2 tokens / VF (≠ AABBCC) | ILP packing; optional later CF |
| **Serial K vs unroll K** | `use_serial_k` vs `T.unroll(num_topk)` | Loop-carried kill pressure |

SV9 already covers keep / remat_scores / remat_idx at **micro** e2e (no token unroll). CF list below adds **schedule + membar + vf_fuse** knobs that production topk-gate flips — as **new CF STs with their own Simt baselines**, not as SV9V extensions.

## Proposed first CF ST list (CF1…)

### Rule: early CF = one Parallel control; complex at the end

See [`ST_CF1_CF6_SLICES.md`](ST_CF1_CF6_SLICES.md). Do **not** put tie-break `if` + kill `if` in the same loop for CF1–CF3.

### CF1 — Predicated kill-eq-max (scores KEEP)

- **Sensitivity:** Sole Parallel CF: after `reduce_max`, `if scores==amax: scores=NEG`. No tie-break, no IdxOut required.
- **Knobs:** E∈{128,256}, K∈{1,8}, T/lanes=32.
- **Simt vs VMI:** Stay-alive kill; expect membar≈0.
- **Stub:** `kernels/cf1_pred_kill_keep.py` tags `cf1_e{E}_k{K}_t{T}_keep`.

### CF2 — Remat scores + kill-eq-max in shared

- **Sensitivity:** Same one kill `if` as CF1; remat from / kill into shared → MTE/membar vs CF1.
- **Knobs:** E=256, K∈{1,8,16}, T=32.
- **Stub:** `kernels/cf2_remat_scores_kill_shared.py` tags `cf2_e{E}_k{K}_t{T}_remat_scores`.

### CF3 — Kill-by-index (KEEP idxs vs remat `i`)

- **Sensitivity:** `best` from `reduce_argmax` (no Parallel tie-break); sole `if` kills by index. Arms: KEEP idxs vs remat `i`.
- **Knobs:** E=256, K=8, T∈{32,128}, arm∈{keep_idx,remat_idx}.
- **Stub:** `kernels/cf3_kill_by_idx.py` tags `cf3_e{E}_k{K}_t{T}_{keep_idx,remat_idx}`.

### CF4 — Schedule AABBCC vs ABCABC (membar)

- **Sensitivity:** Phase grouping only; **body = CF1** kill-eq-max (still one `if`).
- **Knobs:** `mte_mode∈{aabbcc,abcabc}`, token_tile=2, E=256, K=8.
- **Prerequisite:** Own Simt `token_tile` ST — **not** a bolt on SV9V.
- **Stub:** `kernels/cf4_sched_aabbcc_abcabc.py`.

### CF5 — vf_fuse (one SimdVF vs per-token)

- **Sensitivity:** VF push / fuse; body = CF1 kill-eq-max.
- **Knobs:** `vf_fuse∈{0,1}`, token_tile=2.
- **Prerequisite:** Same as CF4.
- **Stub:** `kernels/cf5_vf_fuse.py`.

### CF6 — Complex: linked tie-break + kill (full top-K)

- **Sensitivity:** **Two** Parallel controls (tie-break + kill) — production `one_k`. First complex CF.
- **Knobs:** arm∈{keep,remat_scores,remat_idx}, E=256, K=8.
- **Optional later CF7:** CF6 × CF4/CF5 schedule umbrella.
- **Stub:** `kernels/cf6_topk_linked_cf.py`.

## Suggested first implementation order

1. **CF1 → CF2 → CF3** — one-control micros + VMI twins.  
2. **CF4 → CF5** — schedule / fuse with CF1 body.  
3. **CF6** — linked tie-break+kill.  
4. **CF7 (optional)** — CF6 × schedule matrix.

## Out of scope (this first cut)

- VL128 / `num_groups=2` packing (note only; optional CF7 later).  
- Full MoE gate + expert dispatch beyond top-K indices.  
- Running opsim for CF (design only).

## Sources (pto-b10 harvest)

- `/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/examples/ascend/example_simdvf_topk_gate.py`
- `…/topk_gate_vl128.py` + msprof `*_aabbcc_*` / `*_abcabc_*` artifacts
- `~/projects/TileKernels-vmi/tile_kernels_vmi/moe/topk_gate_vmi.py`
