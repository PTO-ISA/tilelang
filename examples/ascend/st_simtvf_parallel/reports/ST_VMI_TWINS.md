# VMI / SimdVF twins — mapping contract (SV1V–SV9V)

**Date:** 2026-09-24 ~14:30 HKT  
**Suite:** `/workspace/st_simtvf_parallel`  
**Rule (Lok, 2026-09-24):** VMI/Simd twins MUST follow the **original Simt ST** and show the **expected translation mapping**. Do **not** invent VMI-only schedules the Simt case does not have.

## Twin fidelity contract

| Must stay identical to Simt | Allowed to differ (ABI only) | Forbidden on a twin |
|-----------------------------|------------------------------|---------------------|
| Loop nest (Parallel / serial / chunk order) | `T.SimtVF(threads=)` → `T.SimdVF(lanes=)` | Extra token tiling / `token_tile` unroll |
| Arms and tag semantics (`keep`, `reload`, `live`, …) | `alloc_fragment` → prefer `alloc_shared` for VF working set | AABBCC / ABCABC / `mte_mode` unless Simt ST has it |
| G / E / K / B / CM / R×C knobs and defaults | `target=ascend`+cython → `target=pto`+overlay | `vf_fuse` / multi-SimdVF schedule unless Simt ST has it |
| Remat vs KEEP meaning (what lives across which loop) | `alloc_var` for scalars; `T.max(v,-v)` not `T.abs` | Different remat arms or different kill home semantics |
| Reduce / emit / kill control-flow order | 1-elem reduce scratch may stay fragment if API requires | Bolting CF4/CF5 knobs onto SV9V “because VMI can” |

**Harness:** Simt = `common_asc_harness` (deps-native). VMI = `common_pto_harness` (overlay). Never pin both libs in one opsim session.

## Fragment → VMI mapping modes

For **each** case, document how Simt **fragment compute** translates to VMI:

| Mode | Name | Meaning | When it applies |
|------|------|---------|-----------------|
| **1** | **ld + st + compute** | Explicit loads into working set, compute, stores out (or publish to shared, then reload). | Remat / reload / spill_dist / stream temps that are not carried across an outer loop. |
| **2** | **fused with compute inline** | Ld/st folded into the compute expression (or producer writes a live buffer the consumer reads without a separate remat stage). | Simple eltwise / reduce→write where the working buffer is produced and consumed in the same VF without outer-loop remat. |
| **3** | **stay-alive / forward live state in loop** | Live scores/idxs/Acc/`sf_inv` carried across an outer `serial(K|B|…)` with in-place update (topk-gate style: predicated kill writes `NEG` into live scores; next K continues). | KEEP arms; Acc KEEP; idxs KEEP across B; live scales across consumer. |

**Production reference for mode 3** (pto-b10 topk-gate, *not* an SV9V schedule license):  
`score_vec`/`index_vec` stay in VL regs across `one_k`; after winner, `vsel(vcmp(eq), neg_inf, score)` kills the lane; next K uses the same live vectors. UB-stream remat + `mem_bar(VST_VLD)` is the remat_scores analogue. AABBCC/`vf_fuse`/`token_tile` are **separate** production knobs → CF4/CF5 only.

## Per-case mapping table

| Case | Simt structure (loops/arms) | VMI mapping mode (1/2/3) | What stays identical | What is allowed to differ (ABI only) | Forbidden on this twin |
|------|----------------------------|---------------------------|----------------------|--------------------------------------|------------------------|
| **SV1→SV1V** | One `SimtVF`; `Parallel(E)`: `t1←A`, `t2←B`, `t3=t1+t2`, `C←t3`. No outer carry. Tags `sv1_e{E}_t{T}`. | **1** (explicit temps ld/st around add); compute itself is fused add. | Single Parallel(E); E∈{256,2048}, T/lanes=64; no bcast/carry. | Fragment `t1/t2/t3` → shared; `SimdVF(lanes=)`. | Token tile, multi-VF, remat arms. |
| **SV2→SV2V** | Arms `frag_live` / `reload`. `Parallel(R)` absmax → `scale[R]`; consumer either uses live scale (`Parallel(R)+serial(C)`) or reloads from `scale_s` (`Parallel(R,C)`). | **`frag_live` → 2** (scale live, fused into consumer). **`reload` → 1** (publish + reload). | Both arms; R=C=32 default; abs via `T.max(v,-v)`; `alloc_var` absmax. | Fragment `x`/`scale` → shared. | Extra G ladder, Case-3 bcast, AABBCC. |
| **SV3→SV3V** | Arms `keep` / `split`. `keep`: `Acc[M,VL]` zero once; `serial(K)` reload A/x, Acc KEEP; flush. `split`: outer `serial(NC)`, Acc `[CM,VL]` KEEP across K, flush per chunk. | **3** (Acc stay-alive across K). Flush is mode-1 store. | keep/split_cm{CM}; M∈{24,32}, VL=64, K=16; A/x reload each K. | Acc fragment → shared. | Different CM/K semantics; token schedule. |
| **SV4→SV4V** | Arms `keep_idx` / `remat_idx`. `Out[i]=sum_b A[b,i]*W[Idx[i]]`; Idx non-identity tensor. keep: load Idx→idxs once, KEEP idxs+acc across B. remat: no live idxs; remat `idx_i` from shared Idx each gather; KEEP acc. | **`keep_idx` → 3** (idxs+acc stay-alive). **`remat_idx` → 1+3** (index remat + acc KEEP). | Both arms; E=256, B∈{8,16}; same loop nest; Idx from tensor. | idxs/acc fragment → shared under SimdVF. | topk CF; token_tile; AABBCC; vf_fuse. |
| **SV5→SV5V** | Small-G RF capacity: group inputs fit in ~32 VL-regs / one Simt thread. Arms `keep_in_rf` / `ub_reload` (scale keep/ub secondary). G=16. No bcast. | **keep_in_rf → 2/3**; **ub_reload → 1**. | Same G=16; Parallel(R,CG)+serial(G); no `j//G`. | Working set fragment → shared. | Mid/large G; SV8 bcast; AABBCC. |
| **SV6→SV6V** | Mid-G: SIMD must reload; Simt warp-level keep. G mid (hist G=32). | **VMI primary → 1** (reload). Simt `keep_in_warp` has no fake VL-keep twin. | Same mid G; same nest / arms semantics. | ABI fragment→shared. | Pretending whole group fits in 32 VL-regs. |
| **SV7→SV7V** | Large-G: multi-warp Simt + UB; SIMD reload/multi-pass. G∈{64,128}+. | **1** (reload / multi-pass). | Large-G ladder; multiwarp_ub semantics. | ABI as SV5V. | Bcast consumer (SV8); inventing keep that needs >32 VL. |
| **SV8→SV8V** | Arms `live` / `spill_dist`. Reduce→`sf_inv[R,CG]`; **live**: consumer `Parallel(R,CG)+serial(G)` reads live `sf_inv`. **spill_dist**: expand to `scale_ub[R,C]` then `Parallel(R,C)` reload. | **`live` → 3** (`sf_inv` stay-alive across reduce→consumer). **`spill_dist` → 1** (layout expand st + reload). | Both arms; G=16 Case-3 e2e; avoid `j//G` InverseAffine. | `sf_inv` fragment → shared inside SimdVF; `scale_ub` Kernel-scope shared (same as Simt). | AABBCC; inventing a third arm; different G without renaming ST. |
| **SV9→SV9V** | Arms `keep` / `remat_scores` / `remat_idx`. Single expert vector E; **`serial(K)` only** (no token loop). keep: scores+idxs KEEP, kill NEG in RF. remat_scores: reload scores each K, kill in shared `s`. remat_idx: no idxs buffer, remat `i`. | **`keep` → 3** (stay-alive scores+idxs + predicated kill). **`remat_scores` → 1** on scores + **3** on idxs. **`remat_idx` → 3** on scores; index rematerialized. | Same three arms; E=256, K∈{1,8}; `serial(K)` order; min-index-on-ties; kill after emit. | Working set fragment→shared; 1-elem `amax`/`best` may stay fragment; `SimdVF(lanes=)`. | **`token_tile` unroll, AABBCC/ABCABC, `vf_fuse`, multi-token SimdVF** — those are **CF4/CF5**, not SV9V. |

## Coverage / run status

| ST | Simt kernel | VMI twin | Opsim |
|----|-------------|----------|-------|
| SV1 | `sv1_stream_eltwise.py` | `sv1v_stream_eltwise.py` | deferred |
| SV2 | `sv2_eltwise_bcast_rf.py` | `sv2v_eltwise_bcast_rf.py` | deferred |
| SV3 | `sv3_gemv_partial_keep.py` | `sv3v_gemv_partial_keep.py` | **PASS** 2026-10-05 (m24 keep 4.57 µs, m32 keep 5.81 µs, m32 split_cm16 5.84 µs; see [`ST_SV3V_FIX.md`](ST_SV3V_FIX.md)) |
| SV4 | `sv4_index_gather_psum.py` | `sv4v_index_gather_psum.py` | deferred (keep2 replaced) |
| SV5 | `sv5_reduce_small_eltwise.py` | `sv5v_reduce_small_eltwise.py` | deferred |
| SV6 | `sv6_reduce_mid_eltwise.py` | `sv6v_reduce_mid_eltwise.py` | deferred |
| SV7 | `sv7_reduce_large_eltwise.py` | `sv7v_reduce_large_eltwise.py` | deferred |
| SV8 | `sv8_case3_bcast.py` | `sv8v_case3_bcast.py` | deferred |
| SV9 | `sv9_topk_e2e.py` | `sv9v_topk_e2e.py` | deferred (`run_opsim_topk.py`) |

```bash
bash oneshot_sv1v_sv9v.sh
# from laptop:
powershell -File .\run_via_laptop_sv1v_sv9v.ps1
```

Case-3-only: `oneshot_sv5v_sv6v_sv7v_sv8v.sh`. Detail: [`ST_CASE3_VMI.md`](ST_CASE3_VMI.md).


## Compare report (required)

Every twin opsim (SV1V–SV9V) must refresh [`ST_VMI_COMPARE.md`](ST_VMI_COMPARE.md): paired µs / cycles / instr# / IPC and **EX-pipe opcode highlight** (RVECEX / EX / `simt_*`). Emitter: `harvest_simt_vmi_compare.py`.

## Pointers

| Doc | Role |
|-----|------|
| [`ST_LIST_DESIGN.md`](ST_LIST_DESIGN.md) | Canonical SV1–SV9 + **VMI twin translation rules** |
| [`ST_CF_TOPK_GATE_DESIGN.md`](ST_CF_TOPK_GATE_DESIGN.md) | CF STs; CF4/CF5 ≠ free knobs on SV9V |
| [`ST_CASE3_VMI.md`](ST_CASE3_VMI.md) | Case-3 VMI detail (SV5V–SV8V) |
| [`ST_VMI_COMPARE.md`](ST_VMI_COMPARE.md) | **Required** Simt↔VMI twin compare (cycles / instr / EX highlight / IPC); every twin opsim must emit/refresh this |

## CF1V–CF6V (control-flow twins, 2026-09-28)

| Case | VMI mapping mode | Control-flow form | Opsim |
|------|------------------|-------------------|-------|
| CF1→CF1V | **3** KEEP Select | `T.Select(scores>thr, NEG, scores)` | PASS |
| CF2→CF2V | **1** remat + Select kill | same if → Select into shared | PASS |
| CF3→CF3V | **3** + remat `i` | `T.Select(i==v, NEG, scores)` | PASS |
| CF4→CF4V | **2** nested Select | nested `T.Select` | PASS |
| CF5→CF5V | **1** (Div outside VF) | near-0 Select; Div Kernel serial | PASS (ABI tax) |
| CF6→CF6V | **1** (Newton outside VF) | Newton Kernel serial; VF publish | PASS (ABI tax) |

Oneshot: `oneshot_cf1v_cf6v.sh`. Detail: [`ST_CF_IMPLEMENT.md`](ST_CF_IMPLEMENT.md).
