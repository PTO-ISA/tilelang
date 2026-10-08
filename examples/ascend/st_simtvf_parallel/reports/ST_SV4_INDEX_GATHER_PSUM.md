# ST-SV4 — Index gather + partial-sum across batch B (keep_idx / remat_idx)

**Date:** 2026-09-24 ~17:15 HKT (replace keep2) · **updated 2026-10-05 ~13:40 HKT (near-spill ladder + first opsim numbers)**
**Suite:** `/workspace/st_simtvf_parallel`
**Scope:** Active ST-SV4 probes **non-identity VCI gather** while a **fp32 partial sum across batch B** occupies RF. No topk CF (SV9).
**Redirect:** Former report `ST_SV4_INDEX_KEEP2.md` replaced by this file (keep2 = identity idxs, Out[B,E]).

## Suite contract (applies to every ST)

- **Sensitivity target:** state what RF / Parallel / remat behavior this ST is probing (one sentence).
- **Turning-point knobs:** which parameters move the design across a performance cliff; list expected cliff.
- **Artifacts required:** TileLang source slice / TIR or Parallel+fragment view / thread-block RF map / µs+IPC (opsim Ascend950PR_9599 when available).

---

## Sensitivity target

Can VMI/Simt **remat a long VCI index** (`Idx[E]` loaded from tensor, not `idxs[i]=i`) while a **large register keep-alive** (`acc[E]` partial sum across batch) already occupies RF? Contrast against **KEEP both** idxs + acc across `serial(B)`.

**Gold:** `Out[i] = sum_{b=0..B-1} A[b,i] * W[Idx[i]]`
**Idx construction (seed 4 IO; fixed table):** `Idx[i] = (i*7 + 3) % E` — non-identity so gather is real.

## Arms

| arm | live working set across `serial(B)` | status on SimtVF (ascend) |
|-----|--------------------------------------|---------------------------|
| `keep_idx` | `idxs[E/T]` int32 **+** `acc[E/T]` fp32 | **green** |
| `remat_idx` | `acc[E/T]` fp32 only; index re-read from shared `idx_ub[i]` inside the gather | **COMPILE_FAIL** (stack limitation, see “Blockers”) |
| `remat_idx_calc` *(new 2026-10-05)* | `acc[E/T]` fp32 only; index **recomputed** as `(i*7+3) % E` at each gather | **green** — this is the measurable remat arm |

`remat_idx_calc` has exactly the same gold and the same MTE traffic (Idx is still copied to UB); it
differs only in that no index register stays live, so it isolates the RF effect the ST is after.

## Turning-point knobs

| knob | values | cliff |
|------|--------|-------|
| E | **256**, 512, **1024**, 1536, **2048** (T=32) | elems/thread `N = E/T`; keep_idx holds `2N` per-thread scalars, remat holds `N` |
| B | **8**, **16** | outer reuse — keep cost per-b grows once spilled; remat tax ∝ B |
| threads T / lanes L | Simt **32**, **16** / VMI **64** | T=16 doubles `N` at the same E (fast way to cross the cliff) |
| arm | `keep_idx` \| `remat_idx(_calc)` | keep = mode **3** (idxs+acc stay-alive); remat = mode **1** on index + **3** on acc |

**Primary matrix (unchanged):** `sv4_e256_b{8,16}_t32_{keep_idx,remat_idx_calc}`
**Near-spill ladder (new):** `sv4_e{512,1024,1536,2048}_b8_t32_*`, `sv4_e1024_b16_t32_*`, `sv4_e1024_b8_t16_*`

## RF math (Simt SimtVF, T threads)

`N = E / T` elements per thread. The CCE that SimtVF emits declares the fragments as **per-thread
local arrays** (verified in the dumped sources):

```
keep_idx : int32_t idxs[N];  float acc[N];   -> 2N live per-thread scalars (8N bytes)
remat_*  :                   float acc[N];   ->  N live per-thread scalars (4N bytes)
```

Measured allocator behaviour on Ascend950PR_9599 (opsim instruction mix, `SIMT_STK` = stack store,
`SIMT_LDK` = stack/const reload):

| N = E/T | example | `SIMT_LDK` | `SIMT_STK` | verdict |
|--------:|---------|-----------:|-----------:|---------|
| 8 | e256 t32 | not in top-8 opcodes | 0 | comfortable |
| 16 | e512 t32 | not in top-8 opcodes | 0 | comfortable |
| **32** | **e1024 t32** | **161** (B16: 321) | 0 | **near-spill** (keep = 64 live scalars) |
| 48 | e1536 t32 | 481 | 0 | spilling (reload-only) |
| **64** | **e2048 t32 / e1024 t16** | **1313** | **417** | **hard spill, both arms** |

So the usable per-thread budget is **≈64 live scalars**: `keep_idx` reaches it at `N=32` (E=1024,
T=32) while `remat_*` is still at half that — exactly the window where remat wins. Past `N≈48` the
accumulator **alone** busts RF and both arms spill, which closes the window again.

PTO-DSL twin (SV4d, VL=64): `NCH = E/64` vector registers per array, `keep = 2·NCH`, `remat = NCH`.
bisheng refuses `E=2048` keep (`Vector Slots: 70`, `Total Spilled Byte Size: 17952`, clang exit 70),
pinning the vpto vector-slot budget at ~64 — same shape of cliff, hard failure instead of slow code.

## Measured — opsim Ascend950PR_9599 (2026-10-05, deps-native lib, gold rtol/atol 1e-5, seed 4)

Simt (`kernels/sv4_index_gather_psum.py`, `core0.veccore0 duration_time`):

| tag (E / B / T) | N | keep live | keep_idx µs | remat_idx_calc µs | Δ remat vs keep | keep IPC | remat IPC |
|-----------------|--:|----------:|------------:|------------------:|----------------:|---------:|----------:|
| e256 b8 t32 | 8 | 16 | 1.67 | **1.62** | **−3.0 %** | 0.367 | 0.417 |
| e512 b8 t32 | 16 | 32 | 2.83 | **2.71** | **−4.2 %** | 0.290 | 0.288 |
| **e1024 b8 t32** | 32 | 64 | 7.16 | **6.30** | **−12.0 %** | 0.174 | 0.192 |
| **e1024 b16 t32** | 32 | 64 | 11.83 | **10.53** | **−11.0 %** | 0.172 | 0.190 |
| e256 b16 t32 | 8 | 16 | 2.39 | **2.22** | **−7.1 %** | 0.343 | 0.369 |
| e1536 b8 t32 | 48 | 96 | **12.09** | 12.79 | +5.8 % | 0.130 | 0.132 |
| e2048 b8 t32 | 64 | 128 | 26.08 | 26.01 | −0.3 % | 0.114 | 0.107 |
| e1024 b8 t16 | 64 | 128 | **84.44** | 86.69 | +2.7 % | 0.033 | 0.030 |

All rows PASS (`maxabs ≤ 1.91e-06` vs numpy gold). `sv4_*_remat_idx` (remat from shared) rows are COMPILE_FAIL.

### Where remat pulls ahead

* **Crossover = `N≈32` (E=1024 at T=32).** keep_idx sits right at the ≈64-scalar budget (32 int32 +
  32 fp32), remat_idx_calc at 32 — remat wins **12 % at B=8 and 11 % at B=16**, with a visibly
  better IPC proxy (0.192 vs 0.174) and ~5 % fewer body instructions (1659 vs 1754) and 1437 fewer VF cycles (8647 vs 10084).
* Below the boundary the win is only the saved index loads (3–7 %): real but small.
* Raising **B** at the crossover shape keeps the win (−11 % at B=16), i.e. the remat tax ∝ B does not
  eat the register saving while keep is near-spill.
* **Past the boundary the window closes.** At `N=48` (e1536) and `N=64` (e2048 t32, e1024 t16) the
  accumulator alone spills (`SIMT_STK` 417, `SIMT_LDK` 1313), so keep's extra index registers barely
  move the spill (keep STK 417 vs remat 425, identical LDK) and remat's arithmetic index tax makes it
  **equal or slightly worse** (+5.8 % at e1536, +2.7 % at e1024 t16).
* e1536 also shows a second-order knob: `E` not a power of two turns `(i*7+3) % E` into a real
  `IMAD/IADD` sequence (`SIMT_IMAD_I` 176 vs 128) instead of an `& (E-1)` mask — keep the ladder on
  powers of two when comparing arms.

## Source slices

| twin | kernel | harness |
|------|--------|---------|
| Simt | `kernels/sv4_index_gather_psum.py` | `target=ascend` + cython (`common_asc_harness`) |
| VMI | `kernels/sv4v_index_gather_psum.py` | `target=pto` (`common_pto_harness`); idxs/acc `alloc_shared` under SimdVF |
| PTO-DSL | `kernels_ptodsl/sv4d_index_gather_psum.py` | explicit `vpto` (`ptodsl.pto.jit`), VL=64 |

CLI: `python kernels/sv4_index_gather_psum.py [E] [B] [threads] [arm]`
Defaults `256 8 32 keep_idx`; `arm ∈ {keep_idx, remat_idx, remat_idx_calc}`. Tag: `sv4_e{E}_b{B}_t{threads}_{arm}`.

IO: `A[B,E] fp32`, `W[E] fp32`, `Idx[E] int32`, `Out[E] fp32`.

### Arm `keep_idx` (mode 3)

```python
with T.SimtVF(threads=threads):
    idxs = T.alloc_fragment((E,), "int32")
    acc = T.alloc_fragment((E,), "float32")
    for i in T.Parallel(E):
        idxs[i] = idx_ub[i]          # load Idx once
    for i in T.Parallel(E):
        acc[i] = T.float32(0.0)
    for b in T.serial(B):
        for i in T.Parallel(E):
            acc[i] = acc[i] + a_ub[b, i] * w_ub[idxs[i]]  # KEEP idxs + KEEP acc
```

### Arm `remat_idx_calc` (mode 1 on index, mode 3 on acc) — green remat

```python
with T.SimtVF(threads=threads):
    acc = T.alloc_fragment((E,), "float32")   # NO live idxs
    for i in T.Parallel(E):
        acc[i] = T.float32(0.0)
    for b in T.serial(B):
        for i in T.Parallel(E):
            acc[i] = acc[i] + a_ub[b, i] * w_ub[(i * 7 + 3) % E]   # VCI recomputed
```

Emitted CCE (e256 b8 t32, verbatim lines from `sources/*_source.txt`) — only the accumulator survives:

```c
// sv4_e256_b8_t32_remat_idx_calc
__simt_vf__ __launch_bounds__(32) inline void simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float acc[8];
      acc[i_1] = (acc[i_1] + (((__ubuf__ float*)buf_dyn_shmem)[((((b * 256) + ((i_1 >> 1) * 64)) + (((int32_t)threadIdx.x) * 2)) + (i_1 & 1))] * ((__ubuf__ float*)buf_dyn_shmem)[(((((((i_1 >> 1) * 448) + (((int32_t)threadIdx.x) * 14)) + ((i_1 & 1) * 7)) + 3) & 255) + 2048)]));

// sv4_e256_b8_t32_keep_idx
  int32_t idxs[8];
  float acc[8];
      acc[i_2] = (acc[i_2] + (((__ubuf__ float*)buf_dyn_shmem)[...] * ((__ubuf__ float*)buf_dyn_shmem)[(idxs[((int64_t)i_2)] + 2048)]));
```

## Blockers (keep as suite evidence — do not silently “fix” by changing the ST)

1. **`remat_idx` (re-read `Idx` from shared inside `T.Parallel`) does not compile on SimtVF/ascend**
   with the deps-native libtilelang (2026-10-05). The per-lane gather address becomes a vector value:
   * stock: `TileLangThreadSync → arith::EvalSet` `ICHECK(eval_vec_)` failure;
   * `tl.disable_thread_storage_sync=1`: codegen `Cannot convert type int64x2 (lanes=2) to Ascend type`;
   * `tl.disable_vectorize_256=1` does not defuse it; `tl.config_index_bitwidth=32` turns it into
     `variables (idx_i,) are used, but are not passed in as API arguments`.
   `keep_idx` is unaffected because its index comes from a per-thread fragment. This is why
   `remat_idx_calc` exists — it keeps the RF question answerable while the shared-remat path is blocked.
2. **SV4v (SimdVF / VMI twin) — COMPILE_FAIL, shape independent.** With the PR258 overlay lib, both
   arms fail in PTO lowering:
   `'func.func' op contains an uncovered top-level op segment whose section kind cannot be inferred uniquely; ambiguous op(s): 'pto.set_flag'; wrap the ambiguous region in pto.section.cube or pto.section.vector`
   → `Error: failed to normalize uncovered PTO tile sections`. Same at e256/e1024/e2048, b8, lanes=64.
   **No µs invented for sv4v.**
3. **SV4d arms collapse to the same code.** The emitted MLIR does differ (remat e256: 64 `vload` vs
   keep 36), but the vpto/bisheng backend hoists the loop-invariant index reloads, so every measured
   shape gives *identical* µs for both arms, and both die the same way at E=2048.

## PTO-DSL twin (SV4d) — opsim Ascend950PR_9599

`E`/`B` are real knobs now (they used to be hard-wired 256/8); `NCH = E/64` vector registers per array.

| tag | keep_idx µs | remat_idx µs | note |
|-----|------------:|-------------:|------|
| sv4d_e256_b8_t32 | 1.02 | 1.02 | NCH=4 |
| sv4d_e512_b8_t32 | 1.25 | 1.25 | NCH=8 |
| sv4d_e1024_b8_t32 | 2.32 | 2.32 | NCH=16 (keep = 32 VRs) |
| sv4d_e2048_b8_t32 | **FAIL** | **FAIL** | NCH=32 — bisheng `Vector Slots: 70`, `Total Spilled Byte Size: 17952`, exit 70 |
| sv4d_e256_b16_t32 | 1.14 | 1.14 | B knob |

## Thread-block RF map (Simt, measured from the CCE dumps)

| tag | E | B | T | idxs elems/thread | acc elems/thread | spill markers |
|-----|--:|--:|--:|------------------:|-----------------:|---------------|
| sv4_e256_b8_t32_keep_idx | 256 | 8 | 32 | 8 int32 | 8 fp32 | none |
| sv4_e256_b8_t32_remat_idx_calc | 256 | 8 | 32 | 0 (remat) | 8 fp32 | none |
| sv4_e1024_b8_t32_keep_idx | 1024 | 8 | 32 | **32** int32 | 32 fp32 | LDK 161 (near-spill) |
| sv4_e1024_b8_t32_remat_idx_calc | 1024 | 8 | 32 | 0 | 32 fp32 | LDK 160 |
| sv4_e2048_b8_t32_keep_idx | 2048 | 8 | 32 | **64** int32 | 64 fp32 | LDK 1313 / STK 417 |
| sv4_e2048_b8_t32_remat_idx_calc | 2048 | 8 | 32 | 0 | 64 fp32 | LDK 1315 / STK 425 |

## Repro

```bash
# whole ladder, both arms (writes /tmp/st_simtvf_parallel/SUMMARY_sv4_raw.txt)
bash examples/ascend/st_simtvf_parallel/oneshot_sv4.sh
```

Isolated run used for the 2026-10-05 numbers (private deps copy, shared lib untouched):
`/tmp/pr272_sv4_cliff_20261005/{envrc.sh,cc.sh,sim.sh,lane.sh,laned.sh}`, results in
`/tmp/pr272_sv4_cliff_20261005/out/{SUMMARY_sv4_ladder.txt,SUMMARY_sv4d_ladder.txt,HARVEST_sv4.txt}`
and `RESULT.md` in that directory.

## Artifact paths

| kind | path pattern |
|------|----------------|
| SO | `<OUT>/so/sv4*_e*_b*_t*_{keep_idx,remat_idx,remat_idx_calc}.so` |
| TIR / source / lowered | `<OUT>/sources/*sv4*_{keep_idx,remat_idx,remat_idx_calc}*` |
| compile / opsim logs | `<OUT>/logs/compile_sv4*.log`, `<OUT>/opsim_sv4*.log` |
| oneshot summary | `<OUT>/SUMMARY_sv4_raw.txt` |

## Contrast vs replaced keep2 / SV9

| | active gather_psum | replaced keep2 | SV9 remat_idx |
|--|--------------------|----------------|---------------|
| file | `sv4_index_gather_psum.py` | `sv4_index_keep2.py` (removed) | `sv9_topk_e2e.py` |
| tag | `sv4_e{E}_b{B}_t{T}_{keep_idx,remat_idx,remat_idx_calc}` | `sv4_e*_b*_t*_keep2` | `sv9_e*_k*_t*_remat_idx` |
| Idx | tensor `(i*7+3)%E` | identity `i` | remat loop `i` in topk CF |
| Out | `[E]` partial sum across B | `[B,E]` per-batch | IdxOut `[K]` |
| CF | none | none | reduce_max / kill |
