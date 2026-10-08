# ST-SV3 — GEMV / thin-GEMM loop-carried fp32 partial-sum RF stress

**Date:** 2026-09-23 ~15:30 HKT (opsim GREEN)  
**Suite:** `/workspace/st_simtvf_parallel`  
**Scope:** NEW ST-SV2 sensitivity story (GEMV partial-sum KEEP). Legacy topk CF remains in `kernels/sv2_topk_keep.py` (tags `sv2_*`); this ST uses tags `sv3_*`.

## Suite contract (applies to every ST)

- **Sensitivity target:** state what RF / Parallel / remat behavior this ST is probing (one sentence).
- **Turning-point knobs:** which parameters move the design across a performance cliff; list expected cliff.
- **Artifacts required:** TileLang source slice / TIR or Parallel+fragment view / thread-block RF map / µs+IPC (opsim Ascend950PR_9599 when available).

---

## Sensitivity target

Loop-carried **fp32 partial-sum tile** `Acc[M, VL]` stays in SimtVF fragment RF across outer K while A-tile and `x[k]` **reload** each K — probing the cliff where SIMD ~32 arch VL-regs are not enough for a large Acc, GPU MRF can still hold it, and NPU SimtVF folded RF should keep a relatively large partial-sum tile (or else pay a **split**/flush membar tax).

## Turning-point knobs

| knob | values | expected cliff |
|------|--------|----------------|
| M | **24**, **32** | Acc elems = M×VL; at VL=64, M=24→1536 / M=32→2048 fp32 live in frag before fold |
| VL | **64** (fixed) | strip width; matches “SIMD VL-reg” analogy |
| K | 16 (default; try 8/32) | outer loop length — KEEP cost grows with K if spilled; split flush count independent of K |
| threads T | **32** (default; optional 128) | elems/thread ≈ M×VL/T; T=32 M=32 VL=64 → **64** fp32/thread from Acc alone |
| arm | `keep` \| `split` | keep = full Acc in RF; split = Acc[CM,VL] + flush |
| CM | **16** (split) | chunk rows; num_chunks = ceil(M/CM); **membar ≈ O(chunks) not O(K)** |

**Primary matrix (oneshot):** M=24 keep · M=32 keep · M=32 split CM=16 — all VL=64 K=16 T=32.

## Source slices

Kernel: `kernels/sv3_gemv_partial_keep.py`  
CLI: `python kernels/sv3_gemv_partial_keep.py [M] [VL] [K] [threads] [arm] [CM]`  
Defaults `32 64 16 32 keep 16`.  
Tags: `sv3_m{M}_vl{VL}_k{K}_t{threads}_keep` or `…_split_cm{CM}`.

**Math:** `Acc[M, VL] += A[k, M, VL] * x[k]` (broadcast), fp16 A/x in shared, Acc/Out fp32.

### Arm `keep`

```python
@T.prim_func
def main(
    A: T.Tensor((K, M, VL), "float16"),
    X: T.Tensor((K,), "float16"),
    Out: T.Tensor((M, VL), "float32"),
):
    with T.Kernel(1):
        a_ub = T.alloc_shared((K, M, VL), "float16")
        x_ub = T.alloc_shared((K,), "float16")
        out_ub = T.alloc_shared((M, VL), "float32")
        T.copy(A, a_ub)
        T.copy(X, x_ub)
        with T.SimtVF(threads=threads):
            acc = T.alloc_fragment((M, VL), "float32")
            for i, j in T.Parallel(M, VL):
                acc[i, j] = T.float32(0.0)
            for k in T.serial(K):
                xk = T.alloc_var("float32")
                xk = T.Cast("float32", x_ub[k])          # x reload
                for i, j in T.Parallel(M, VL):
                    # A reload from shared; Acc KEEP
                    acc[i, j] = acc[i, j] + (
                        T.Cast("float32", a_ub[k, i, j]) * xk
                    )
            for i, j in T.Parallel(M, VL):
                out_ub[i, j] = acc[i, j]
        T.copy(out_ub, Out)
```

### Arm `split`

```python
# NC = ceil(M/CM); Acc tile is CM×VL (not M×VL)
acc_c = T.alloc_fragment((CM, VL), "float32")
for c in T.serial(NC):
    for i, j in T.Parallel(CM, VL):
        acc_c[i, j] = T.float32(0.0)
    for k in T.serial(K):
        xk = T.alloc_var("float32")
        xk = T.Cast("float32", x_ub[k])
        for i, j in T.Parallel(CM, VL):
            row = c * CM + i
            if row < M:
                acc_c[i, j] = acc_c[i, j] + (
                    T.Cast("float32", a_ub[k, row, j]) * xk
                )
    # One flush per chunk (natural MTE barrier) — NOT per K
    for i, j in T.Parallel(CM, VL):
        row = c * CM + i
        if row < M:
            out_ub[row, j] = acc_c[i, j]
```

Separate Python `@T.prim_func` bodies for keep vs split (SV6 lesson).

## TIR / Parallel view

1. Shared: `a_ub` fp16 `[K,M,VL]`, `x_ub` fp16 `[K]`, `out_ub` fp32 `[M,VL]`.
2. SimtVF:
   - **keep:** one `alloc_fragment((M, VL), fp32)` Acc; init Parallel; `serial(K)` with Parallel MAC; store Parallel.
   - **split:** one `alloc_fragment((CM, VL), fp32)` reused; outer `serial(NC)` × inner `serial(K)` MAC; flush Parallel per chunk.
3. Mutable accum uses `acc[i,j] = acc[i,j] + …` on fragment (SV6/SV2: OK; avoid immutable rebind outside region).
4. Out is **fp32** to avoid half `uint2{…}` pack issues (harness still has `_fix_uint2_half_pack` for other STs).

## Thread-block RF map (predicted)

Folded SimtVF layout: Acc elems/thread ≈ `M * VL / T` (keep) or `CM * VL / T` (split).  
A/x are shared reloads → **no** Acc-sized A fragment assumed live across K (footprint is temps + scalar `xk`).

| arm | M | VL | T | Acc elems (block) | elems/thread (Acc) | notes |
|-----|--:|---:|--:|------------------:|-------------------:|-------|
| keep | 24 | 64 | 32 | 1536 | **48** | below “64 fp32/thread” comfort line |
| keep | 32 | 64 | 32 | 2048 | **64** | cliff probe: SIMD ~32 VL-regs insufficient for this strip as arch regs; GPU MRF OK; NPU folded RF should hold |
| split CM=16 | 32 | 64 | 32 | 1024 live | **32** | half Acc RF vs keep; NC=2 flushes |
| keep (opt) | 32 | 64 | 128 | 2048 | **16** | more threads shrink per-thread Acc |

At **T=32 · M=32 · VL=64**: **64 fp32/thread from Acc alone** (+ A reload address/temps, scalar `xk`). That is the headline predicted RF number for fragment-RA discussion.

### Contrast: SIMD VL-reg vs GPU MRF vs NPU folded RF

| backend | Acc[32, 64] fp32 story |
|---------|------------------------|
| **SIMD** (~32 arch VL-regs) | Cannot hold a 32×64 partial-sum as VL-regs; must slice / spill / rematerialize — cliff. |
| **GPU MRF** | Large register file can keep Acc tile in RF for thin-GEMM MAC loop. |
| **NPU SimtVF folded RF** | Fragment folds to `acc[M*VL/T]` per thread; at 64 fp32/thread should still KEEP; if not, **split** is the controlled escape hatch. |

## Split / membar hypothesis

- **keep:** Acc never leaves RF during K → shared/MTE traffic for Acc is **one final store**; A/x loads each K.
- **split:** Acc chunk flushed to `out_ub` **once per chunk** after its K loop. Expected barrier/membar count **O(num_chunks) = O(ceil(M/CM))**, **not O(K)**.
- If compiler inserts a barrier every K iteration (bug / over-sync), wall time and MTE counters should expose it vs keep.

## Host gold

`run_opsim_generic.py` tag family `sv3_`: parse M, VL, K;  
`Acc += A[k].astype(fp32) * float(X[k])`; rtol/atol `2e-2` (fp16→fp32 accum).

## Status / opsim results

| item | state |
|------|--------|
| Kernel | `kernels/sv3_gemv_partial_keep.py` (`keep` + `split`) |
| Legacy topk | `kernels/sv2_topk_keep.py` **retained** |
| Opsim gold | `run_opsim_generic.py` `sv3_*` |
| Oneshot | `oneshot_sv2g.sh` + `run_via_laptop_sv2g.ps1` |
| IR dumps | `OUT/sources/{tag}_tir.txt` + `{tag}_source.txt` (+ optional `_lowered.txt`) |

### Measured matrix (Ascend950PR_9599) — GREEN 2026-09-23 ~15:30 HKT

Host: pto-b10 · lib `/tmp/libtilelang.so.deps_backup_ab` · harness uint1{half,half} pack fix.

| tag | PASS/FAIL | wall µs | IPC_proxy | Acc RF (CCE) | dumps |
|-----|-----------|--------:|----------:|--------------|-------|
| sv3_m24_vl64_k16_t32_keep | **PASS** | **6.79** | **0.324** | `float acc[48]` | `reports/sources/{tag}_{tir,source,lowered}.txt` |
| sv3_m32_vl64_k16_t32_keep | **PASS** | **8.63** | **0.340** | `float acc[64]` | `reports/sources/{tag}_{tir,source,lowered}.txt` |
| sv3_m32_vl64_k16_t32_split_cm16 | **PASS** | **7.79** | **0.355** | `float acc_c[32]` | `reports/sources/{tag}_{tir,source,lowered}.txt` |

**VF body:** m24 VF_us=5.06 / 9105 cyc; m32 keep VF_us=6.60 / 11874 cyc; split VF_us=5.77 / 10384 cyc.

**Membar / MTE notes:** CCE emits only **2** HardEvent pairs for all three arms (`MTE2_V` before VF + `V_MTE3` after) — **no per-K and no per-chunk SetFlag**. Chunk flush is in-VF `SIMT_STS` to UB. Instr-pipe `MTE2` call_count=4 for all tags. Split wall **faster** than M32 keep (7.79 vs 8.63 µs) with half Acc RF (`acc_c[32]` vs `acc[64]`).

**Compile note:** first attempt failed on ASC `uint1{half,half}` narrowing; fixed in `common_asc_harness._fix_uint1_half_pack` (same class as uint2 pack).

### What would flip the cliff next

- Confirm CCE fold: `acc[64]` @ M32 VL64 T32 keep; `acc[32]` @ split CM16.
- Count MTE/membar in split vs keep logs (hypothesis: bars ≈ NC, not K).
- Optional T=128 keep; K={8,32}; M=24 split CM=8 if needed.

## Report series note

This is the **SV3** deep-dive (GEMV partial-sum). Do not confuse with legacy SV2 topk KEEP numbers in `RESULTS.md` (`sv2_e256_*`).

## VMI twin (SV3V) — unblocked 2026-10-05 (REGRESSION-2)

`sv3v_gemv_partial_keep.py` compiled and ran on opsim for the three primary shapes
(`target=pto` + pr258 overlay, lanes=64, Ascend950PR_9599):

| tag | PASS/FAIL | wall µs | VMI IPC_proxy(VF) | Simt twin µs |
|-----|-----------|--------:|------------------:|-------------:|
| sv3v_m24_vl64_k16_t64_keep | **PASS** (`maxabs=0.0`) | **4.57** | 0.714 | 6.79 |
| sv3v_m32_vl64_k16_t64_keep | **PASS** (`maxabs=0.0`) | **5.81** | 0.712 | 8.63 |
| sv3v_m32_vl64_k16_t64_split_cm16 | **PASS** (`maxabs=0.0`) | **5.84** | 0.719 | 7.79 |

The KEEP/split sensitivity is preserved (Acc live across the outer K; one flush per chunk in
split). Four ABI blockers had to be removed — `alloc_var` → `local.var`, scalar f16→f32 cast,
non-lane-aligned `T.Parallel` extents (PTODSL has no `pto.vmi.mask_and`), and the `row < M`
guard. Details: [`ST_SV3V_FIX.md`](ST_SV3V_FIX.md); paired tables: [`ST_VMI_COMPARE.md`](ST_VMI_COMPARE.md).

Note the split arm is no longer the cheapest on the VMI side (5.84 vs 5.81 µs for m32 keep): with
UB-resident Acc the chunk loop adds a second zero/flush pass without saving RF pressure.
