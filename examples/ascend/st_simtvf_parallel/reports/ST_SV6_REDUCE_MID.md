# ST-SV6 — Mid-G reduce (SIMD reload vs SIMT one-warp keep)

**Date:** 2026-09-23 · **reframe 2026-10-06**  
**Kernel:** `kernels/sv6_reduce_mid_eltwise.py`  
**Umbrella:** [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md)  
**Tag baseline:** `sv6_r64_c128_g32_t32`

---

## Sensitivity (primary)

**Mid G=32:** group no longer fits in **SIMD ~32 arch VL-regs** → SIMD must **slice / reload**. **SIMT** can still hold the group in **one warp** (≈1 elem/thread @ T=32) and finish with a **warp-level reduce** — **no UB write for a multi-warp allreduce**.

Purpose: compare **SIMD reload tax** vs **SIMT in-warp keep** on the same math.

## Math / IO

Same as SV5: `Y = sf_inv` on `[R,CG]`, **no bcast**.  
**R=64, C=128, G=32 → CG=4, T=32**.

## Schedule / arms (planned)

| backend | arm | meaning | VMI mode |
|---------|-----|---------|----------|
| Simt | **`keep_in_warp`** | Distribute G across one warp; in-warp reduce; no UB allreduce of partials | — |
| SimdVF | **`reload`** / slice | VL-reg budget overflow → UB round-trip or multi-pass | **1** |
| both | baseline single body | Current `Parallel(R,CG)+serial(G)` until arms land | **2** |

## Baseline µs

| tag | PASS | µs |
|-----|------|---:|
| sv6_r64_c128_g32_t32 | PASS | **12.25** |

Warp_keep / SIMD reload arms: **TBD**.

## VMI twin

`sv6v_*` — mode **1** on reload arm; fidelity = same G=32 nest. Prior COMPILE_FAIL: CG vs lanes alignment — fix with lane-legal Parallel extents (SV3V lesson).

---

## RESULT (2026-10-06, pto-b10) — `keep_in_warp` vs `reload`

`kernels/sv6_reduce_mid_eltwise.py` arms: `keep_in_warp` (group preloaded into a
fragment, warp-local reduce, **no warp-partial scratch in UB**) and `reload`
(baseline `Parallel(R,CG)+serial(G)` stream from `x_ub`).

| tag | PASS | wall µs | VF µs | IPC_meas | maxabs |
|-----|------|--------:|------:|---------:|--------|
| `sv6_r64_c128_g32_t32_keep_in_warp` | PASS | 14.83 | 13.72 | 0.092 | 5.96e-08 |
| `sv6_r64_c128_g32_t32_reload` | PASS | **12.58** | 11.53 | 0.107 | 5.96e-08 |

**Winner: `reload`, −2.25 µs (−15.2%).** The in-warp keep is genuinely warp-local
(nothing is written to UB except the final `sf_inv`), but at G=32:

- the preload spills to the stack (`SIMT_LDK:256` / `SIMT_STK:256`);
- G=32 is past the ≈16-step unroll budget, so **both** arms run branch loops
  (`SIMT_BRANCH:256`, `SIMT_ISETP_I:256`) at IPC ≈0.09–0.11, and the keep arm
  simply pays that branchy cost **twice** (preload pass + reduce pass).

So "SIMT can keep a G=32 group in one warp" is true structurally but is not a
win on this uarch/compiler: the reload tax (≈1 `SIMT_LDS` per element) is
cheaper than an extra RF pass plus spill. Baseline no-arm row
(`sv6_r64_c128_g32_t32`, 12.25 µs) stays historical.

Theory (`reports/ST_VF_THEORY.md`): `reload` C_pred 21020 vs C_meas 20748
(ratio 0.99); `keep_in_warp` C_pred 34070 vs C_meas 24697 (0.72 — the model
over-charges the second branchy pass).
