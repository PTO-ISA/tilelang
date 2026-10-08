# ST-SV7 — Large-G reduce (multi-warp / UB vs SIMD reload)

**Date:** 2026-09-23 · **reframe 2026-10-06**  
**Kernel:** `kernels/sv7_reduce_large_eltwise.py`  
**Umbrella:** [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md)

---

## Sensitivity (primary)

**Large G≈128:** group needs **~4×32 threads** to keep one elem/thread in SIMT → **multi-warp** + **UB** (or cross-warp) partial reduce. **SIMD** also cannot hold the group → **reload / multi-slice**. Compare reduce wall when **both** backends leave the single-warp / 32-reg comfort zone.

Optional **G=64** (CG=2) as a step between mid and full 4-warp.

## Math / IO

Same fusion as SV5/SV6; **no bcast**.  
Primary: **R=64, C=128, G=128 → CG=1, T=32** (or T=128 for explicit 4-warp). Also `G=64 → CG=2`.

## Schedule / arms (planned)

| backend | arm | meaning | VMI mode |
|---------|-----|---------|----------|
| Simt | **`multiwarp_ub`** | Partials per warp → UB → final reduce | **1** on partial path |
| SimdVF | **`reload`** | Multi-pass VL load/reduce | **1** |
| both | baseline | Current serial(G) bodies | **2** |

## Baseline µs (Simt)

| tag | PASS | µs | note |
|-----|------|---:|------|
| sv7_r64_c128_g64_t32 | PASS | **14.44** | CG=2 |
| sv7_r64_c128_g128_t32 | PASS | **14.69** | CG=1 OK; +0.25 vs G64 |

Multi-warp UB arm: **TBD**.

## vs legacy

Old remat/reload-with-bcast moved to **SV8**. This ST stays reduced-grid only.

## VMI twin

`sv7v_*` — reload mode **1**; same large-G ladder. Prior COMPILE_FAIL same Case-3 ABI class as SV5V/SV6V.

---

## RESULT (2026-10-06, pto-b10) — `multiwarp_ub` vs `reload`

`kernels/sv7_reduce_large_eltwise.py` arms: `multiwarp_ub` (G chunked into NW=4
chunks of G/4; per-chunk partials written to `part_ub[R,CG,NW]` in UB, then a
final `Parallel` max over the NW partials → Y) and `reload` (single `serial(G)`
stream). **`threads=32`** for both — the chunk/warp loop is serial inside the
Parallel body, so the knob measured is the **UB partial traffic**, not the thread
count. (T=128 was not needed: T=32 compiled cleanly on the first try.)

| tag | PASS | wall µs | VF µs | IPC_meas | maxabs |
|-----|------|--------:|------:|---------:|--------|
| `sv7_r64_c128_g64_t32_multiwarp_ub` | PASS | **7.32** | 6.21 | 0.114 | 2.98e-08 |
| `sv7_r64_c128_g64_t32_reload` | PASS | 14.54 | 13.52 | 0.090 | 2.98e-08 |
| `sv7_r64_c128_g128_t32_multiwarp_ub` | PASS | 15.42 | 14.42 | 0.089 | 2.98e-08 |
| `sv7_r64_c128_g128_t32_reload` | PASS | **14.58** | 13.57 | 0.089 | 2.98e-08 |

**The winner flips with chunk width:**

- **G=64:** `multiwarp_ub` wins by **−7.22 µs (−49.7%)**. NW=4 makes the inner
  chunk **16 steps**, which is inside the unroll budget: its histogram has **no**
  `SIMT_BRANCH`/`ISETP`, while the 64-step single stream is a branch loop.
- **G=128:** chunk = 32 steps, the unroll is lost, and the UB partial round-trip
  is pure added cost: `reload` wins by **0.84 µs (+5.8% for `multiwarp_ub`)`.

**Answer to "does `multiwarp_ub` cost vs `reload` at G=128?" → yes, ≈ +0.84 µs
(+5.8%)**, i.e. the UB allreduce of 4 partials is a small but real tax once the
chunks are no longer unrollable. The useful structural result is that *chunking
to ≤16 steps* is worth ~2× — far more than where the inputs live.

Theory (`reports/ST_VF_THEORY.md`): G=128 `multiwarp_ub` C_pred 21730 / C_meas
25948 (1.19), `reload` 20840 / 24423 (1.17); G=64 `multiwarp_ub` 6370 / 11182
(1.76 — UB partial traffic priced only as flat `O`), `reload` 20900 / 24339 (1.16).
