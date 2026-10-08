# ST — VF cycle THEORY model (predicted vs camodel measured)

**Date:** 2026-10-06 HKT · **Scope:** Case-3 reduce-group RF-capacity ladder SV5/SV6/SV7
(+ VMI twins sv5v/sv6v/sv7v) · **Host:** pto-b10 · **Sim:** `Ascend950PR_9599`
**Code:** `harvest_vf_theory.py` (module + CLI), wired into
`harvest_sv567_rfladder.py` so every `RESULT.md` carries the theory columns.
No push; local suite only.

Purpose: for **each VF case/arm** have an a-priori cycle estimate next to the
measured VF cycles, so (a) we can tell whether a measurement makes sense at all,
and (b) the same `W` can later be replayed against a **a5 vs a6** uarch with only
`I_assum`, `O_*` and `freq` changed.

## Formula

```
W        = useful work ops in the VF body (whole kernel, all lanes)
W_thr    = W / T                      # T = tag `t` field: SIMT threads (or SimdVF lanes)
C_pred   = O_arm + ceil(W_thr / I_assum)
us_pred  = C_pred / freq
C_meas   = VF_SIMT (or VF_SIMD) `cycles` from core0.veccore0_instr_exe.csv
ratio    = C_meas / C_pred            # OUTLIER_SLOW if >2.0, OUTLIER_FAST if <0.5
IPC_meas = body_instr / C_meas        # body_instr = Σ call_count except VF_* and
                                      #   set_flag/wait_flag/end/nop/push_pb/dcci
```

`W_thr` exists because the camodel instruction table counts **per thread / per
lane-group**, not per element: the SIMT body issues `R*C/T` element-iterations
per thread. (Cross-check: SV5 G=16 `ub_stream` predicts `W/T = 1072` body
instructions, measured `body_instr` lands in the same bracket.)

## W — useful work ops (E = R·C elements, N = R·CG groups, NW = 4 chunks)

| term | ops | note |
|------|-----|------|
| `W_arith` | **3E** | f16→f32 cast (1) + abs as `max(v,-v)` (1) + running `max` (1) per element |
| `W_mem` (`ub_stream`, `reload`) | **1E** | one UB load per element |
| `W_mem` (`keep_in_rf`, `keep_in_warp`) | **2E** | RF store in the preload loop + RF load in the reduce loop |
| `W_mem` (`multiwarp_ub`) | **1E + 3·N·NW** | UB load per element + partial store + partial load + merge `max` |
| `W_emit` | **3N** | clamp to 1e-6, reciprocal, store `sf_inv` |
| legacy `keep_reg` / `ub_reload` | **+3E + 1N** | the sink'd subsequent compute (reload + `|x|·s` FFMA + sink store) |
| `W_loop` (SimtVF only) | **4·E·passes** when the inner serial trip > `UNROLL_MAX` (16) | a serial loop longer than the unroll budget stays a real branch loop: `SIMT_LEA` + `SIMT_ISETP_I` + `SIMT_BRANCH` + `SIMT_IADD` per element, none of which is "useful work". `passes` = 1 for `ub_stream`/`reload`/`multiwarp_ub`, 2 for the keep arms (preload pass + reduce pass) and the legacy arms (reduce + subsequent). Inner trip is `G`, or `G/NW` for `multiwarp_ub`. |

`W` is pure arithmetic/memory-op counting from shape + arm; it is **identical
for a5 and a6**. That is the point of splitting it out.

## Assumptions (all env-overridable)

| knob | env | default | rationale |
|------|-----|---------|-----------|
| `I_assum` Simt, **unrolled** body | `ST_VF_IPC_SIMT` | **0.20** | measured SimtVF proxy IPC on this suite sits ≈0.07–0.35 (SV1–SV8); the unrolled Case-3 bodies (G≤16) measure 0.22 |
| `I_assum` Simt, **branchy** body | `ST_VF_IPC_SIMT_BRANCHY` | **0.10** | every non-unrolled Case-3 body (G=32/64/128 with inner trip > 16) measures 0.088–0.106 |
| unroll budget | `ST_VF_UNROLL_MAX` | **16** | opcode histograms: inner trip ≤16 has **no** `SIMT_BRANCH`; trip 32/64/128 shows `SIMT_BRANCH:256` + `SIMT_ISETP_I:256`. `sv7 G=64 multiwarp_ub` (chunk = 16) is branch-free, `G=128 multiwarp_ub` (chunk = 32) is not — that single fact drives the SV7 winner flip. |
| loop ops per element pass | `ST_VF_W_LOOP_OPS` | **4** | LEA + ISETP + BRANCH + IADD seen per element in the branchy histograms |
| `I_assum` VMI (SimdVF EX IPC) | `ST_VF_IPC_VMI` | **0.85** | SV3V EX IPC measured 0.71–0.72; VMI band quoted 0.7–1.2 |
| `O_base` | `ST_VF_O_BASE` | **300 cyc** | VF enter/exit + UB prologue (GM→UB copy, flag setup) |
| `O_loop` | `ST_VF_O_LOOP` | **250 cyc** | each *extra* `T.Parallel` loop in the VF body (loop setup, index init) |
| `O_ub_roundtrip` | `ST_VF_O_UBRT` | **400 cyc** | one store→load UB round-trip incl. membar / set_flag+wait_flag visibility |
| `freq` | `ST_VF_FREQ_HZ` | **1.8e9** | camodel implied: `VF_SIMT.cycles / running_time(us)` = 5506/3.06 µs ≈ **1.799 GHz** on `Ascend950PR_9599`. The harvester re-derives `freq_implied` per tag and prints the range, so µs are never guessed. |

Overhead by arm class (`O_arm = O_base + O_loop·n_extra_parallel + O_ubrt·n_ub_roundtrips`):

| arm | extra Parallel loops | UB round-trips | `O_arm` |
|-----|---------------------:|---------------:|--------:|
| `ub_stream` / `reload` | 0 | 0 | 300 |
| `keep_in_rf` / `keep_in_warp` | 1 (preload + reduce) | 0 | 550 |
| `multiwarp_ub` | 1 (partials + merge) | 1 (`part_ub`) | 950 |
| legacy `keep_reg` | 1 | 0 | 550 |
| legacy `ub_reload` | 1 | 1 (`scale_ub`) | 950 |

## Calibration (2026-10-06 Simt ladder, pto-b10)

With the defaults above, all eight measured Simt arms land inside the `OK` band:

| tag | C_pred | C_meas | ratio |
|-----|-------:|-------:|------:|
| sv5 G16 `keep_in_rf` | 7190 | 10641 | 1.48 |
| sv5 G16 `ub_stream` | 5660 | 5506 | **0.97** |
| sv6 G32 `keep_in_warp` | 34070 | 24697 | 0.72 |
| sv6 G32 `reload` | 21020 | 20748 | **0.99** |
| sv7 G64 `multiwarp_ub` | 6370 | 11182 | 1.76 |
| sv7 G64 `reload` | 20900 | 24339 | 1.16 |
| sv7 G128 `multiwarp_ub` | 21730 | 25948 | 1.19 |
| sv7 G128 `reload` | 20840 | 24423 | 1.17 |

The two loosest rows are the RF-residency arms (`keep_in_rf` 1.48, `multiwarp_ub`
G64 1.76): both spill (`SIMT_LDK`/`SIMT_STK` in the histogram) or pay UB partial
traffic the model prices only as flat `O`. Residual >1 is therefore "cost the
model does not know", which is exactly the a5↔a6 quantity of interest.

## How a5 vs a6 plugs in later

Same `W` (shape+arm only), different machine terms:

```bash
# a5 profile (example)
ST_VF_IPC_SIMT=0.20 ST_VF_O_UBRT=400 ST_VF_FREQ_HZ=1.8e9 python3 harvest_vf_theory.py $OUT
# a6 profile (example: better issue, cheaper UB round-trip, higher clock)
ST_VF_IPC_SIMT=0.30 ST_VF_O_UBRT=250 ST_VF_FREQ_HZ=2.0e9 python3 harvest_vf_theory.py $OUT
```

The uarch efficiency statement is then `C_meas(a5)/C_pred(a5)` vs
`C_meas(a6)/C_pred(a6)` on an unchanged `W`, plus the `IPC_meas` column against
the assumed `I_assum`. Deviation of `ratio` from 1.0 isolates what the model
does **not** know (scheduling stalls, UB bank conflicts, RF spill), which is
exactly the quantity worth comparing between uarchs.

## Measuring `C_meas` on the two backends

- **SimtVF**: the camodel trace carries a single `VF_SIMT` row covering the whole
  VF region → that row's `cycles` is `C_meas` (and `running_time(us)` gives
  `freq_implied`). `meas_src = VF_SIMT`.
- **SimdVF / VMI**: this PTODSL emits **no** `VF_*` row and no `EXIPC`/`IFU`
  counters, so `C_meas` is the sum of `cycles` over the **vector pipes**
  (`RVECEX`/`RVECLD`/`RVECST`/`RVECSU`), i.e. the VF body only. That deliberately
  **excludes** the scalar transpose prologue and scalar reciprocal epilogue the
  Case-3 VMI ABI forces on us (see `reports/ST_CASE3_VMI.md`) — they are real
  wall-clock cost but are not what `W` models. `meas_src = RVEC pipes`.
- First VMI data point (`sv5v_r64_c128_g16_t64_ub_stream`, PASS, maxabs 0.0):
  vector pipes = 1490 instr / 11817 cycles (RVECEX alone: 772 / 5016) against
  `C_pred` 931 at `I_assum=0.85`. So the **VMI assumption is far off for this
  Case-3 body** (achieved vector IPC ≈0.13 RVEC-total, ≈0.15 RVECEX-only, not
  0.7–1.2): the harvester flags it `OUTLIER_SLOW`, which is the intended use of
  the model. Recalibrating `ST_VF_IPC_VMI` for Case-3 reduce bodies (or counting
  per-instruction vector latency ≈8 cycles explicitly) is the obvious next step;
  the SV3V GEMV numbers that produced 0.7 came from `EXIPC` counters on a
  different kernel class, so the two are not directly comparable. Wall µs for
  this tag is 109.39 µs, nearly all of it the scalar transpose prologue.

## Caveats

- `W` counts op *slots*, not issue slots: a fused `max(v,-v)` or an FFMA may be
  one instruction, so `W` over-counts slightly on arms the compiler fuses well,
  which shows up as `ratio < 1`.
- `O_*` values are engineering estimates, not measurements. They matter most on
  the small rungs (G=16) where `W_thr/I_assum` is only a few thousand cycles.
- RF spill (`LDK`/`STK` in the opcode histogram) is **not** modelled. A keep arm
  that spills shows up as `OUTLIER_SLOW` — that is a useful signal, not a bug.
- A `COMPILE_FAIL` arm still gets its theory row (`C_meas` blank, verdict
  `NO_MEAS`). No measured µs is ever invented.
