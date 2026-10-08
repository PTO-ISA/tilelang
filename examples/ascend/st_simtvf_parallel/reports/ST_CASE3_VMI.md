# Case-3 VMI / SimdVF twins — SV5V / SV6V / SV8V

> **Renumber:** VMI twins track Case-3: sv5v small / sv6v mid / sv7v large / sv8v e2e.

**Date:** 2026-09-23 ~17:15 HKT  
**Suite:** `/workspace/st_simtvf_parallel`  
**Status:** sources + gold + oneshot **ready**; opsim **BLOCKED** this pass

## Why blocked

Sibling Simt agent was mid-`oneshot_sv5_sv6_sv8` on pto-b10 (compiling `sv5_reduce_small_eltwise.py`) and holds **deps-native** `libtilelang.so`. VMI needs **overlay** lib (`GetPredicate(PrimExpr)` + `SimdVFLower*`). Mixing overlays mid-oneshot is a known hard fail — skipped opsim; left oneshot scripts ready.

## Contract (do not mix)

| arm | target | lib | VF |
|-----|--------|-----|-----|
| Simt Case-3 | `ascend` + cython | deps-native | `T.SimtVF(threads=)` + `alloc_fragment` |
| VMI twin | `pto` | **overlay** | `T.SimdVF(lanes=)` + prefer `alloc_shared` |

## Files

| kind | path |
|------|------|
| harness | `common_pto_harness.py` |
| SV5V | `kernels/sv5v_reduce_small_eltwise.py` |
| SV6V | `kernels/sv7v_reduce_large_eltwise.py` |
| SV8V | `kernels/sv8v_case3_bcast.py` |
| gold | `run_opsim_generic.py` branches `sv5v_` / `sv6v_` / `sv8v_` |
| oneshot | `oneshot_sv5v_sv6v_sv8v.sh` |
| laptop | `run_via_laptop_sv5v_sv6v_sv8v.ps1` |

## Tags

| tag | IO |
|-----|-----|
| `sv5v_r64_c128_g16_t64` | X fp16 `[64,128]` → Y fp32 `[64,8]` |
| `sv6v_r64_c128_g64_t64` | Y fp32 `[64,2]` |
| `sv6v_r64_c128_g128_t64` | Y fp32 `[64,1]` (optional) |
| `sv8v_r64_c128_g16_t64_live` | Out fp16 `[64,128]` |
| `sv8v_r64_c128_g16_t64_spill_dist` | same; expand `[R,CG]→[R,C]` in shared |

## Missing env (if opsim fails later)

1. Overlay tree: `/mnt/fluxdata/happybot/projects/tilelang-pr258-overlay/build/lib/libtilelang.so`
2. Simt oneshot idle (no deps-lib lock)
3. `common_pto_harness` + `T.SimdVF(lanes=)` FFI on overlay Python
4. camodel / `sim_dsl.sh` + `run_cf_mb_opsim.py` as for Simt

## Run (after Simt idle)

```powershell
powershell -File .\run_via_laptop_sv5v_sv6v_sv8v.ps1
```


Also mid twin: `kernels/sv6v_reduce_mid_eltwise.py` (G=32).

---

**Update 2026-09-24:** Full VMI matrix SV1V–SV9V now lives in [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md) + `oneshot_sv1v_sv9v.sh`. This doc remains Case-3-focused (SV5V–SV8V).

---

## 2026-10-06 — Case-3 VMI twins rewritten for the RF-capacity ladder (SV5V/SV6V/SV7V)

Arms now mirror the Simt ladder: `sv5v {keep_in_rf | ub_stream}`,
`sv6v {reload | keep_in_warp}`, `sv7v {reload | multiwarp_ub}` (tags
`sv*v_r64_c128_g{G}_t64_{arm}`, oneshot `oneshot_sv567v_rfladder.sh`,
PYTHONPATH **DEPS before CAMO**).

### Three stacked ABI walls (each found by fixing the previous one)

| # | Symptom | Cause | Fix in the twin |
|---|---------|-------|-----------------|
| 1 | `[VerifyParallelToPTO] bind \`vv\`: buffer \`xf\`: cannot prove the address is continuous (data_offset == start + lane over the full lane range) or lane-uniform [unit #2 (loop var \`n\`, extent 64)]` | the natural Case-3 body reduces group `n` over `t`, i.e. reads `xf[n*G + t]` — stride `G` across the lane domain `n`. PTO's first version accepts only lane-continuous or lane-uniform addresses. | stage a **transposed** working set `xt[t, n]`; every VF access becomes `xf[t*N + n]`, continuous over the lanes. |
| 2 | `Check failed: stride % 32 == 0 (2 vs 0): PTO GM->UB MTE destination row stride must be 32-byte aligned for potentially multi-burst copies, got 2 bytes` | the obvious transpose `T.copy(X[:, t], xt[t, :])` is a strided GM→UB MTE with a 2-byte row stride. The UB→UB variant (`T.copy(x_ub[:, t], …)`) crashes the overlay (rc=127). | contiguous `T.copy(X, x_ub)` plus a **scalar transpose loop outside the VF region**. It is prologue cost only — the VF cycles the theory model compares against do not contain it. |
| 3 | `[VerifyParallelToPTO] store value: unsupported expression node \`Div\`` | `1.0 / m` inside a `Parallel` is not on the Parallel→PTO whitelist (same wall as CF5V). | the reciprocal **and** the 1e-6 clamp run in a Kernel-scope `serial(N)` loop **outside** `T.SimdVF`. |

Plus the SV3V rules that already applied: no `T.alloc_var` in the VF body (the
running absmax lives in `m_ub` in UB), one lane-aligned vector f16→f32 upcast,
every `Parallel` 1-D over `N = R*CG` (512/256/128/64 — all multiples of
lanes=64, so no `pto.vmi.mask_and` is needed), no guards in the body.

### Status

- Before the rewrite: **all 8 arms COMPILE_FAIL** (wall #1).
- After the rewrite: `sv5v_r64_c128_g16_t64_ub_stream` is **COMPILE_OK** — the
  first Case-3 VMI twin to get past the Parallel→PTO verifier. opsim for the VMI
  matrix is slow (the scalar transpose prologue is ~N·G simulated scalar
  iterations) and was still running when this note was written; numbers land in
  `/tmp/pr272_sv567_rfladder_20261006/RESULT.md` +
  `PASS_TABLE_sv1v_sv9v.txt` as each tag finishes. **No µs is recorded until the
  camodel CSV exists.**
- Gold is unchanged (`sv5v` seed 5, `sv6v`/`sv7v` seed 6); the flat `(N,G)` /
  `(N,)` tensor declarations are byte-identical to the 2-D `(R,C)` / `(R,CG)`
  ones, so `run_opsim_generic.py` compares against the same reference.
