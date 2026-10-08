# ST-SV3V — SimdVF / VMI twin fix (COMPILE_FAIL → PASS)

**Date:** 2026-10-05 HKT · **Job:** REGRESSION-2 · **Host:** pto-b10
**Branch:** PR #272 `feat/st-simtvf-parallel-sv1-sv9` tip `4a01f71e` (nothing committed or pushed)
**Kernel:** `kernels/sv3v_gemv_partial_keep.py` · **Twin of:** `kernels/sv3_gemv_partial_keep.py` (SimtVF)
**Toolchain:** `target=pto` + pr258 overlay `libtilelang.so` + deps-stack python, PTODSL `ptoas_vmi 0.1.8`,
simulator `Ascend950PR_9599`.

## Before

```
tvm.error.InternalError: [VerifyParallelToPTO] memory access: buffer `xk` has scope `local.var`;
the first version supports only UB (shared/shared.dyn) accesses [at region #0]
```
(full log: `/tmp/pr272_sv3v_fix_20261005/compile_before.log`, shape m32 keep lanes=64)

All six `sv3v_*` rows in `PASS_TABLE_sv1v_sv9v.txt` were `COMPILE_FAIL rc=1`.

## Four stacked ABI blockers (each found by fixing the previous one)

| # | Symptom | Cause | Fix in the twin |
|---|---------|-------|-----------------|
| 1 | `[VerifyParallelToPTO] buffer \`xk\` has scope \`local.var\`` | `xk = T.alloc_var("float32")` for the per-K x reload becomes a `local.var` buffer; the PTO Parallel path only accepts UB (`shared`/`shared.dyn`). | Drop `alloc_var`; read the scalar straight out of UB (`x_f32[k]`). Its index does not depend on the vectorised Parallel var, so it lowers to a broadcast (`RV_VDUPS`) instead of a local var. |
| 2 | `Unsupported PTO cast: float16 -> float32` | `T.Cast("float32", x_ub[k])` is lane-invariant → it is emitted in the **scalar** domain, and `codegen_pto` only has scalar `f32 <-> bf16`. | Upcast x **vectorially, once**: a lanes-wide `T.Parallel` does `x_f32[t] = T.Cast("float32", x16[t])` (vload f16 + `RV_VCVT_F2F`). The K loop then broadcasts an already-f32 scalar. `x16` is padded to `lanes` so the cast loop is lane-aligned; only the first K lanes are read back. |
| 3 | `AttributeError: '_VMINamespace' object has no attribute 'mask_and'` | A `T.Parallel` extent that is not a multiple of `lanes` makes `SimdVFExpandParallelDomain` emit a domain mask, and the installed PTODSL has no `pto.vmi.mask_and` (`simdvf_lower_control_flow.cc` notes the same gap). 2-D `T.Parallel(M, VL)` with M=32/24 trips it. | Every `T.Parallel` is 1-D over a lane-aligned extent: `M*VL` = 2048 / 1536, `CM*VL` = 1024, and `lanes` = 64. This is the same iteration space in the same element order, just linearised; buffers are declared flat (`(K, M*VL)`, `(M*VL,)`) so the GM/UB bytes are byte-identical to the 3-D/2-D declarations. |
| 4 | (split arm) non-continuous / predicated store | `if row < M` inside the Parallel body makes the body a guarded store, which the Parallel→PTO path rejects. | Guard dropped; the arm now requires `M % CM == 0` (raises otherwise). The oneshot shape m32/CM16 satisfies it. |

The A-side `T.Cast("float32", a_ub[k, t])` is *kept* in the VF body: it is lane-varying, so it
vectorises to `RV_VCVT_F2F` (visible in the EX tables of `ST_VMI_COMPARE.md`).

## Contract preserved (Lok's SV3 sensitivity)

- **keep** — `acc[M*VL]` is zeroed once and stays live across the whole outer `T.serial(K)`; it is
  never re-materialised from UB per K. A (`a_ub[k, …]`) and x (`x_f32[k]`) are re-read from shared
  every K. Mapping mode **3**.
- **split** — `acc_c[CM*VL]` only; the chunk loop `T.serial(NC)` flushes to `out_ub` **once per
  chunk**, so barriers scale with chunk count, not K.
- Math/dtypes unchanged: `Acc += A[k]*x[k]`, A/x fp16, Acc/Out fp32, VL=64, K=16, lanes=64,
  CM=16, same CLI (`M VL K lanes arm [CM]`) and same tags.
- Not changed: no token tiling, no AABBCC/`vf_fuse`, no extra arms, no multi-SimdVF schedule.

## After — compile + opsim (Ascend950PR_9599)

| Tag | Compile | opsim | wall µs | check | Simt µs | Δµs |
|-----|---------|-------|--------:|-------|--------:|----:|
| `sv3v_m24_vl64_k16_t64_keep` | COMPILE_OK | **PASS** | 4.57 | `maxabs=0.0` | 6.79 | −2.22 |
| `sv3v_m32_vl64_k16_t64_keep` | COMPILE_OK | **PASS** | 5.81 | `maxabs=0.0` | 8.63 | −2.82 |
| `sv3v_m32_vl64_k16_t64_split_cm16` | COMPILE_OK | **PASS** | 5.84 | `maxabs=0.0` | 7.79 | −1.95 |

Cycles / instr# / IPC (`IPC_proxy(VF)`), from `harvest_simt_vmi_compare.py`:

| Pair | Simt cycles | VMI cycles | Simt instr | VMI instr | Simt IPC | VMI IPC |
|------|------------:|-----------:|-----------:|----------:|---------:|--------:|
| sv3_m24_keep ↔ sv3v_m24_keep | 9105 | 5181 | 2941 | 3701 | 0.323 | 0.714 |
| sv3_m32_keep ↔ sv3v_m32_keep | 11874 | 6861 | 4032 | 4885 | 0.340 | 0.712 |
| sv3_split_cm16 ↔ sv3v_split_cm16 | 10384 | 6950 | 3680 | 4997 | 0.354 | 0.719 |

EX-pipe highlight: all `SIMT_LDS`/`SIMT_FFMA`/`SIMT_F2F`/`SIMT_PRMT` disappear and are replaced by
`RV_VMUL` + `RV_VADD` + `RV_VCVT_F2F` + `RV_VDUPS` (per-K broadcast of `x_f32[k]`). Full tables in
[`ST_VMI_COMPARE.md`](ST_VMI_COMPARE.md).

## Notes for the other blocked twins

SV2V / SV4V fail with the same `local.var` message, and SV5V–SV8V with the Case-3 lane-alignment
problem. The three rules this fix establishes should carry over:

1. no `T.alloc_var` inside a SimdVF Parallel — keep scalars in UB and read them lane-invariantly;
2. no f16 → f32 cast in the scalar domain — do it once, vectorially, over a lane-aligned extent;
3. every Parallel extent must be a multiple of `lanes` (no `mask_and` in this PTODSL), and no
   guards inside the Parallel body.

## Repro (pto-b10, nothing committed)

```bash
# work dir: /tmp/pr272_sv3v_fix_20261005   (suite copy + clean worktree at 4a01f71e)
bash /tmp/pr272_sv3v_fix_20261005/cc_any.sh  suite/kernels/sv3v_gemv_partial_keep.py \
     sv3v_m32_vl64_k16_t64_keep 32 64 16 64 keep
bash /tmp/pr272_sv3v_fix_20261005/sim.sh     sv3v_m32_vl64_k16_t64_keep
```

`envrc.sh` is the oneshot env with one correction: `$DEPS` must precede `$CAMO` on `PYTHONPATH`,
otherwise `import tilelang` resolves to the camodel copy, whose `ascend/pipeline.py` has no
`target_is_pto` branch and which pulls the Aug-03 venv `libtilelang.so` — that configuration lowers through
`AscendSimdVFLowerParallel` and fails even on the known-good CF twins. It also uses a private
hard-link copy of the deps stack with the overlay `libtilelang.so` dropped in, so the shared
`/home/happybot/projects/tilelang-pto-vmi-deps-stack/build/lib/libtilelang.so` is never modified.
