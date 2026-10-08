# ST-CF5d / CF6d — PTO-DSL Layer D (vector Div/Newton in RF)

**Date:** 2026-10-07 ~10:20 HKT (emit-mlir + opsim ~10:18 HKT on pto-b10)  
**Box tree:** `/workspace/st_simtvf_parallel/`  
**Intent:** beat CF5v/CF6v **ABI tax** by keeping Div / Newton entirely in vector ISA (vregs), not Kernel-serial outside SimdVF.

## Why *d* (not *v*)

| Twin | Path | Problem | e2e µs |
|------|------|---------|-------:|
| CF5 (Simt) | `T.SimtVF` Parallel `if abs(a)<eps: 0 else a/b` | — | **1.18** |
| CF5v (SimdVF) | Div in Kernel `serial` **outside** SimdVF; VF only Select | VerifyParallelToPTO rejects Div in Parallel; serial-in-VF → AIV scalar fail | **6.23** |
| **CF5d (ptodsl)** | `vdiv` + `vcmp`/`vsel` in RF | — | **0.89** |
| CF6 (Simt) | Parallel near-0 + `serial(N_FAST)` Newton | — | **1.37** |
| CF6v (SimdVF) | Newton Kernel-serial outside VF; VF publish only | same ABI | **12.35** |
| **CF6d (ptodsl)** | seed `vdiv` + 3× `vmul`/`vsub` Newton in vregs | — | **0.92** |

A/B numbers from `reports/ST_CF_IMPLEMENT.md` (2026-09-28). D numbers: opsim `core0.veccore0` duration_time(us) on Ascend950PR_9599 (2026-10-07).

**Result:** CF5d/CF6d eliminate the outside-VF serial remat tax → **~7× / ~13×** faster than CF5v/CF6v, and slightly faster than Simt CF5/CF6 on this micro.

**Forbidden:** AABBCC / `vf_fuse` / `token_tile` / TileLang `T.Parallel` *v rewrite. Pure `@pto.jit` explicit vpto only.  
**Not pushed** to PR #272 in this pass.

## Sources / tags

| Twin | File | Primary tag | Also |
|------|------|-------------|------|
| CF5d | `kernels_ptodsl/cf5d_div_ulp_branch.py` | `cf5d_e256_t32_pnear5` | `pnear25` via argv |
| CF6d | `kernels_ptodsl/cf6d_newton_branch.py` | `cf6d_e256_t32_pnear5` | `pnear25` via argv |

Harness style matches CF1d / SV1d: `@pto.jit(..., backend=vpto, mode=explicit)`, `mte_load/store`, `vmi.vload/vstore`, ACL `run_acl` + `--emit-mlir`.  
`E=256`, `T=32` (tag mirror), `VL=64`, `N_CHUNKS=4`. `EPS=1e-4` fixed; `pnear` is **host-data skew only**.

## Ops used (VMI rooftop)

Confirmed present in camodel `.camodel_deps_vmi018` `pto.vmi`: `vdiv`, `vmul`, `vsub`, `vmax`, `vcmp`, `vsel`, `vbrc`, `vload`, `vstore` (also `vabs`/`vneg` exist but unused — prefer `vmax(v,vmul(v,-1))`).

### CF5d — mapping (2) fused compute + near-0 Select

Per VL chunk (×4):

1. `vload` a, b  
2. abs: `vmax(a, vmul(a, -1))`  
3. `near = vcmp(ax, eps, "lt")`  
4. `quot = vdiv(a, b)`  
5. `y = vsel(near, 0, quot)`  
6. `vstore` y  

emit-mlir ops count: 4×`vdiv`, 4×`vmax`, 4×`vmul`, 4×`vcmp`, 4×`vsel` (88 lines).

### CF6d — mapping (2)/(3) fused Select + stay-alive y across Newton

Per VL chunk (×4):

1. `vload` x; abs via `vmax`/`vmul(-1)`  
2. `near = vcmp(ax, eps, "lt")`  
3. `safe_x = vsel(near, 1, x)` — avoid Inf/NaN on near-0 lanes  
4. seed: `y = vdiv(1, safe_x)` in vreg  
5. `N_FAST=3` iters **in vregs** (no UB remat of intermediate y):  
   `y = vmul(y, vsub(2, vmul(safe_x, y)))`  ≡ `y*(2-x*y)`  
6. `y = vsel(near, 0, y)`; `vstore`

emit-mlir ops count: 4×`vdiv`, 12×`vsub`, 28×`vmul`, 4×`vmax`, 4×`vcmp`, 8×`vsel` (116 lines).

## Host gold

Matches Simt / `run_opsim_generic.py` recipes (seeds 15/16, force first `n_near=round(E*pnear/100)` lanes near-0).  
ACL check: `rtol=2e-5 atol=1e-6`.

## Status (2026-10-07)

| Check | Result |
|-------|--------|
| Sources landed (box) | **YES** |
| Host gold sanity (numpy) | **OK** |
| emit-mlir (pto-b10, camodel deps) | **OK** — CF5d 88 lines, CF6d 116 lines |
| opsim Ascend950PR_9599 | **PASS** both primary tags |
| CF5d `cf5d_e256_t32_pnear5` | **PASS** wall **0.89 µs** (`core0.veccore0`); maxabs≈4.8e-7 |
| CF6d `cf6d_e256_t32_pnear5` | **PASS** wall **0.92 µs** (`core0.veccore0`); maxabs≈2.4e-7 |
| CF5d `cf5d_e256_t32_pnear25` | **PASS** wall **0.89 µs** (same as pnear5) |
| CF6d `cf6d_e256_t32_pnear25` | **PASS** wall **0.93 µs** (≈pnear5 0.92) |
| PR #272 | **not pushed** (per request) |

Logs on pto-b10: `/tmp/st_simtvf_parallel/ptodsl_cf5d_cf6d/`.

### Re-run command (pto-b10)

```bash
export PTODSL_DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
# Do NOT prepend bare $PTOAS_ROOT/ptodsl (missing ptoas.mlir) — prefer camodel deps first.
cd /path/to/st_simtvf_parallel
bash oneshot_ptodsl_cf5d_cf6d.sh
```

Oneshot: `oneshot_ptodsl_cf5d_cf6d.sh` — emit-mlir both + opsim pnear∈{5,25}.  
IPC/EXIPC: not harvested this pass (wall µs only).
