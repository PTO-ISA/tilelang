# ST-CF2d / CF3d / CF4d — PTO-DSL Layer D

**Date:** 2026-10-07 ~10:34 HKT (emit-mlir + opsim on pto-b10 Ascend950PR_9599)  
**Box tree:** `/workspace/st_simtvf_parallel/`  
**Intent:** finish remaining CF Layer-D twins after CF1d / CF5d / CF6d.

## Sources / tags

| Twin | File | Mapping | Primary tag |
|------|------|---------|-------------|
| CF2d | `kernels_ptodsl/cf2d_remat_thresh_kill_shared.py` | **(1)** remat scores each K + kill-in-shared (UB storeback) | `cf2d_e256_k8_t32_remat` |
| CF3d | `kernels_ptodsl/cf3d_remat_idx.py` | **(3)** scores KEEP + remat `i` via `vci` each use | `cf3d_e256_k8_t32_remat_idx` |
| CF4d | `kernels_ptodsl/cf4d_nested_if.py` | **(2)** nested `vsel` hi/lo/fat | `cf4d_e256_t32_pfat{5,25}` |

Pattern matches CF1d/CF5d: pure `@pto.jit` explicit vpto; `VL=64` `E=256`; no AABBCC/`vf_fuse`; no PR #272 push.  
Opsim via laptop→pto-b10 with `.camodel_deps_vmi018` (not bare `PTOAS-vmi/ptodsl`).

### Kernel notes

- **CF2d vs CF1d:** CF1d KEEP scores in vregs across K; CF2d reloads from UB each k and stores kill back to shared (publish tax). Same gold as CF1.
- **CF3d:** KEEP `s0..s3`; each k rematerializes lane indices `i{c}=vci(0)+c*VL`, `vsel(i==victim[k], NEG, s)`.
- **CF4d:** `y = vsel(x>hi, hi, vsel(x<lo, lo, x*scale))`. `pfat` is host skew only. **UB layout:** `lo`/`hi`/`out` padded to **32B alignment** (MTE/VSTI reject unaligned `0x804`/`0x808`).

## Opsim results (core0.veccore0 µs)

| tag | status | wall µs | maxabs | emit-mlir lines |
|-----|--------|--------:|-------:|----------------:|
| `cf2d_e256_k8_t32_remat` | **PASS** | **0.90** | 0 | 272 |
| `cf3d_e256_k8_t32_remat_idx` | **PASS** | **0.83** | 0 | 242 |
| `cf4d_e256_t32_pfat5` | **PASS** | **0.89** | 0 | 104 |
| `cf4d_e256_t32_pfat25` | **PASS** | **0.89** | 0 | (same binary) |

Logs: `/tmp/st_simtvf_parallel/ptodsl_cf2d_cf4d/` on pto-b10.  
Oneshot: `oneshot_ptodsl_cf2d_cf4d.sh` (also merged `oneshot_ptodsl_cf1d_cf6d.sh`).

## Twin compare (Simt / VMI *v / Layer-D *d)

| case | Simt µs | *v µs | *d µs | note |
|------|--------:|------:|------:|------|
| CF1 | 0.97 (k8) | 1.01 | **0.90** | KEEP thresh kill |
| CF2 | 1.95 | 1.04 | **0.90** | remat+publish; Simt pays most |
| CF3 | 1.33 | 1.02 | **0.83** | remat_idx |
| CF4 | 1.23–1.25 | 0.93 | **0.89** | nested; pfat flat on all layers |

Simt A/B from `ST_CF_IMPLEMENT.md` (2026-09-28). Layer-D is the vector rooftop for these micros at E=256.

## Re-run

```bash
export PTODSL_DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
cd /path/to/st_simtvf_parallel
bash oneshot_ptodsl_cf2d_cf4d.sh
# or full: bash oneshot_ptodsl_cf1d_cf6d.sh
```
