# ST CF1d–CF6d Layer-D summary

**Date:** 2026-10-07 HKT · Ascend950PR_9599 opsim (`core0.veccore0` µs)

| Twin | tag | wall µs | mapping |
|------|-----|--------:|---------|
| CF1d | `cf1d_e256_k8_t32_keep` | **0.90** | (3) KEEP thresh kill |
| CF2d | `cf2d_e256_k8_t32_remat` | **0.90** | (1) remat + kill-in-shared |
| CF3d | `cf3d_e256_k8_t32_remat_idx` | **0.83** | (3)+remat i |
| CF4d | `cf4d_e256_t32_pfat5` | **0.89** | (2) nested vsel |
| CF4d | `cf4d_e256_t32_pfat25` | **0.89** | same |
| CF5d | `cf5d_e256_t32_pnear5` | **0.89** | (2) fused div+near0 |
| CF5d | `cf5d_e256_t32_pnear25` | **0.89** | same |
| CF6d | `cf6d_e256_t32_pnear5` | **0.92** | (2)/(3) Newton in RF |
| CF6d | `cf6d_e256_t32_pnear25` | **0.93** | same |

All **PASS**. No PR push. See `ST_CF2D_CF4D.md`, `ST_CF5D_CF6D.md`, `ST_CF5_CF6_DIVERGENCE.md`.
