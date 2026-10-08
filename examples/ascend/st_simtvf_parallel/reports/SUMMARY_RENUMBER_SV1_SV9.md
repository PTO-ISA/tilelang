# SUMMARY — SimtVF Parallel ST renumber to SV1–SV9

**Date:** 2026-09-23 ~18:40 HKT  
**Suite:** `/workspace/st_simtvf_parallel`  
**Deliverable tarball:** `/workspace/st_simtvf_parallel_sv1_sv9_src.tgz`

## File renames

| Old | New |
|-----|-----|
| `kernels/sv1b_eltwise_bcast_rf.py` | `kernels/sv2_eltwise_bcast_rf.py` (`sv1b_*`→`sv2_*`) |
| `kernels/sv2_gemv_partial_keep.py` | `kernels/sv3_gemv_partial_keep.py` (`sv2g_*`→`sv3_*`) |
| `kernels/sv6_reduce_large_eltwise.py` | `kernels/sv7_reduce_large_eltwise.py` (`sv6_g64/128`→`sv7_*`) |
| `kernels/sv6v_reduce_large_eltwise.py` | `kernels/sv7v_reduce_large_eltwise.py` |
| *(new)* | `kernels/sv6_reduce_mid_eltwise.py` + `sv6v_*` (G=32) |
| *(new)* | `kernels/sv9_topk_e2e.py` (keep / remat_scores / remat_idx) |
| `reports/ST_SV1_AND_SV1B.md` | `ST_SV1_AND_SV2.md` |
| `reports/ST_SV2_GEMV.md` | `ST_SV3_GEMV.md` |
| `reports/ST_SV6_REDUCE_LARGE.md` | `ST_SV7_REDUCE_LARGE.md` |
| *(new)* | `ST_SV6_REDUCE_MID.md`, `ST_SV9_TOPK_E2E.md` |
| `reports/ST_LIST_DESIGN.md` | rewritten locked SV1–SV9 table |

Kept: SV1 stream, SV4 gather_psum (keep2 replaced), SV5 small, SV8 quant e2e, SV5V. Legacy micros stay in `kernels/_removed_legacy/` (not wired).

## Oneshot (all Simt SV1–SV9)
```bash
bash oneshot_sv1_sv9.sh
# laptop relay:
powershell -File .\run_via_laptop_sv1_sv9.ps1
```

Per-family: `oneshot_sv2.sh`, `oneshot_sv3.sh`, `oneshot_sv4.sh`, `oneshot_sv5_sv6_sv7_sv8.sh`, `oneshot_sv9.sh`.

## SV9
Extracted tournament topk from `_removed_legacy/sv2_topk_keep.py` (+ remat forks) into
`sv9_topk_e2e.py`. Gold = stable min-index-on-ties top-K. Arms: KEEP (NEG in RF),
`remat_scores` (kill in shared), `remat_idx` (no idxs fragment).

## Success check
- Contiguous SV1–SV9 in `reports/ST_LIST_DESIGN.md`
- Tarball sources+md only at `/workspace/st_simtvf_parallel_sv1_sv9_src.tgz`
