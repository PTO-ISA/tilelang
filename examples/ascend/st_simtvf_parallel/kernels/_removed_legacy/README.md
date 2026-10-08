# Removed from active ST list (2026-09-23; renumber SV1–SV9)

Kept as source reference for **ST-SV9 topk e2e** (and historical opsim tags).

| file | was | why removed |
|------|-----|-------------|
| sv2_topk_keep.py | old ST-SV2 topk KEEP micro | folded into SV9 e2e `keep` arm |
| sv3_topk_reload_remat.py | old ST-SV3 reload remat micro | folded into SV9 e2e `remat_scores` arm |
| sv4_topk_index_remat.py | old ST-SV4 index remat micro | folded into SV9 e2e `remat_idx` arm |
| sv5_bcast_multiconsumer.py | old SV5 | covered by SV2 (ex-SV1B) + Case-3 |
| sv6_group_scale_remat.py | old SV6 remat/reload Out[R,C] | Case-3 SV5/SV6/SV7/SV8 |
| sv7_block_reduce_128.py | 1D block reduce | number reused by Case-3 large-G SV7 |
| sv8_block_reduce_32x32.py | 2D block reduce | active SV8 = Case-3 quant e2e |

Do not wire these into oneshots. Active SV9: `kernels/sv9_topk_e2e.py`.
