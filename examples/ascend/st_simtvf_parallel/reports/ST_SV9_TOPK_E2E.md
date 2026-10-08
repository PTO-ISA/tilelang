# ST-SV9 — Topk e2e (keep / remat_scores / remat_idx)

**Status:** implemented (sources). Opsim µs **TBD**.  
**Kernel:** `kernels/sv9_topk_e2e.py`  
**Extracted from:** `kernels/_removed_legacy/sv2_topk_keep.py` (+ SV3 remat_scores / SV4 remat_idx forks)

## Purpose
One MoE-style topk end-to-end ST that **replaces** the three legacy topk micros.
Full tournament in one kernel with three arms:

1. Load `scores[E]` (+ optional `idxs[E]=i`).
2. For `k in serial(K)`: `reduce_max(scores)` → among ties `reduce_min(idx_cand)` → emit `IdxOut[k]` → **kill** winner (`NEG`).
3. Gold: stable **min-index-on-ties** top-K indices (same as REGRESSION-2 / legacy).

## Arms
| arm | scores | idxs | kill | probes |
|-----|--------|------|------|--------|
| **keep** | KEEP across K | KEEP `idxs` | NEG in RF | Hot K rewrite of large fragment RF |
| **remat_scores** | Reload from shared each K | KEEP idxs | NEG in shared | Reload + membar tax |
| **remat_idx** | KEEP scores | No idxs frag; remat `i` | NEG in RF via `i==best` | Index RF vs remat |

## GPU vs NPU
| concern | GPU | NPU SimtVF |
|---------|-----|------------|
| KEEP tournament | Large MRF holds scores±idxs across K | Folded `scores[E/T]`/thr; E=256 T=32 → 8 fp32/thr |
| Remat scores | Explicit LDS reload each K | Shared stage + Parallel remat; tax ↑ with K |
| Remat idx | Cheap integer remat | Drop int fragment; compare remat `i` to `best` |
| T=128 | Occupancy vs RF | Shrinks elems/thr; can change reduce path (legacy: T128 K8 ≫ T32) |

## Tags (primary oneshot)
- `sv9_e256_k1_t32_keep`
- `sv9_e256_k8_t32_keep`
- `sv9_e256_k8_t32_remat_scores`
- `sv9_e256_k8_t32_remat_idx`
- optional: `sv9_e256_k8_t128_keep`

## Run
```bash
bash oneshot_sv9.sh
# or
bash oneshot_sv1_sv9.sh
```

## Why this replaces three micros
| old micro | SV9 arm |
|-----------|---------|
| SV2 topk KEEP | `keep` |
| SV3 reload remat | `remat_scores` |
| SV4 index remat | `remat_idx` |

µs / IPC: fill after first Ascend950PR_9599 opsim. Legacy archived numbers may appear under RESULTS “Archived”.
