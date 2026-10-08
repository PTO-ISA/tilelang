# TileKernels packed-structure survey — MoE / MX-quant vs ST SimtVF suite

**Date:** 2026-09-28 ~10:23 HKT  
**Source repo:** [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels) (`main` @ `36d9e45`, MIT, TileLang GPU kernels)  
**Suite context:** `/workspace/st_simtvf_parallel` (SV1–SV9 + CF1–CF6)  
**Scope:** Read-only survey of packed layouts (esp. MoE + MX-style SF). **No CF VMI / pto-b10 opsim.** Box-only deliverable (not pushed to PR #272).

---

## 1. Executive answer (one paragraph)

TileKernels does **not** ship attention / SFA / KV-cache kernels. Its MoE path *does* implement the structural pattern Lok asked about: **quantized activations travel as a logical `QuantTensor = (data, sf)` pair**, and MoE expand/reduce **co-moves value and scale under the same index map** (`token_topk_to_pos`). Scales are often **MX-style**: power-of-two **UE8M0**, optionally **packed 4× into `int32`**, with **TMA col-major** SF layout for SM100. That is the same *shape* of problem as Ascend **SFA gather/scatter of scale+value for KV-cache-like layouts** — not the same kernel, but the same packed-structure sensitivity (index KEEP/remat, dual-tensor bandwidth, SF packing density, alignment pad). ST suite today covers **gather of values** (SV4), **local group-scale compute+bcast** (SV5–SV8), and **topk indices** (SV9) separately; it **lacks** co-scatter/gather of `(value, sf)`, packed-UE8M0 load/store, and expert-major fused-buffer padding.

---

## 2. Repo map (relevant paths)

| Area | Paths | Why it matters |
|------|-------|----------------|
| MoE API | `tile_kernels/moe/__init__.py` | Exports `expand_to_fused(_with_sf)`, `reduce_fused`, `get_fused_mapping`, `topk_gate`, `top2_sum_gate`, … |
| MoE expand/scatter | `tile_kernels/moe/expand_to_fused_kernel.py` | **Core pack pattern:** scatter `x` and optional `x_sf` via `token_topk_to_pos` |
| MoE mapping | `tile_kernels/moe/get_fused_mapping_kernel.py` | Builds expert-major fused layout + per-expert aligned ranges |
| MoE reduce/gather | `tile_kernels/moe/reduce_fused_kernel.py` | Gather expanded rows; optional `topk_weights` × per-pos `x_sf` |
| MoE topk | `tile_kernels/moe/topk_gate_kernel.py`, `top2_sum_gate_kernel.py`, `common.py` | Routing indices/weights only (no MX in gate itself) |
| Quant / MX SF | `tile_kernels/quant/common.py`, `types.py` | `QuantTensor`, `use_packed_ue8m0`, `load_sf`/`store_sf`/`transform_sf` |
| Cast kernels | `quant/per_token_cast_kernel.py`, `per_block_cast_kernel.py`, FP4/E5M6 variants | Produce `(out, out_sf)`; FP4 packs 2×e2m1 per `int8` |
| Torch refs | `tile_kernels/torch/expand_to_fused.py`, `moe.py`, `cast.py` | Clear semantics for expand_with_sf |
| Tests | `tests/moe/test_expand_to_fused.py`, `tests/quant/*` | Param matrix: `num_per_channels∈{32,128}`, packed UE8M0 + col-major |
| Sibling (ecosystem) | deepseek-ai/DeepGEMM | Mega MoE / FP8×FP4 + packed UE8M0 TMA helpers (consumer of TileKernels SF layouts) |

**Not present in TileKernels:** `SFA`, `KV`, `cache`, attention gather/scatter. Analogy is **structural only**.

---

## 3. Pattern catalog

### 3.1 Logical pack: `QuantTensor = (data, sf)`

```python
# tile_kernels/quant/types.py
QuantTensor = tuple[torch.Tensor, torch.Tensor]
```

Physically **two tensors**, not a bit-interleaved struct. Call sites treat them as one logical object. This matches “scale+value packed” at the **API / movement** layer (always move both; same indices), not necessarily at the **byte-interleave** layer.

### 3.2 MX-style scale packing (UE8M0 / TMA)

From `tile_kernels/quant/common.py`:

| Knob | Meaning |
|------|---------|
| `use_packed_ue8m0` | 4 UE8M0 exponents packed; view as `int32`; `sf_torch_dtype=uint8` internally |
| `use_tma_aligned_col_major_sf` | SF stored `(K_blocks, M_blocks)` with TMA alignment (pad to 4/16) |
| `round_sf` | Round SF to power-of-two (exponent-only) |
| `sf_block` | `(num_per_tokens, num_per_channels)` — e.g. per-token `(1,128)` or per-block `(128,128)` |
| FP4 `e2m1` | Values packed 2-per-byte in `int8` (`get_logical_hidden` / `unpack_from_e2m1fn_x2`) |

**SF load/store macro (packed UE8M0):**

```python
# common.py — store_sf / load_sf
if config.use_packed_ue8m0:
    tensor[k_idx // 4, m_idx * 4 + k_idx % 4] = sf
# transform_sf: UE8M0 → fp32 via (uint32(sf) << 23)
```

**SF compute (`get_sf_and_inv`):** `sf = amax / max_value(dtype)`; optional ceil-log2 round; returns `(sf, sf_inv)`.

This is DeepSeek’s MX / SM100 path (also documented in DeepGEMM: SM90 = FP32 SF, SM100 = packed UE8M0).

### 3.3 MoE routing → fused expert-major layout

`get_fused_mapping(topk_idx, num_experts, num_expanded_tokens, alignment)` returns:

| Tensor | Role |
|--------|------|
| `token_topk_to_pos[T,K]` | Scatter index: token×topk → fused position (−1 unused) |
| `pos_to_expert[Eexp]` | Expert id at fused pos (−1 pad) |
| `pos_to_token`, `pos_to_token_topk` | Reverse maps |
| `expert_start/end`, `num_tokens_per_expert` | Per-expert ranges, **aligned** to `alignment` (pad slots) |

Pattern: **sparse index map + dense padded expert-major buffer** — same family as KV-cache block tables / SFA gather tables (indirection + pad).

### 3.4 Co-scatter: `expand_to_fused_with_sf` (★ primary MoE pack)

```python
# expand_to_fused_kernel.py (simplified)
T.copy(token_topk_to_pos[pid], pos_local)
T.copy(x[pid], x_fragment)
T.copy(x_sf[pid], x_sf_fragment)          # when num_per_channels is set
for k in T.serial(num_topk):
    if pos_local[k] >= 0:
        expanded_x[pos_local[k], :] = x_fragment
        expanded_x_sf[pos_local[k] or :, ...] = x_sf_fragment   # same pos
# pad slots (pos_to_expert < 0): zero both value and sf
```

**API:** `expand_to_fused_with_sf((x_fp8, x_sf), num_per_channels, token_topk_to_pos, pos_to_expert, use_tma_aligned_col_major_sf)`.

Tests (`tests/moe/test_expand_to_fused.py`) matrix: `num_per_channels∈{32,128}`, `(col_major=False, packed_ue8m0=False)` vs `(True, True)`.

**Analogy to SFA gather/scatter KV:**  
`token_topk_to_pos` ≈ cache-slot / page index; `(x, x_sf)` ≈ `(value, scale)` payload written to the same slot. Scatter is **indexed write of a packed pair**.

### 3.5 Co-gather + weighted reduce: `reduce_fused`

```python
# reduce_fused_kernel.py (simplified)
for k in T.unroll(num_topk):
    pos = topk_to_pos_local[k]
    if pos >= 0:
        s = topk_weights[k] if with_weights else 1
        if with_x_sf: s *= x_sf[pos]          # per-expanded-token SF
        reduced += x[pos, :] * s
# optional global out sf for FP8
```

Gather of values under the same map; SF can be (a) per-pos sideband multiply, or (b) global scalar for FP8 out. **Not** a fused byte-pack decode in this kernel — SF is already a separate tensor.

### 3.6 Gate / topk (no MX in routing)

- `topk_gate`: Parallel scores KEEP + serial K reduce_max + min-index tie-break + kill −∞ → `topk_idx[T,K]`.  
  **Direct cousin of ST SV9 / CF1–CF3** (suite already covers this micro).
- `top2_sum_gate`: DeepSeek-V3-style group top-2-sum → group filter → expert topk + weight normalize + EP/TP mask + optional physical remap. Richer CF / shfl; still **indices+weights only**.

### 3.7 Gather/scatter APIs (summary)

| Op | Index | Payload | Direction |
|----|-------|---------|-----------|
| `expand_to_fused` | `token_topk_to_pos` | value | scatter token→fused |
| `expand_to_fused_with_sf` | same | **value + sf** | co-scatter |
| `reduce_fused` | `token_topk_to_pos` | value (+ optional `x_sf`, weights) | gather fused→token |
| `get_fused_mapping` | from `topk_idx` | builds maps + aligned pad | setup |

No generic `gather(sf)` / `scatter(sf)` primitives beyond these MoE-shaped kernels.

---

## 4. Side-by-side: TileKernels MoE pack vs ST suite (SV4 / SV8 / SV9)

| Axis | TileKernels MoE+quant | ST SimtVF suite today |
|------|----------------------|------------------------|
| Topk / kill CF | `topk_gate`, `top2_sum_gate` | **SV9** + **CF1–CF3** (and CF4–6 schedule/math) |
| Index gather of **values** | expand/reduce via `token_topk_to_pos` | **SV4** `W[Idx[i]]` + psum across B (`keep_idx`/`remat_idx`) |
| Group scale **compute** | `per_token_cast` / `per_block_cast` absmax→sf | **SV5–SV7** reduce-only; **SV8** reduce+bcast mul |
| Scale **layout** (live vs expand) | TMA col-major SF; packed UE8M0; sideband `(data,sf)` | **SV8** `live` vs `spill_dist` expand `[R,CG]→[R,C]` |
| **Co-move (v, sf) under index** | **`expand_to_fused_with_sf` / reduce×`x_sf`** | **Missing** |
| Packed UE8M0 / FP4 e2m1 | First-class in `quant/common.py` | **Missing** (fp16/fp32 only in ST) |
| Expert-major aligned pad | `get_fused_mapping` alignment | **Missing** (SV4 Idx is dense remapped, no pad slots) |
| SFA / KV cache | Not in repo | Not in suite; **structural gap** = packed gather/scatter ST |
| Target | NVIDIA SM90/SM100 TileLang | Ascend SimtVF + PTO VMI twins |

**Closest existing ST relatives:**

1. **SV4** ≈ value-only indexed gather + keep/remat of VCI — add a twin that also gathers/scatters an `sf` row.  
2. **SV8** ≈ local scale+value *compute/bcast* — not indexed movement of a pre-packed pair.  
3. **SV9** ≈ TileKernels `topk_gate` micro — already aligned; do **not** bolt MX onto SV9V (fidelity rule).

---

## 5. Gaps — what TileKernels does that the ST suite lacks

1. **Dual-tensor indexed scatter** of `(value, sf)` with shared `pos` and pad-zero of both.  
2. **Dual-tensor indexed gather** with optional per-pos SF × routing weight.  
3. **Packed UE8M0** load/store/decode macros (density + shift-to-fp32).  
4. **TMA / col-major SF** layout contrast vs row-major sideband (NPU analogy: VL-friendly scale expand vs compact SF).  
5. **Expert-aligned fused buffer** (pad to alignment; negative expert = hole) — stresses gather over sparse/padded slots.  
6. **FP4 value packing** (2×e2m1 / byte) co-scattered with SF — value packing orthogonal to SF packing.  
7. **End-to-end MoE pack pipeline micro:** topk idx → fused map → expand_with_sf (structure only; no full GEMM).

None of these require implementing full DeepSeek MoE; they need **structure benchmarks** that probe RF/Parallel/remat under packed layouts.

---

## 6. Proposed structure benchmarks (ready to implement later)

Naming: **SP** = Structure Pack (new family after SV/CF). Do **not** implement full kernels in this pass — design only. Tags follow suite convention.

### SP1 — Dual scatter `(V, Sf)` via index map (MoE expand_with_sf micro)

| Field | Spec |
|-------|------|
| **Name** | `sp1_dual_scatter_vsf` |
| **Packed layout** | Sideband: `V[T,H]`, `Sf[T,Hs]` → `Vexp[E,H]`, `Sfexp[E,Hs]` via `Pos[T,K]` (−1 skip); pad rows zero both |
| **Sensitivity** | Co-scatter bandwidth + KEEP `pos_local[K]` vs remat; fragment holding both V and Sf rows |
| **Knobs** | `T∈{32,128}`, `K∈{2,8}`, `H∈{128,512}`, `Hs=H/G` with `G∈{32,128}`, arm∈{`keep_pos`,`remat_pos`}, threads 32 |
| **Simt vs Simd** | High — dual live fragments vs shared staging; VMI twin must mirror Simt arms |
| **Tags** | `sp1_t{T}_k{K}_h{H}_g{G}_t{Thr}_{keep_pos,remat_pos}` |
| **TileKernels cite** | `expand_to_fused_with_sf` |

### SP2 — Dual gather + weighted reduce (MoE reduce_fused micro)

| Field | Spec |
|-------|------|
| **Name** | `sp2_dual_gather_wreduce` |
| **Packed layout** | Gather `Vexp[pos]` × `W[k]` × optional `Sf[pos]` into `Out[T,H]` |
| **Sensitivity** | Index remat across K; live weights vs remat; SF sideband multiply tax |
| **Knobs** | `T,K,H`, `with_sf∈{0,1}`, `with_w∈{0,1}`, arm∈{`keep_pos`,`remat_pos`} |
| **Simt vs Simd** | High — reduce fragment KEEP + gather VCI |
| **Tags** | `sp2_t{T}_k{K}_h{H}_t{Thr}_{sf0,sf1}_{w0,w1}_{keep_pos,remat_pos}` |
| **Cite** | `reduce_fused` |

### SP3 — Packed UE8M0 SF load/decode vs FP32 SF (MX scale pack)

| Field | Spec |
|-------|------|
| **Name** | `sp3_ue8m0_sf_pack` |
| **Packed layout** | Arm A: `Sf_fp32[M,K]`; Arm B: packed `Sf_u8` / `int32` view with `load→(e<<23)` decode; consumer `Out = V * transform_sf(Sf)` |
| **Sensitivity** | Pack density vs decode tax; bank/align of packed vs unpacked |
| **Knobs** | `M,K`, `pack∈{fp32,ue8m0}`, `G` (channels per SF) |
| **Simt vs Simd** | Medium — layout/decode; good VL overlay contrast |
| **Tags** | `sp3_m{M}_k{K}_g{G}_t{Thr}_{fp32,ue8m0}` |
| **Cite** | `quant/common.py` `load_sf`/`transform_sf`/`use_packed_ue8m0` |

### SP4 — Expert-major aligned pad gather (fused mapping consumer)

| Field | Spec |
|-------|------|
| **Name** | `sp4_aligned_pad_gather` |
| **Packed layout** | Dense `Buf[Eexp,H]` with `Expert[Eexp]` (−1 = pad hole); gather only `Expert>=0` slots into compacted or per-expert ranges (`start/end` aligned to `A∈{8,16}`) |
| **Sensitivity** | Predicated gather over pad; alignment waste vs dense SV4 |
| **Knobs** | `E,Eexp,H,A`, fill ratio / pad fraction |
| **Simt vs Simd** | Medium-high — predication + irregular live counts |
| **Tags** | `sp4_e{E}_exp{Eexp}_h{H}_a{A}_t{Thr}_pad{Pct}` |
| **Cite** | `get_fused_mapping` + expand pad zeros |

### SP5 — Scale+value **interleave vs sideband** (SFA-layout contrast)

| Field | Spec |
|-------|------|
| **Name** | `sp5_interleave_vs_sideband` |
| **Packed layout** | **Sideband:** `V[N,H]` + `Sf[N,Hs]` (TileKernels style). **Interleave:** synthetic `Pack[N, H+Hs]` or struct-of-arrays chunk `chunk=(v_tile, sf_scalar)` — Ascend SFA / KV-ish “one slot carries scale+value” |
| **Sensitivity** | Same gold math; measure RF/MTE when gather one slot vs two tensors; closest **SFA KV-cache layout analogy** |
| **Knobs** | `N,H,G`, layout∈{`sideband`,`interleave`}, gather stride / slot size |
| **Simt vs Simd** | **Highest interest** for SFA story — layout is the point |
| **Tags** | `sp5_n{N}_h{H}_g{G}_t{Thr}_{sideband,interleave}` |
| **Cite** | TileKernels sideband; ST SV8 spill_dist as weak cousin (expand scale, not interleave) |

### SP6 — FP4 value pack + SF co-apply (optional / later)

| Field | Spec |
|-------|------|
| **Name** | `sp6_fp4_unpack_sf` |
| **Packed layout** | `V_i8[N, H/2]` (2×e2m1) + `Sf[N,Hs]`; unpack lo/hi nibble → apply SF → fp16/fp32 out |
| **Sensitivity** | Value-pack decode + SF bcast; orthogonal to SP3 |
| **Knobs** | `N,H,G`, arm∈{`unpack_only`,`unpack_sf`} |
| **Simt vs Simd** | Medium — bit ops vs VL |
| **Tags** | `sp6_n{N}_h{H}_g{G}_t{Thr}_{unpack,unpack_sf}` |
| **Cite** | `unpack_from_e2m1fn_x2`, per_token_cast e2m1 |

### Suggested implementation order

1. **SP1 → SP2** (MoE expand/reduce structure; reuses SV4 index arms).  
2. **SP5** (explicit SFA layout contrast — answers the user ask most directly).  
3. **SP3** (MX UE8M0).  
4. **SP4** (aligned pad).  
5. **SP6** if FP4 on Ascend path becomes relevant.

Each SP needs a Simt baseline first, then VMI twin per `ST_VMI_TWINS.md` (no bolt-ons to SV9V).

---

## 7. Short code citations (anchors)

**Co-scatter (value + sf):**  
`tile_kernels/moe/expand_to_fused_kernel.py` — `get_expand_to_fused_kernel` loop `for k in T.serial(num_topk)` writes `expanded_x` and `expanded_x_sf` at `pos_local[k]`.

**Packed UE8M0:**  
`tile_kernels/quant/common.py` — `BaseCastConfig.use_packed_ue8m0`, `store_sf` / `load_sf` / `transform_sf`, `cast_epilogue` int32 view.

**QuantTensor:**  
`tile_kernels/quant/types.py` — `QuantTensor = tuple[torch.Tensor, torch.Tensor]`.

**Topk (SV9 cousin):**  
`tile_kernels/moe/topk_gate_kernel.py` — Parallel load + `reduce_max` + min-index reducer + kill −∞.

**ST counterparts:**  
`reports/ST_SV4_INDEX_GATHER_PSUM.md`, `ST_SV8_CASE3_BCAST.md`, `ST_SV9_TOPK_E2E.md`, `ST_LIST_DESIGN.md`.

---

## 8. Constraints / non-actions this pass

- No CF VMI work; no opsim on pto-b10.  
- No PR #272 push — survey lives on box only at this path.  
- No full SP kernel implementation (design + tag naming only).

---

## 9. Bottom line for suite planning

Treat TileKernels MoE MX path as a **reference for packed-structure STs**, not as code to port. The missing ST slice is: **indexed co-movement of scale+value (and denser SF/value packs)**, which is exactly the sensitivity needed before claiming SimtVF/Parallel readiness for **SFA gather/scatter KV-cache-style layouts**. SP1/SP2/SP5 are the highest-value next structure benchmarks.
