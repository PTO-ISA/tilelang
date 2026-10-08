# Adaptations for PTO-ISA/tilelang `pto-dev`

Source tip ported: Wenbo ST `3828bbcf` (`examples/ascend/st_simtvf_parallel/` tree copy).
Base: PTO-ISA `pto-dev` @ `3d70ede9`.

## Why tree-copy (not rebase)

Wenbo `pto-dev` and PTO-ISA `pto-dev` have diverged substantially. Suite is
examples-only; copying the tree onto PTO-ISA tip avoids dragging Wenbo src diffs.

## API adaptations (examples only)

1. **Language dialect import** — all Simt kernels / harness helpers that used
   `import tilelang.language as T` now use `import tilelang.ascend.language as T`.
   On PTO-ISA, `T.SimtVF` / Ascend `T.Kernel` live under the Ascend dialect
   (see `examples/ascend/example_simtvf_multi_kernel.py`).

2. **`common_asc_harness.patch_simtvf_ffi`** — no longer imports
   `tilelang.language.vf` (missing on PTO-ISA). Uses stock
   `tilelang.ascend.language.SimtVF` (already 3-arg FFI with `source_index`).

3. **Build / DEPS** — oneshots accept `TILELANG_DEPS` override; regression
   points at this checkout's `build/lib/libtilelang.so` instead of the Sep-18
   Wenbo deps-stack lib.

## Not adapted

- PTO-DSL `kernels_ptodsl/*d` stay on `ptodsl` / `sim_dsl.sh` (independent of
  TileLang Ascend dialect imports).
- No broad `src/` compiler patches.

## Opsim / CANN blocker (pto-b10)

Built `libtilelang.so` from this PTO-ISA checkout (Ascend ON, Release).
Simt codegen on tip emits MicroAPI DMA surface not present in available
toolkits on pto-b10 (`cann_91b3` / `cann_92b1`):

- `asc_copy_gm2ub_align` + `asc_load_l2_cache_mode` / `asc_store_l2_cache_mode`
- template helpers `asc_pack_to_high` / `asc_pack_to_low` / `asc_storealign_pack_quarter`

Green Wenbo ST runs on the same host used AscendC `copy_gm_to_ubuf_align_v2`
(deps-stack / Wenbo tip). No examples-only shim can rewrite codegen enums.
**No broad src/compiler rollback applied.** Simt SV/CF/SP tags are recorded as
`COMPILE_FAIL` against this host CANN. SIMD (PTO-DSL) `*d` path uses
`ptodsl`/`sim_dsl.sh` and is independent of TileLang Ascend codegen.
