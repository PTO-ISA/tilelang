# STATUS — Port onto PTO-ISA/tilelang pto-dev

**Date:** 2026-10-08 (UTC+8)
**Base:** PTO-ISA pto-dev @ 3d70ede9
**Source suite tip:** Wenbo ST 3828bbcf (tree-copy of examples/ascend/st_simtvf_parallel/)
**Branch:** feat/st-simtvf-parallel-sv1-sv9 on fork peanutchan/tilelang

## Done
- Suite tree-copied; Ascend dialect import + harness FFI adaptations (see ADAPTATIONS_PTO_ISA.md).
- Built this checkout Ascend libtilelang.so on pto-b10.

## Opsim blocker (pto-b10)
Simt compile against tip fails on available CANN (cann_91b3 / cann_92b1) missing MicroAPI symbols:
asc_copy_gm2ub_align, asc_load_l2_cache_mode, asc_store_l2_cache_mode,
asc_pack_to_high, asc_pack_to_low, asc_storealign_pack_quarter.

Representative Simt probes (SV1/SV2/SV3, CF1/CF6, SP1/SP2/SP3/SP6) -> COMPILE_FAIL.
PASS_TABLE_* / SUMMARY numbers in-tree are carried from Wenbo tip 3828bbcf and need
re-validation on a toolkit that provides those APIs. Do not treat them as PTO-ISA tip results.

SIMD (PTO-DSL) *d path is independent of TileLang Ascend codegen; partial *d opsim was
green in the port session but full refreshed PASS tables are deferred until Simt can
compile on a matching CANN.