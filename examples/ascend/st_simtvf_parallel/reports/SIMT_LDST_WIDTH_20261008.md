# Simt `T.Parallel` load/store width audit — SV / CF / SP

**Date:** 2026-10-08 HKT · **SOC:** Ascend950PR_9599 opsim (pto-b10, edgexpert-59a6) · **No git push · PR #272 untouched.**

Companion to `VF_CYCLES_SIMT_VS_DSL_BEST_20261008.md`: that one counts VF (vector-pipe) cycles;
this one asks the other half of the question — **is the Simt side even moving UB data in wide
vector loads/stores, or is it crawling element-by-element?** UB bandwidth is high, so a kernel
stuck on narrow/scalar `LDS`/`STS` is ld/st-bound, not VF-bound, and the VF-cycle comparison
against SIMD (PTO-DSL) is then measuring the wrong bottleneck.

## What is actually measured

There is no `lds.64` / `lds.128` mnemonic in the Ascend trace. The opsim per-instruction CSV
(`OPPROF_*/simulator/core0.veccore0/core0.veccore0_instr_exe.csv`) carries the SIMT ld/st ops with
an explicit width field, which is the equivalent:

| op | pipe | meaning | width field |
|---|---|---|---|
| `SIMT_LDS` / `SIMT_STS` | `RVECLD` / `RVECST` | UB (shared) load / store | `[type:n]` / `[btype:n]` |
| `SIMT_LDK` / `SIMT_STK` | `RVECLD` / `RVECST` | **register-spill** load / store (stack) | same |

Width decode, validated against the generated CUDA-C for the same kernel (`sources/<tag>_source.txt`):

| field | bits | evidence |
|---|---|---|
| `type:0` | 8 b | SP6 reads packed fp4 as `__ubuf__ uint8_t` → `type:0` |
| `type:2` / `btype:1` | 16 b | SV5 `ub_stream` reads `__ubuf__ half` per element → `type:2`; SV2/SV8 `half` store → `btype:1` |
| `type:4` / `btype:2` | 32 b | SP4 `int32_t` predicate load, `float` store → 1 data reg |
| `type:5` / `btype:3` | 64 b | SP4 `*(__ubuf__ float2*)` load/store → 2 data regs (`Rd`,`Rd1` / `Rs`,`Rs1`) |
| `type:6` / `btype:4` | 128 b | **never seen** — a histogram of every `[type:]`/`[btype:]` on every `RVECLD`/`RVECST` row across all 49 Simt traces returns only `type:0/2/4/5` and `btype:1/2/3` |

So the widest UB access any Simt `T.Parallel` kernel emits here is **64-bit (`float2`)**. No kernel
reaches a 128-bit vector load/store, and nothing in the generated code ever uses `float4` / `uint4`.

- "LD / ST widths (dyn calls)" = Σ `call_count` per width class over the whole kernel.
- "ld+st cyc / Σ SIMT-pipe cyc" = ld/st share of all `RVEC*` pipe cycles (the SIMT instruction-stream
  view, *not* the VF_total metric — VF_total is shown in the VF-cycles report).
- `LDK`/`STK` counts are flagged separately: those are RF spill traffic, not algorithmic UB traffic.

## Per-case table

| case | shape / arm | payload access in generated code | LD widths (dyn calls) | ST widths (dyn calls) | widest ld/st seen | ld+st cyc / Σ SIMT-pipe cyc | verdict |
|---|---|---|---|---|---|---:|---|
| SV1 | E256 stream | `float2` (64b) | 64b×8 | 64b×4 | 64b | 358/483 (74%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SV1 | E2048 stream | `float2` (64b) | 64b×64 | 64b×32 | 64b | 2,905/4,046 (72%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SV2 | R16×C64 input_keep | — | — | — | — | — | COMPILE_FAIL — no kernel, no trace |
| SV2 | R16×C32 input_keep (ABI fallback) | scalar `half` (16b) | 16b×16 | 16b×16 | 16b | 752/1,461 (51%) | **NARROW — 16b element LDS/STS** |
| SV2 | R32×C64 input_stream | scalar `half` (16b) | 16b×127 | 16b×64 | 16b | 4,712/8,893 (53%) | **NARROW — 16b element LDS/STS** |
| SV2 | R32×C64 reload (scale foil) | scalar `half` (16b) | 16b×128 + 32b×32 | 16b×64 + 32b×32 | 32b | 6,241/10,687 (58%) | **NARROW — 16b element LDS/STS** |
| SV2 | R32×C128 input_stream | scalar `half` (16b) | 16b×255 + 32b×102 | 16b×128 + 32b×102 | 32b | 15,676/24,017 (65%) | **NARROW — 16b element LDS/STS**; RF spill 102×LDK / 102×STK |
| SV3 | M24 VL64 K16 keep | scalar `float` A + scalar `half` x; `float2` store | 16b×16 + 32b×384 | 64b×24 | 64b | 11,441/24,814 (46%) | **NARROW — A rows read 32b element-at-a-time (LDS.32), x fp16 LDS.16; only the Out store is 64b** |
| SV3 | M32 VL64 K16 keep | scalar `float` A + scalar `half` x; `float2` store | 16b×16 + 32b×512 | 64b×32 | 64b | 15,212/32,793 (46%) | **NARROW — A rows read 32b element-at-a-time (LDS.32), x fp16 LDS.16; only the Out store is 64b** |
| SV3 | M32 VL64 K16 split_cm16 | scalar `float` A + scalar `half` x; `float2` store | 16b×32 + 32b×512 | 64b×32 | 64b | 15,560/34,259 (45%) | **NARROW — A rows read 32b element-at-a-time (LDS.32), x fp16 LDS.16; only the Out store is 64b** |
| SV4 | E256 B8 keep_idx | `float2` (64b) | 32b×72 + 64b×4 | 64b×4 | 64b | 2,238/4,977 (45%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SV4 | E256 B8 remat_idx_calc | `float2` (64b) | 32b×72 | 64b×4 | 64b | 2,118/4,741 (45%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SV5 | R64 C128 G16 keep_in_rf | `uint2` (64b) + scalar `half` | 32b×322 | 32b×210 | 32b | 33,003/37,178 (89%) | **NARROW — 64b `uint2` in source, only 32b LDS in trace**; RF spill 194×LDK / 194×STK |
| SV5 | R64 C128 G16 ub_stream | scalar `half` (16b) | 16b×256 | 32b×16 | 32b | 18,006/24,529 (73%) | **NARROW — 16b element LDS/STS** |
| SV6 | R64 C128 G32 keep_in_warp | `uint2` (64b) + scalar `half` | 32b×384 | 32b×264 | 32b | 25,136/36,580 (69%) | **NARROW — 64b `uint2` in source, only 32b LDS in trace**; RF spill 256×LDK / 256×STK |
| SV6 | R64 C128 G32 reload | scalar `half` (16b) | 16b×256 | 32b×8 | 32b | 10,912/24,748 (44%) | **NARROW — 16b element LDS/STS** |
| SV7 | R64 C128 G64 multiwarp_ub | scalar `half` (16b) | 16b×256 + 32b×16 | 32b×20 | 32b | 70,208/77,041 (91%) | **NARROW — 16b element LDS/STS** |
| SV7 | R64 C128 G64 reload | scalar `half` (16b) | 16b×256 | 32b×4 | 32b | 14,928/28,576 (52%) | **NARROW — 16b element LDS/STS** |
| SV7 | R64 C128 G128 multiwarp_ub | scalar `half` (16b) | 16b×256 + 32b×4 | 32b×10 | 32b | 15,201/30,008 (51%) | **NARROW — 16b element LDS/STS** |
| SV7 | R64 C128 G128 reload | scalar `half` (16b) | 16b×256 | 32b×2 | 32b | 14,888/28,490 (52%) | **NARROW — 16b element LDS/STS** |
| SV8 | R64 C128 G16 live | scalar `half` (16b) | 16b×512 + 32b×17 | 16b×256 + 32b×17 | 32b | 51,816/76,061 (68%) | **NARROW — 16b element LDS/STS**; RF spill 17×LDK / 17×STK |
| SV8 | R64 C128 G16 spill_dist | scalar `half` (16b) | 16b×512 + 32b×406 | 16b×256 + 32b×406 | 32b | 86,463/106,927 (81%) | **NARROW — 16b element LDS/STS**; RF spill 150×LDK / 150×STK |
| SV9 | E256 K8 keep | `float2` (64b) | 64b×4 | 32b×8 | 64b | 290/3,844 (8%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SV9 | E256 K8 remat_scores | `float2` (64b) | 64b×32 | 32b×72 | 64b | 1,992/5,030 (40%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SV9 | E256 K8 remat_idx | `float2` (64b) | 64b×4 | 32b×8 | 64b | 289/3,843 (8%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF1 | E256 K1 keep | `float2` (64b) | 32b×1 + 64b×4 | 64b×4 | 64b | 264/444 (59%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF1 | E256 K8 keep | `float2` (64b) | 32b×8 + 64b×4 | 64b×4 | 64b | 477/1,473 (32%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF2 | E256 K8 remat + shared-kill | `float2` (64b) | 32b×8 + 64b×36 | 32b×64 + 64b×4 | 64b | 2,260/3,055 (74%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF3 | E256 K8 remat_idx | `float2` (64b) | 32b×8 + 64b×4 | 64b×4 | 64b | 478/1,579 (30%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF4 | E256 pfat5 nested_if | `float2` (64b) | 32b×10 + 64b×4 | 64b×4 | 64b | 493/1,068 (46%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF4 | E256 pfat25 nested_if | `float2` (64b) | 32b×10 + 64b×4 | 64b×4 | 64b | 515/1,090 (47%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF5 | E256 pnear5 div_ulp | `float2` (64b) | 32b×2 + 64b×7 | 64b×4 | 64b | 405/734 (55%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF5 | E256 pnear25 div_ulp | `float2` (64b) | 32b×2 + 64b×7 | 64b×4 | 64b | 388/717 (54%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF6 | E256 pnear5 newton | `float2` (64b) | 64b×4 | 64b×4 | 64b | 224/1,086 (21%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| CF6 | E256 pnear25 newton | `float2` (64b) | 64b×4 | 64b×4 | 64b | 224/1,086 (21%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP1 | T32 K2 H128 G32 keep_pos | `float2` (64b) | 32b×292 + 64b×64 | 32b×127 + 64b×132 | 64b | 15,724/30,092 (52%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP1 | T32 K2 H128 G32 remat_pos | `float2` (64b) | 32b×228 + 64b×64 | 32b×63 + 64b×132 | 64b | 12,748/26,315 (48%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP2 | sf0_w0 + keep_acc | `float2` (64b) | 32b×128 + 64b×126 | 32b×64 + 64b×64 | 64b | 10,116/19,814 (51%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP2 | sf0_w0 + remat_acc | `float2` (64b) | 32b×128 + 64b×316 | 32b×64 + 64b×254 | 64b | 20,722/30,701 (67%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP2 | sf1_w1 + keep_acc | `float2` (64b) | 32b×199 + 64b×252 | 32b×64 + 64b×192 | 64b | 60,926/76,895 (79%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP2 | sf1_w1 + remat_acc | `float2` (64b) | 32b×199 + 64b×442 | 32b×64 + 64b×382 | 64b | 71,387/87,826 (81%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP2 | sf1_w1 + keep_pos (legacy) | `float2` (64b) | 32b×199 + 64b×252 | 32b×64 + 64b×192 | 64b | 61,255/77,224 (79%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP2 | sf1_w1 + remat_pos (legacy) | `float2` (64b) | 32b×135 + 64b×252 | 64b×192 | 64b | 57,918/73,087 (79%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP3 | M32 H128 G32 fp32 | `float2` (64b) | 32b×4 + 64b×64 | 64b×64 | 64b | 12,996/17,647 (74%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP3 | M32 H128 G32 e8m0 | `float2` (64b) | 8b×4 + 64b×64 | 64b×64 | 64b | 12,991/17,670 (74%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP4 | E64 H128 pad25 | `float2` (64b) | 32b×128 + 64b×96 | 64b×128 | 64b | 9,280/15,717 (59%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP5 | N64 H128 G32 sideband | `float2` (64b) | 32b×64 + 64b×512 | 64b×512 | 64b | 32,960/85,243 (39%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP5 | N64 H128 G32 interleave | `float2` (64b) | 32b×64 + 64b×512 | 64b×512 | 64b | 32,960/86,164 (38%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |
| SP6 | N32 H128 G32 unpack | scalar `float` (32b) | 8b×64 + 32b×128 | 32b×128 | 32b | 7,872/11,532 (68%) | **NARROW — 8b LDS on the packed fp4 bytes** |
| SP6 | N32 H128 G32 unpack_sf | `float2` (64b) | 8b×64 + 32b×132 + 64b×64 | 32b×128 + 64b×64 | 64b | 21,012/28,520 (74%) | wide — 64b `float2` LDS.64/STS.64 on the data plane |

## Narrow / scalar-bound cases (the flags)

1. **Every fp16-input case reads UB one `half` at a time (`LDS.16`)** — SV2 (all arms), SV5 `ub_stream`,
   SV6 `reload`, SV7 both arms at both G, SV8 both arms. 256–512 `LDS.16` calls per kernel where 4–8×
   fewer 64-bit loads would carry the same bytes. SV8 also stores fp16 one element at a time (`STS.16` ×256).
   This is the single biggest ld/st finding: the whole Case-3 reduce ladder (SV5–SV8) and the whole SV2
   RF/broadcast family are on 16-bit UB accesses.
2. **SV3 (GEMV) reads A 32 bits at a time** — `LDS.32` ×512 per kernel (×384 at M24), plus `LDS.16` on the
   fp16 `x`. Only the `Out` store is 64-bit. Same shape in all three arms (keep / split_cm16).
3. **SV5 `keep_in_rf` and SV6 `keep_in_warp`**: the generated code *does* emit a 64-bit `uint2` UB fetch,
   but the trace contains **no 64-bit load at all** — only `LDS.32`, plus heavy spill (`194`/`256` LDK+STK
   pairs). The intended wide fetch does not survive to the instruction stream in the KEEP arms.
4. **SP6 `unpack`**: packed fp4 bytes are read with `LDS.8` ×64 instead of one 32-bit word + shifts.
5. **RF spill traffic is visible as `LDK`/`STK`**: SV6 `keep_in_warp` 256+256, SV5 `keep_in_rf` 194+194,
   SV8 `spill_dist` 150+150, SV2 R32×C128 102+102, SV8 `live` 17+17. Everywhere else: none. This is
   direct instruction-level evidence for the RF-ladder / remat-cliff story, independent of VF cycles.

**Clean (wide, 64-bit `float2`) on the data plane:** SV1, SV4, SV9, all of CF1–CF6, SP1–SP5 and
SP6 `unpack_sf`. These are the f32 cases; their 32-bit loads are index/predicate scalars, not payload.
Even these never exceed 64 bits, so there is still a 2× headroom to a 128-bit UB access if the
backend ever emits one.

**Consequence for the deck:** in the narrow cases the ld/st stream consumes 44–91 % of all SIMT-pipe
cycles (SV7 G64 `multiwarp_ub` 91 %, SV5 `keep_in_rf` 89 %, SV8 `spill_dist` 81 %), so a Simt-vs-SIMD
gap there is at least as much an ld/st-width gap as a vector-ALU gap.

## Provenance / caveats

- **SP1–SP6, SV4, SV5–SV7:** scanned from the surviving 2026-10-07 / 10-06 / 10-05 opsim CSVs on pto-b10
  (`/tmp/st_simtvf_parallel/opsim_sp*`, `/tmp/pr272_sv567_rfladder_20261006/`, `/tmp/pr272_sv4_cliff_20261005/out/`)
  — the same CSVs the VF-cycle report used.
- **SV1, SV2, SV3, SV8, SV9, CF1–CF6:** their old opsim dirs were gone from pto-b10, so the kernels were
  **re-compiled and re-run on 2026-10-08** into `/tmp/ldst_20261008/out` (24/24 opsim PASS; `sv2_r16_c64_t32_input_keep`
  COMPILE_FAIL as before — `layout_inference: no available layout found`). Walls reproduce the 10-07 values for
  SV1/SV2/SV8/SV9; SV3 and CF came out slightly different on this re-run (e.g. CF1 K8 wall 1.23 vs 0.97 µs,
  SV3 M24 VF 6,729 vs 9,105), so **do not** mix these VF numbers into the VF-cycles table — the ld/st widths,
  which are a property of the generated code, are what this report is for.
- Counts are per core (`core0.veccore0`, single-core kernels) and are dynamic `call_count`, i.e. they already
  include loop trip counts; predicated-off lanes are not counted as separate ops.
- Nothing was pushed; PR #272 was not touched.

## Artifacts

- Scanner: `reports/ldst_20261008/ldst_scan2.py` (remote copy `/tmp/ldst_scan2_20261008.py`)
- Raw scans: `reports/ldst_20261008/ldst_scan2_a.jsonl` (surviving dirs), `ldst_scan2_b.jsonl` (10-08 re-run)
- Re-run driver: `reports/ldst_20261008/run_ldst_missing.sh` → remote `/tmp/ldst_20261008/` (`out/`, `run.log`, `out/SUMMARY_ldst_raw.txt`)
- Static source widths: `reports/ldst_20261008/static_src.txt`; table generator `gen_md.py` → `table.md`
