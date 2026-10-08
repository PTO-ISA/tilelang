# SV5–SV7 (+VMI twins) — reduce-group RF-capacity ladder

OUT: `/tmp/pr272_sv567_rfladder_20261006`

Primary sensitivity: **group-input RF capacity during absmax** (not post-reduce scale residency).

| rung | arms |
|------|------|
| sv5 G=16 | `keep_in_rf` (inputs preloaded into fragment/RF) vs `ub_stream` (stream from UB) |
| sv6 G=32 | `keep_in_warp` (one-warp RF keep, no UB partials) vs `reload` (stream from UB) |
| sv7 G=64/128 | `multiwarp_ub` (NW=4 chunk partials via UB allreduce) vs `reload` (single stream) |

## All tags

| tag | PASS | wall µs | IPC_proxy | VF µs | body_instr | maxabs |
|-----|------|--------:|----------:|------:|-----------:|--------|
| sv5_r64_c128_g16_t32_keep_in_rf | PASS | 7.04 | 0.110 | 5.91 | 1169 | 1.1920928955078125e-07 |
| sv5_r64_c128_g16_t32_ub_stream | PASS | 4.18 | 0.220 | 3.06 | 1211 | 1.1920928955078125e-07 |
| sv5v_r64_c128_g16_t64_ub_stream | PASS | 109.39 | - | - | 111194 | 0.0 |
| sv6_r64_c128_g32_t32_keep_in_warp | PASS | 14.83 | 0.092 | 13.72 | 2280 | 5.960464477539063e-08 |
| sv6_r64_c128_g32_t32_reload | PASS | 12.58 | 0.107 | 11.53 | 2219 | 5.960464477539063e-08 |
| sv6v_r64_c128_g32_t64_reload | PASS | 101.22 | - | - | 105654 | 0.0 |
| sv7_r64_c128_g64_t32_multiwarp_ub | PASS | 7.32 | 0.114 | 6.21 | 1280 | 2.9802322387695312e-08 |
| sv7_r64_c128_g64_t32_reload | PASS | 14.54 | 0.090 | 13.52 | 2187 | 2.9802322387695312e-08 |
| sv7_r64_c128_g128_t32_multiwarp_ub | PASS | 15.42 | 0.089 | 14.42 | 2317 | 2.9802322387695312e-08 |
| sv7_r64_c128_g128_t32_reload | PASS | 14.58 | 0.089 | 13.57 | 2171 | 2.9802322387695312e-08 |
| sv7v_r64_c128_g64_t64_multiwarp_ub | PASS | 97.16 | - | - | 102971 | 0.0 |
| sv7v_r64_c128_g64_t64_reload | PASS | 97.16 | - | - | 102902 | 0.0 |
| sv7v_r64_c128_g128_t64_multiwarp_ub | PASS | 95.44 | - | - | 101594 | 0.0 |
| sv7v_r64_c128_g128_t64_reload | PASS | 95.43 | - | - | 101548 | 0.0 |

## Winner per rung / shape

| family | R | C | G | CG | T | keep arm | keep µs | foil arm | foil µs | Δ µs | Δ % | winner |
|--------|--:|--:|--:|---:|--:|----------|--------:|----------|--------:|-----:|----:|--------|
| sv5 | 64 | 128 | 16 | 8 | 32 | keep_in_rf | 7.04 | ub_stream | 4.18 | -2.86 | -40.6% | **ub_stream** |
| sv5v | 64 | 128 | 16 | 8 | 64 | keep_in_rf | - | ub_stream | 109.39 | - | - | incomplete |
| sv6 | 64 | 128 | 32 | 4 | 32 | keep_in_warp | 14.83 | reload | 12.58 | -2.25 | -15.2% | **reload** |
| sv6v | 64 | 128 | 32 | 4 | 64 | keep_in_warp | - | reload | 101.22 | - | - | incomplete |
| sv7 | 64 | 128 | 64 | 2 | 32 | multiwarp_ub | 7.32 | reload | 14.54 | +7.22 | +98.6% | **multiwarp_ub** |
| sv7 | 64 | 128 | 128 | 1 | 32 | multiwarp_ub | 15.42 | reload | 14.58 | -0.84 | -5.4% | **reload** |
| sv7v | 64 | 128 | 64 | 2 | 64 | multiwarp_ub | 97.16 | reload | 97.16 | +0.00 | +0.0% | **tie** |
| sv7v | 64 | 128 | 128 | 1 | 64 | multiwarp_ub | 95.44 | reload | 95.43 | -0.01 | -0.0% | **reload** |

## VF cycle theory vs measured

`C_pred = O_arm + ceil((W/T) / I_assum)` with I_assum(Simt)=0.2, I_assum(VMI)=0.85, O_base=300, O_loop=250, O_ub_roundtrip=400, freq=1.80 GHz. Assumptions and the a5/a6 re-use plan: `reports/ST_VF_THEORY.md`.

| tag | arm | W | W_loop | W/T | I_assum | O | C_pred | µs_pred | C_meas | meas_src | ratio | IPC_meas | verdict |
|-----|-----|--:|------:|----:|--------:|--:|-------:|--------:|-------:|----------|------:|---------:|---------|
| sv5_r64_c128_g16_t32_keep_in_rf | keep_in_rf | 42496 | 0 | 1328 | 0.20 | 550 | 7190.0 | 3.99 | 10641 | VF_SIMT | 1.48 | 0.109 | OK |
| sv5_r64_c128_g16_t32_ub_stream | ub_stream | 34304 | 0 | 1072 | 0.20 | 300 | 5660.0 | 3.14 | 5506 | VF_SIMT | 0.97 | 0.218 | OK |
| sv5v_r64_c128_g16_t64_ub_stream | ub_stream | 34304 | 0 | 536 | 0.85 | 300 | 931.0 | 0.52 | 11817 | RVEC pipes | 12.69 | 0.126 | OUTLIER_SLOW |
| sv6_r64_c128_g32_t32_keep_in_warp | keep_in_warp | 107264 | 65536 | 3352 | 0.10 | 550 | 34070.0 | 18.93 | 24697 | VF_SIMT | 0.72 | 0.092 | OK |
| sv6_r64_c128_g32_t32_reload | reload | 66304 | 32768 | 2072 | 0.10 | 300 | 21020.0 | 11.68 | 20748 | VF_SIMT | 0.99 | 0.106 | OK |
| sv6v_r64_c128_g32_t64_reload | reload | 33536 | 0 | 524 | 0.85 | 300 | 917.0 | 0.51 | 11397 | RVEC pipes | 12.43 | 0.136 | OUTLIER_SLOW |
| sv7_r64_c128_g64_t32_multiwarp_ub | multiwarp_ub | 34688 | 0 | 1084 | 0.20 | 950 | 6370.0 | 3.54 | 11182 | VF_SIMT | 1.76 | 0.114 | OK |
| sv7_r64_c128_g64_t32_reload | reload | 65920 | 32768 | 2060 | 0.10 | 300 | 20900.0 | 11.61 | 24339 | VF_SIMT | 1.16 | 0.089 | OK |
| sv7_r64_c128_g128_t32_multiwarp_ub | multiwarp_ub | 66496 | 32768 | 2078 | 0.10 | 950 | 21730.0 | 12.07 | 25948 | VF_SIMT | 1.19 | 0.089 | OK |
| sv7_r64_c128_g128_t32_reload | reload | 65728 | 32768 | 2054 | 0.10 | 300 | 20840.0 | 11.58 | 24423 | VF_SIMT | 1.17 | 0.088 | OK |
| sv7v_r64_c128_g64_t64_multiwarp_ub | multiwarp_ub | 34688 | 0 | 542 | 0.85 | 950 | 1588.0 | 0.88 | 13570 | RVEC pipes | 8.55 | 0.109 | OUTLIER_SLOW |
| sv7v_r64_c128_g64_t64_reload | reload | 33152 | 0 | 518 | 0.85 | 300 | 910.0 | 0.51 | 13457 | RVEC pipes | 14.79 | 0.105 | OUTLIER_SLOW |
| sv7v_r64_c128_g128_t64_multiwarp_ub | multiwarp_ub | 33728 | 0 | 527 | 0.85 | 950 | 1570.0 | 0.87 | 12228 | RVEC pipes | 7.79 | 0.119 | OUTLIER_SLOW |
| sv7v_r64_c128_g128_t64_reload | reload | 32960 | 0 | 515 | 0.85 | 300 | 906.0 | 0.50 | 12782 | RVEC pipes | 14.11 | 0.111 | OUTLIER_SLOW |

Implied camodel frequency (VF cycles / VF running_time): 1.799-6.947 GHz.

## Top opcodes

- `sv5_r64_c128_g16_t32_keep_in_rf`: SIMT_FMNMX:240, SIMT_STK:194, SIMT_LDK:194, SIMT_LDS:128, SIMT_F2F:128, SIMT_IADD_I:113, SIMT_FMNMX_I:32, MOV_XD_IMM:23
- `sv5_r64_c128_g16_t32_ub_stream`: SIMT_LDS:256, SIMT_F2F:256, SIMT_IADD_I:241, SIMT_FMNMX:240, SIMT_FMNMX_I:32, SIMT_IMAD_I:31, MOV_XD_IMM:23, SIMT_STS:16
- `sv5v_r64_c128_g16_t64_ub_stream`: ADD_IMM:27133, CMP:26639, JUMPC:26639, ST_XD_XN_IMM:8704, LD_XD_XN_IMM:8704, OR:8192, JUMP:1028, MOV_XD_XN:1024
- `sv6_r64_c128_g32_t32_keep_in_warp`: SIMT_IADD_I:329, SIMT_LDK:256, SIMT_LEA:256, SIMT_FMNMX:256, SIMT_ISETP_I:256, SIMT_BRANCH:256, SIMT_STK:256, SIMT_LDS:128
- `sv6_r64_c128_g32_t32_reload`: SIMT_IADD_I:272, SIMT_LEA:257, SIMT_LDS:256, SIMT_IADD:256, SIMT_F2F:256, SIMT_FMNMX:256, SIMT_ISETP_I:256, SIMT_BRANCH:256
- `sv6v_r64_c128_g32_t64_reload`: ADD_IMM:25853, CMP:25631, JUMPC:25631, ST_XD_XN_IMM:8448, LD_XD_XN_IMM:8448, OR:8192, JUMP:516, MOV_XD_XN:512
- `sv7_r64_c128_g128_t32_multiwarp_ub`: SIMT_IADD_I:274, SIMT_FMNMX:262, SIMT_LDS:260, SIMT_LEA:257, SIMT_F2F:256, SIMT_ISETP_I:256, SIMT_BRANCH:256, SIMT_IADD_X:192
- `sv7_r64_c128_g128_t32_reload`: SIMT_IADD_I:260, SIMT_LEA:257, SIMT_LDS:256, SIMT_IADD:256, SIMT_F2F:256, SIMT_ISETP_I:256, SIMT_FMNMX:256, SIMT_BRANCH:256
- `sv7_r64_c128_g64_t32_multiwarp_ub`: SIMT_LDS:272, SIMT_F2F:256, SIMT_FMNMX:252, SIMT_IADD_I:232, SIMT_LEA:50, SIMT_FMNMX_I:24, MOV_XD_IMM:23, SIMT_STS:20
- `sv7_r64_c128_g64_t32_reload`: SIMT_IADD_I:264, SIMT_LEA:257, SIMT_LDS:256, SIMT_IADD:256, SIMT_F2F:256, SIMT_ISETP_I:256, SIMT_FMNMX:256, SIMT_BRANCH:256
- `sv7v_r64_c128_g128_t64_multiwarp_ub`: CMP:24959, JUMPC:24959, ADD_IMM:24893, ST_XD_XN_IMM:8256, LD_XD_XN_IMM:8256, OR:8192, RV_VSEL:260, RV_VCMP_GT:260
- `sv7v_r64_c128_g128_t64_reload`: CMP:24959, JUMPC:24959, ADD_IMM:24893, ST_XD_XN_IMM:8256, LD_XD_XN_IMM:8256, OR:8192, RV_VLDI:384, RV_VCMP_GT:256
- `sv7v_r64_c128_g64_t64_multiwarp_ub`: ADD_IMM:25213, CMP:25151, JUMPC:25151, ST_XD_XN_IMM:8320, LD_XD_XN_IMM:8320, OR:8192, RV_VLD:392, RV_VST:264
- `sv7v_r64_c128_g64_t64_reload`: ADD_IMM:25213, CMP:25151, JUMPC:25151, ST_XD_XN_IMM:8320, LD_XD_XN_IMM:8320, OR:8192, JUMP:260, RV_VLD:256

## VMI addendum (2026-10-06 ~17:48 HKT)

Oneshot `oneshot_sv567v_rfladder.sh` finished on pto-b10 (`DONE_SV567V_RFLADDER`).
Work dir: `/tmp/pr272_sv567_rfladder_20261006`. **No push.**

### Takeaways

1. **Wall clock is dominated by the scalar transpose prologue** forced by the
   Case-3 VMI ABI (lane-continuous xt[t,n] outside SimdVF; see `reports/ST_CASE3_VMI.md`).
   Wall µs sit ~95–109 µs for every PASS arm; VF-body RVEC work is only a few µs
   (`C_meas` / implied freq in the theory table).
2. **All reload / stream / multiwarp_ub arms PASS** (maxabs 0.0):
   `sv5v ub_stream`, `sv6v reload`, `sv7v g64/g128 reload`, `sv7v g64/g128 multiwarp_ub`.
3. **Keep-fragment arms still blocked** at LayoutInference:
   `sv5v keep_in_rf` and `sv6v keep_in_warp` →
   `The layout for fragment xf can not be inferred correctly`.
4. **VMI C_meas = RVEC pipe sum** (`RVECEX`/`RVECLD`/`RVECST`/`RVECSU`), not a
   `VF_*` row — see `reports/ST_VF_THEORY.md`. All six PASS VMI arms currently flag
   `OUTLIER_SLOW` vs I_assum=0.85 (ratios ~7.8–14.8); model needs a VMI-side recal
   once keep arms unblocked.

### Full VMI tag table

| tag | status | wall µs | C_meas (RVEC) | maxabs | note |
|-----|--------|--------:|--------------:|--------|------|
| sv5v_r64_c128_g16_t64_ub_stream | **PASS** | 109.39 | 11817 | 0.0 | stream foil |
| sv5v_r64_c128_g16_t64_keep_in_rf | **COMPILE_FAIL** | — | — | — | fragment xf layout |
| sv6v_r64_c128_g32_t64_reload | **PASS** | 101.22 | 11397 | 0.0 | reload foil |
| sv6v_r64_c128_g32_t64_keep_in_warp | **COMPILE_FAIL** | — | — | — | fragment xf layout |
| sv7v_r64_c128_g128_t64_reload | **PASS** | 95.43 | 12782 | 0.0 | |
| sv7v_r64_c128_g128_t64_multiwarp_ub | **PASS** | 95.44 | 12228 | 0.0 | ≈tie with reload |
| sv7v_r64_c128_g64_t64_reload | **PASS** | 97.16 | 13457 | 0.0 | |
| sv7v_r64_c128_g64_t64_multiwarp_ub | **PASS** | 97.16 | 13570 | 0.0 | tie with reload |

sv7v winners on wall µs: G=64 **tie**, G=128 **reload** by 0.01 µs (noise).
Unlike Simt, multiwarp_ub does **not** win G=64 on VMI wall — scalar transpose
swamps the VF-body difference the Simt ladder measures.
