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
| sv6_r64_c128_g32_t32_keep_in_warp | PASS | 14.83 | 0.092 | 13.72 | 2280 | 5.960464477539063e-08 |
| sv6_r64_c128_g32_t32_reload | PASS | 12.58 | 0.107 | 11.53 | 2219 | 5.960464477539063e-08 |
| sv7_r64_c128_g64_t32_multiwarp_ub | PASS | 7.32 | 0.114 | 6.21 | 1280 | 2.9802322387695312e-08 |
| sv7_r64_c128_g64_t32_reload | PASS | 14.54 | 0.090 | 13.52 | 2187 | 2.9802322387695312e-08 |
| sv7_r64_c128_g128_t32_multiwarp_ub | PASS | 15.42 | 0.089 | 14.42 | 2317 | 2.9802322387695312e-08 |
| sv7_r64_c128_g128_t32_reload | PASS | 14.58 | 0.089 | 13.57 | 2171 | 2.9802322387695312e-08 |

## Winner per rung / shape

| family | R | C | G | CG | T | keep arm | keep µs | foil arm | foil µs | Δ µs | Δ % | winner |
|--------|--:|--:|--:|---:|--:|----------|--------:|----------|--------:|-----:|----:|--------|
| sv5 | 64 | 128 | 16 | 8 | 32 | keep_in_rf | 7.04 | ub_stream | 4.18 | -2.86 | -40.6% | **ub_stream** |
| sv6 | 64 | 128 | 32 | 4 | 32 | keep_in_warp | 14.83 | reload | 12.58 | -2.25 | -15.2% | **reload** |
| sv7 | 64 | 128 | 64 | 2 | 32 | multiwarp_ub | 7.32 | reload | 14.54 | +7.22 | +98.6% | **multiwarp_ub** |
| sv7 | 64 | 128 | 128 | 1 | 32 | multiwarp_ub | 15.42 | reload | 14.58 | -0.84 | -5.4% | **reload** |

## VF cycle theory vs measured

`C_pred = O_arm + ceil((W/T) / I_assum)` with I_assum(Simt)=0.2, I_assum(VMI)=0.85, O_base=300, O_loop=250, O_ub_roundtrip=400, freq=1.80 GHz. Assumptions and the a5/a6 re-use plan: `reports/ST_VF_THEORY.md`.

| tag | arm | W | W_loop | W/T | I_assum | O | C_pred | µs_pred | C_meas | ratio | IPC_meas | verdict |
|-----|-----|--:|------:|----:|--------:|--:|-------:|--------:|-------:|------:|---------:|---------|
| sv5_r64_c128_g16_t32_keep_in_rf | keep_in_rf | 42496 | 0 | 1328 | 0.20 | 550 | 7190.0 | 3.99 | 10641 | 1.48 | 0.109 | OK |
| sv5_r64_c128_g16_t32_ub_stream | ub_stream | 34304 | 0 | 1072 | 0.20 | 300 | 5660.0 | 3.14 | 5506 | 0.97 | 0.218 | OK |
| sv6_r64_c128_g32_t32_keep_in_warp | keep_in_warp | 107264 | 65536 | 3352 | 0.10 | 550 | 34070.0 | 18.93 | 24697 | 0.72 | 0.092 | OK |
| sv6_r64_c128_g32_t32_reload | reload | 66304 | 32768 | 2072 | 0.10 | 300 | 21020.0 | 11.68 | 20748 | 0.99 | 0.106 | OK |
| sv7_r64_c128_g64_t32_multiwarp_ub | multiwarp_ub | 34688 | 0 | 1084 | 0.20 | 950 | 6370.0 | 3.54 | 11182 | 1.76 | 0.114 | OK |
| sv7_r64_c128_g64_t32_reload | reload | 65920 | 32768 | 2060 | 0.10 | 300 | 20900.0 | 11.61 | 24339 | 1.16 | 0.089 | OK |
| sv7_r64_c128_g128_t32_multiwarp_ub | multiwarp_ub | 66496 | 32768 | 2078 | 0.10 | 950 | 21730.0 | 12.07 | 25948 | 1.19 | 0.089 | OK |
| sv7_r64_c128_g128_t32_reload | reload | 65728 | 32768 | 2054 | 0.10 | 300 | 20840.0 | 11.58 | 24423 | 1.17 | 0.088 | OK |

Implied camodel frequency (VF cycles / VF running_time): 1.799-1.801 GHz.

## Top opcodes

- `sv5_r64_c128_g16_t32_keep_in_rf`: SIMT_FMNMX:240, SIMT_STK:194, SIMT_LDK:194, SIMT_LDS:128, SIMT_F2F:128, SIMT_IADD_I:113, SIMT_FMNMX_I:32, MOV_XD_IMM:23
- `sv5_r64_c128_g16_t32_ub_stream`: SIMT_LDS:256, SIMT_F2F:256, SIMT_IADD_I:241, SIMT_FMNMX:240, SIMT_FMNMX_I:32, SIMT_IMAD_I:31, MOV_XD_IMM:23, SIMT_STS:16
- `sv6_r64_c128_g32_t32_keep_in_warp`: SIMT_IADD_I:329, SIMT_LDK:256, SIMT_LEA:256, SIMT_FMNMX:256, SIMT_ISETP_I:256, SIMT_BRANCH:256, SIMT_STK:256, SIMT_LDS:128
- `sv6_r64_c128_g32_t32_reload`: SIMT_IADD_I:272, SIMT_LEA:257, SIMT_LDS:256, SIMT_IADD:256, SIMT_F2F:256, SIMT_FMNMX:256, SIMT_ISETP_I:256, SIMT_BRANCH:256
- `sv7_r64_c128_g128_t32_multiwarp_ub`: SIMT_IADD_I:274, SIMT_FMNMX:262, SIMT_LDS:260, SIMT_LEA:257, SIMT_F2F:256, SIMT_ISETP_I:256, SIMT_BRANCH:256, SIMT_IADD_X:192
- `sv7_r64_c128_g128_t32_reload`: SIMT_IADD_I:260, SIMT_LEA:257, SIMT_LDS:256, SIMT_IADD:256, SIMT_F2F:256, SIMT_ISETP_I:256, SIMT_FMNMX:256, SIMT_BRANCH:256
- `sv7_r64_c128_g64_t32_multiwarp_ub`: SIMT_LDS:272, SIMT_F2F:256, SIMT_FMNMX:252, SIMT_IADD_I:232, SIMT_LEA:50, SIMT_FMNMX_I:24, MOV_XD_IMM:23, SIMT_STS:20
- `sv7_r64_c128_g64_t32_reload`: SIMT_IADD_I:264, SIMT_LEA:257, SIMT_LDS:256, SIMT_IADD:256, SIMT_F2F:256, SIMT_ISETP_I:256, SIMT_FMNMX:256, SIMT_BRANCH:256
