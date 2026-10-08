# VF cycle theory vs camodel measurement — SV5–SV7 RF-capacity ladder

OUT: `/tmp/pr272_sv567_rfladder_20261006`

Assumptions: I_assum(Simt)=0.2 (branchy 0.1, unroll<= 16), I_assum(VMI)=0.85, O_base=300, O_loop=250, O_ub_roundtrip=400, freq=1.80 GHz.

`C_pred = O_arm + ceil((W/T) / I_assum)`; see `reports/ST_VF_THEORY.md`.

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

Implied camodel frequency from VF rows (cycles/running_time): 1.799–6.947 GHz (mean 3.333 GHz) — used as the µs conversion basis.
