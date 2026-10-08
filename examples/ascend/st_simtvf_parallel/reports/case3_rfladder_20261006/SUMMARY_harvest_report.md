# SimtVF Parallel ST suite — SUMMARY

Remote OUT: `/tmp/pr272_sv567_rfladder_20261006`

| tag | status | µs | prior/notes | layout |
|-----|--------|----|-------------|--------|
| probe_sv6v_t_r64_c128_g32_t64_scalar | MISSING | - | (no instr csv) | tir=probe_sv6v_t_r64_c128_g32_t64_scalar_tir.txt; src=probe_sv6v_t_r64_c128_g32_ |
| sv5_r64_c128_g16_t32_keep_in_rf | PASS | 7.040 | VF_SIMT us=5.91 cyc=10641 body_instr=1169 IPC_proxy=0.110 /  | launch_bounds=32; tir=sv5_r64_c128_g16_t32_keep_in_rf_tir.txt; src=sv5_r64_c128_ |
| sv5_r64_c128_g16_t32_ub_stream | PASS | 4.180 | VF_SIMT us=3.06 cyc=5506 body_instr=1211 IPC_proxy=0.220 / t | launch_bounds=32; tir=sv5_r64_c128_g16_t32_ub_stream_tir.txt; src=sv5_r64_c128_g |
| sv5v_r64_c128_g16_t64_ub_stream | PASS | 109.390 | top_opcodes: ADD_IMM:27133, CMP:26639, JUMPC:26639, ST_XD_XN | tir=sv5v_r64_c128_g16_t64_ub_stream_tir.txt; src=sv5v_r64_c128_g16_t64_ub_stream |
| sv6_r64_c128_g32_t32_keep_in_warp | PASS | 14.830 | VF_SIMT us=13.72 cyc=24697 body_instr=2280 IPC_proxy=0.092 / | launch_bounds=32; tir=sv6_r64_c128_g32_t32_keep_in_warp_tir.txt; src=sv6_r64_c12 |
| sv6_r64_c128_g32_t32_reload | PASS | 12.580 | VF_SIMT us=11.53 cyc=20748 body_instr=2219 IPC_proxy=0.107 / | launch_bounds=32; tir=sv6_r64_c128_g32_t32_reload_tir.txt; src=sv6_r64_c128_g32_ |
| sv6v_r64_c128_g32_t64_reload | PASS | 101.220 | top_opcodes: ADD_IMM:25853, CMP:25631, JUMPC:25631, ST_XD_XN | tir=sv6v_r64_c128_g32_t64_reload_tir.txt; src=sv6v_r64_c128_g32_t64_reload_sourc |
| sv7_r64_c128_g128_t32_multiwarp_ub | PASS | 15.420 | VF_SIMT us=14.42 cyc=25948 body_instr=2317 IPC_proxy=0.089 / | launch_bounds=32; tir=sv7_r64_c128_g128_t32_multiwarp_ub_tir.txt; src=sv7_r64_c1 |
| sv7_r64_c128_g128_t32_reload | PASS | 14.580 | VF_SIMT us=13.57 cyc=24423 body_instr=2171 IPC_proxy=0.089 / | launch_bounds=32; tir=sv7_r64_c128_g128_t32_reload_tir.txt; src=sv7_r64_c128_g12 |
| sv7_r64_c128_g64_t32_multiwarp_ub | PASS | 7.320 | VF_SIMT us=6.21 cyc=11182 body_instr=1280 IPC_proxy=0.114 /  | launch_bounds=32; tir=sv7_r64_c128_g64_t32_multiwarp_ub_tir.txt; src=sv7_r64_c12 |
| sv7_r64_c128_g64_t32_reload | PASS | 14.540 | VF_SIMT us=13.52 cyc=24339 body_instr=2187 IPC_proxy=0.090 / | launch_bounds=32; tir=sv7_r64_c128_g64_t32_reload_tir.txt; src=sv7_r64_c128_g64_ |
| sv7v_r64_c128_g128_t64_multiwarp_ub | PASS | 95.440 | top_opcodes: CMP:24959, JUMPC:24959, ADD_IMM:24893, ST_XD_XN | tir=sv7v_r64_c128_g128_t64_multiwarp_ub_tir.txt; src=sv7v_r64_c128_g128_t64_mult |
| sv7v_r64_c128_g128_t64_reload | PASS | 95.430 | top_opcodes: CMP:24959, JUMPC:24959, ADD_IMM:24893, ST_XD_XN | tir=sv7v_r64_c128_g128_t64_reload_tir.txt; src=sv7v_r64_c128_g128_t64_reload_sou |
| sv7v_r64_c128_g64_t64_multiwarp_ub | PASS | 97.160 | top_opcodes: ADD_IMM:25213, CMP:25151, JUMPC:25151, ST_XD_XN | tir=sv7v_r64_c128_g64_t64_multiwarp_ub_tir.txt; src=sv7v_r64_c128_g64_t64_multiw |
| sv7v_r64_c128_g64_t64_reload | PASS | 97.160 | top_opcodes: ADD_IMM:25213, CMP:25151, JUMPC:25151, ST_XD_XN | tir=sv7v_r64_c128_g64_t64_reload_tir.txt; src=sv7v_r64_c128_g64_t64_reload_sourc |

## Priority matrix (SV2)

Expect T32 K8 ≈ 1.48µs, T128 K8 ≈ 3.03µs (prior green AB).

## Per-tag reports

- [`reports/probe_sv6v_t_r64_c128_g32_t64_scalar.md`](probe_sv6v_t_r64_c128_g32_t64_scalar.md)
- [`reports/sv5_r64_c128_g16_t32_keep_in_rf.md`](sv5_r64_c128_g16_t32_keep_in_rf.md)
- [`reports/sv5_r64_c128_g16_t32_ub_stream.md`](sv5_r64_c128_g16_t32_ub_stream.md)
- [`reports/sv5v_r64_c128_g16_t64_ub_stream.md`](sv5v_r64_c128_g16_t64_ub_stream.md)
- [`reports/sv6_r64_c128_g32_t32_keep_in_warp.md`](sv6_r64_c128_g32_t32_keep_in_warp.md)
- [`reports/sv6_r64_c128_g32_t32_reload.md`](sv6_r64_c128_g32_t32_reload.md)
- [`reports/sv6v_r64_c128_g32_t64_reload.md`](sv6v_r64_c128_g32_t64_reload.md)
- [`reports/sv7_r64_c128_g128_t32_multiwarp_ub.md`](sv7_r64_c128_g128_t32_multiwarp_ub.md)
- [`reports/sv7_r64_c128_g128_t32_reload.md`](sv7_r64_c128_g128_t32_reload.md)
- [`reports/sv7_r64_c128_g64_t32_multiwarp_ub.md`](sv7_r64_c128_g64_t32_multiwarp_ub.md)
- [`reports/sv7_r64_c128_g64_t32_reload.md`](sv7_r64_c128_g64_t32_reload.md)
- [`reports/sv7v_r64_c128_g128_t64_multiwarp_ub.md`](sv7v_r64_c128_g128_t64_multiwarp_ub.md)
- [`reports/sv7v_r64_c128_g128_t64_reload.md`](sv7v_r64_c128_g128_t64_reload.md)
- [`reports/sv7v_r64_c128_g64_t64_multiwarp_ub.md`](sv7v_r64_c128_g64_t64_multiwarp_ub.md)
- [`reports/sv7v_r64_c128_g64_t64_reload.md`](sv7v_r64_c128_g64_t64_reload.md)
