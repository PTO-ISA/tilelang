# SimtVF Parallel ST suite — SUMMARY

Remote OUT: `/tmp/st_simtvf_parallel`

| tag | status | µs | prior/notes | layout |
|-----|--------|----|-------------|--------|
| sv1_e2048_t32 | PASS | 2.580 | VF_SIMT us=1.46 cyc=2625 body_instr=525 IPC_proxy=0.200 / to | launch_bounds=32; t1[] sizes=['64']; elems_per_thread≈64.00 |
| sv1_e256_t32 | PASS | 1.210 | VF_SIMT us=0.4 cyc=727 body_instr=141 IPC_proxy=0.194 / top_ | launch_bounds=32; t1[] sizes=['8']; elems_per_thread≈8.00 |
| sv2_e256_k1_t128 | PASS | 1.180 | prior_AB=1.18us | launch_bounds=128; scores[] sizes=['0', '1', '2']; idxs[] sizes=['0', '1', '2']; |
| sv2_e256_k1_t32 | PASS | 1.170 | prior_AB=1.17us | launch_bounds=32; scores[] sizes=['8']; idxs[] sizes=['8']; idx_cand[] sizes=['8 |
| sv2_e256_k8_t128 | PASS | 3.030 | prior_AB=3.03us | launch_bounds=128; scores[] sizes=['0', '1', '2']; idxs[] sizes=['0', '1', '2']; |
| sv2_e256_k8_t32 | PASS | 1.480 | prior_AB=1.48us | launch_bounds=32; scores[] sizes=['8']; idxs[] sizes=['8']; idx_cand[] sizes=['8 |
| sv3_e256_k1_t32 | PASS | 1.150 | VF_SIMT us=0.3 cyc=532 body_instr=190 IPC_proxy=0.357 / top_ | launch_bounds=32; scores[] sizes=['8']; idxs[] sizes=['8']; idx_cand[] sizes=['8 |
| sv3_e256_k8_t32 | PASS | 2.020 | VF_SIMT us=1.16 cyc=2094 body_instr=631 IPC_proxy=0.301 / to | launch_bounds=32; scores[] sizes=['8']; idxs[] sizes=['8']; idx_cand[] sizes=['8 |
| sv4_e256_k8_t128 | PASS | 3.030 | VF_SIMT us=2.17 cyc=3904 body_instr=1650 IPC_proxy=0.423 / t | launch_bounds=128; scores[] sizes=['0', '1', '2']; idx_cand[] sizes=['2']; amax[ |
| sv4_e256_k8_t32 | PASS | 1.480 | VF_SIMT us=0.62 cyc=1122 body_instr=567 IPC_proxy=0.505 / to | launch_bounds=32; scores[] sizes=['8']; idx_cand[] sizes=['8']; amax[] sizes=['0 |
| sv5_e256_t32_frag | PASS | 1.170 | VF_SIMT us=0.35 cyc=638 body_instr=169 IPC_proxy=0.265 / top | launch_bounds=32; x[] sizes=['8']; amax[] sizes=['0', '1']; has AscendAllReduce; |
| sv5_e256_t32_reload | PASS | 1.150 | VF_SIMT us=0.3 cyc=532 body_instr=181 IPC_proxy=0.340 / top_ | launch_bounds=32; x[] sizes=['8']; amax[] sizes=['0', '1']; has AscendAllReduce; |
| sv7_n128_t128 | PASS | 1.220 | VF_SIMT us=0.39 cyc=700 body_instr=231 IPC_proxy=0.330 / top | launch_bounds=128; v[] sizes=['0', '1']; has AscendAllReduce; cross-thread reduc |
| sv7_n128_t32 | PASS | 1.140 | VF_SIMT us=0.32 cyc=569 body_instr=120 IPC_proxy=0.211 / top | launch_bounds=32; v[] sizes=['4']; has AscendAllReduce; cross-thread reduce path |

## Priority matrix (SV2)

Expect T32 K8 ≈ 1.48µs, T128 K8 ≈ 3.03µs (prior green AB).

## Per-tag reports

- [`reports/sv1_e2048_t32.md`](sv1_e2048_t32.md)
- [`reports/sv1_e256_t32.md`](sv1_e256_t32.md)
- [`reports/sv2_e256_k1_t128.md`](sv2_e256_k1_t128.md)
- [`reports/sv2_e256_k1_t32.md`](sv2_e256_k1_t32.md)
- [`reports/sv2_e256_k8_t128.md`](sv2_e256_k8_t128.md)
- [`reports/sv2_e256_k8_t32.md`](sv2_e256_k8_t32.md)
- [`reports/sv3_e256_k1_t32.md`](sv3_e256_k1_t32.md)
- [`reports/sv3_e256_k8_t32.md`](sv3_e256_k8_t32.md)
- [`reports/sv4_e256_k8_t128.md`](sv4_e256_k8_t128.md)
- [`reports/sv4_e256_k8_t32.md`](sv4_e256_k8_t32.md)
- [`reports/sv5_e256_t32_frag.md`](sv5_e256_t32_frag.md)
- [`reports/sv5_e256_t32_reload.md`](sv5_e256_t32_reload.md)
- [`reports/sv7_n128_t128.md`](sv7_n128_t128.md)
- [`reports/sv7_n128_t32.md`](sv7_n128_t32.md)

## CF1–CF6 (Simt control-flow slices)

**Product cut (2026-10-07):** primary = **CF1–CF3 + CF6** only. CF4 / CF5 / CF6b = appendix / historical (files kept). Compare: `SIMT_SIMD_E2E_COMPARE_CF_20261007.md` / `ST_CF_FULL_COMPARE.md`.

**Date:** 2026-09-25 ~14:00 HKT · **Oneshot:** `oneshot_cf1_cf6.sh` · **Status:** all 10 tags PASS  
**Detail:** [`reports/ST_CF_IMPLEMENT.md`](ST_CF_IMPLEMENT.md) · [`reports/ST_CF1_CF6_SLICES.md`](ST_CF1_CF6_SLICES.md)

| case | tag | what to test | status | e2e µs | VF cyc | instr# | IPC | EX instr# | EXIPC |
|------|-----|--------------|--------|-------:|-------:|-------:|----:|----------:|------:|
| CF1 | `cf1_e256_k1_t32_keep` | One Parallel if vs scalar thresh; scores KEEP (K=1) | PASS | 1.15 | 576 | 137 | 0.238 | 40 | — |
| CF1 | `cf1_e256_k8_t32_keep` | One Parallel if vs scalar thresh; scores KEEP (K=8) | PASS | 0.97 | 309 | 256 | 0.828 | 159 | — |
| CF2 | `cf2_e256_k8_t32_remat` | Same if; remat + kill-in-shared | PASS | 1.95 | 2041 | 362 | 0.177 | 265 | — |
| CF3 | `cf3_e256_k8_t32_remat_idx` | remat_idx only (kill/select by rematted i) | PASS | 1.33 | 851 | 272 | 0.320 | 175 | — |
| CF4 | `cf4_e256_t32_pfat5` | Nested if (2-level); p_fat=5% skew | PASS | 1.23 | 758 | 212 | 0.280 | 110 | — |
| CF4 | `cf4_e256_t32_pfat25` | Nested if (2-level); p_fat=25% skew | PASS | 1.25 | 781 | 212 | 0.271 | 110 | — |
| CF5 | `cf5_e256_t32_pnear5` | Div + near-0 range branch (ULP intent); pnear=5% | PASS | 1.18 | 657 | 149 | 0.227 | 56 | — |
| CF5 | `cf5_e256_t32_pnear25` | Div + near-0 range branch (ULP intent); pnear=25% | PASS | 1.18 | 656 | 149 | 0.227 | 56 | — |
| CF6 | `cf6_e256_t32_pnear5` | Newton recip-style + near-0 / extra-iter; pnear=5% | PASS | 1.37 | 961 | 192 | 0.200 | 102 | — |
| CF6 | `cf6_e256_t32_pnear25` | Newton recip-style + near-0 / extra-iter; pnear=25% | PASS | 1.37 | 961 | 192 | 0.200 | 102 | — |

**Metric recipes** (same as `harvest_simt_vmi_compare.py` / `harvest_cf_summary.py`):
- **e2e µs** = `core0.veccore0` wall duration from opsim log
- **VF cyc** = `VF_SIMT.cycles` in `core0.veccore0_instr_exe.csv`
- **instr#** = body opcodes excl. framing `{set_flag,wait_flag,end_label,end,nop,push_pb,dcci}` and the VF_SIMT wrapper
- **IPC** = `IPC_proxy` = body_instr / VF_SIMT.cycles
- **EX instr#** = sum `call_count` where pipe contains EX/RVECEX, or opcode starts with `simt_`
- **EXIPC** = native EXIPC/IFU row if present; SimtVF dumps usually lack it → `—` (do not invent; same practice as ST_VMI_COMPARE)

