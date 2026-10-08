# Simt ↔ VMI twin compare report

**Generated:** harvest_simt_vmi_compare.py from `/tmp/st_simtvf_parallel`
**Suite reports dir:** `/tmp/st_simtvf_parallel_suite/reports`

## Purpose

Paired Simt vs VMI twin of the **same** ST. Mapping modes from [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md).
Perf deltas = translation cost, not schedule cheating.

**IPC recipes:** Simt = `IPC_proxy` (body_instr / VF_SIMT.cycles). VMI = `EXIPC` or `IFU` if present, else `IPC_proxy(VF)`.

## Performance table (cycles / instr / IPC)

| Tag pair | Mode | Simt µs | VMI µs | Δµs | Simt cycles | VMI cycles | Δcycles | Simt instr# | VMI instr# | Δinstr | Simt IPC | VMI IPC | ΔIPC | Simt st | VMI st |
|----------|------|--------:|-------:|----:|------------:|-----------:|--------:|------------:|-----------:|-------:|---------:|--------:|-----:|---------|--------|
| sv1_e256_t32 ↔ sv1v_e256_t64 | 1 | 1.210 | 0.900 | -0.310 | 727 | 91 | -636 | 131 | 116 | -15 | 0.180 | 1.275 (IPC_proxy(VF)) | 1.095 | PASS | PASS |
| sv1_e2048_t32 ↔ sv1v_e2048_t64 | 1 | 2.580 | 1.250 | -1.330 | 2625 | 343 | -2282 | 515 | 480 | -35 | 0.196 | 1.399 (IPC_proxy(VF)) | 1.203 | PASS | PASS |
| sv2_frag_live ↔ sv2v_frag_live | 2 | 4.930 | — | — | 7379 | — | — | 743 | — | — | 0.101 | — | — | PASS | MISSING |
| sv2_reload ↔ sv2v_reload | 1 | 4.240 | — | — | 6077 | — | — | 585 | — | — | 0.096 | — | — | PASS | MISSING |
| sv3_m24_keep ↔ sv3v_m24_keep | 3 | 6.790 | 4.570 | -2.220 | 9105 | 5181 | -3924 | 2941 | 3701 | 760 | 0.323 | 0.714 (IPC_proxy(VF)) | 0.391 | PASS | PASS |
| sv3_m32_keep ↔ sv3v_m32_keep | 3 | 8.630 | 5.810 | -2.820 | 11874 | 6861 | -5013 | 4032 | 4885 | 853 | 0.340 | 0.712 (IPC_proxy(VF)) | 0.372 | PASS | PASS |
| sv3_split_cm16 ↔ sv3v_split_cm16 | 3 | 7.790 | 5.840 | -1.950 | 10384 | 6950 | -3434 | 3680 | 4997 | 1317 | 0.354 | 0.719 (IPC_proxy(VF)) | 0.365 | PASS | PASS |

> **Note (2026-09-24):** SV4 keep2 pairs below are **historical**. Active ST-SV4 is gather+psum: `keep_idx` (mode 3) / `remat_idx` (mode 1+3). See [`ST_SV4_INDEX_GATHER_PSUM.md`](ST_SV4_INDEX_GATHER_PSUM.md). Re-harvest after opsim to fill new pairs.
| sv4_b8_keep2 ↔ sv4v_b8_keep2 | 3 | 3.790 | — | — | 5069 | — | — | 587 | — | — | 0.116 | — | — | PASS | MISSING |
| sv4_b16_keep2 ↔ sv4v_b16_keep2 | 3 | 6.830 | — | — | 10077 | — | — | 1059 | — | — | 0.105 | — | — | PASS | MISSING |
| sv5_g16 ↔ sv5v_g16 | 2 | 9.340 | — | — | 14835 | — | — | 1529 | — | — | 0.103 | — | — | PASS | MISSING |
| sv6_g32 ↔ sv6v_g32 | 2 | 12.250 | — | — | 20094 | — | — | 2202 | — | — | 0.110 | — | — | PASS | MISSING |
| sv7_g64 ↔ sv7v_g64 | 2 | 14.440 | — | — | 24157 | — | — | 2174 | — | — | 0.090 | — | — | PASS | MISSING |
| sv7_g128 ↔ sv7v_g128 | 2 | 14.690 | — | — | 24614 | — | — | 2160 | — | — | 0.088 | — | — | PASS | MISSING |
| sv8_live ↔ sv8v_live | 3 | 17.040 | — | — | 28402 | — | — | 3986 | — | — | 0.140 | — | — | PASS | MISSING |
| sv8_spill_dist ↔ sv8v_spill_dist | 1 | 18.230 | — | — | 30545 | — | — | 4210 | — | — | 0.138 | — | — | PASS | MISSING |
| sv9_k1_keep ↔ sv9v_k1_keep | 3 | 1.170 | — | — | 637 | — | — | 157 | — | — | 0.246 | — | — | PASS | MISSING |
| sv9_k8_keep ↔ sv9v_k8_keep | 3 | 1.480 | — | — | 1122 | — | — | 557 | — | — | 0.496 | — | — | PASS | MISSING |
| sv9_k8_remat_scores ↔ sv9v_k8_remat_scores | 1+3 | 2.020 | — | — | 2094 | — | — | 621 | — | — | 0.297 | — | — | PASS | MISSING |
| sv9_k8_remat_idx ↔ sv9v_k8_remat_idx | 3 | 1.480 | — | — | 1122 | — | — | 557 | — | — | 0.496 | — | — | PASS | MISSING |

**Filled pairs (both PASS + csv):** 5 / 19

> **SV3V update (2026-10-05, REGRESSION-2):** the three `sv3*` rows and their EX sections were
> (re-)harvested after the SV3V SimdVF rewrite — see [`ST_SV3V_FIX.md`](ST_SV3V_FIX.md).
> All other rows are left as previously harvested.

## EX highlight (per pair)

Filter: `pipe` contains `EX`/`RVECEX`, plus Simt `simt_*` opcodes. Δ = VMI − Simt.

### sv1_e256_t32 ↔ sv1v_e256_t64 (mode 1)

| opcode | pipe | Simt call_count | Simt cycles | VMI call_count | VMI cycles | Δcount | Δcycles |
|--------|------|----------------:|------------:|---------------:|-----------:|-------:|--------:|
| `SIMT_LDS` | RVECLD (Simt) | 16 | 451 | 0 | 0 | -16 | -451 |
| `SIMT_STS` | RVECST (Simt) | 8 | 172 | 0 | 0 | -8 | -172 |
| `SIMT_FADD` | RVECEX (Simt) | 8 | 64 | 0 | 0 | -8 | -64 |
| `RV_VADD` | RVECEX (VMI/EX) | 0 | 0 | 4 | 28 | +4 | +28 |
| `SIMT_S2R` | RVECEX (Simt) | 2 | 20 | 0 | 0 | -2 | -20 |
| `SIMT_LEA` | RVECEX (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |
| `SIMT_END` | RVECLP (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |
| `SIMT_IADD_I` | RVECEX (Simt) | 1 | 8 | 0 | 0 | -1 | -8 |
| `SIMT_SHFI` | RVECEX (Simt) | 1 | 7 | 0 | 0 | -1 | -7 |
| `RV_PSET` | RVECEX (VMI/EX) | 0 | 0 | 1 | 6 | +1 | +6 |

EX totals: Simt count=38 cycles=740; VMI count=5 cycles=34; Δcount=-33 Δcycles=-706.

**Top diffs:**
- **removed on VMI:** `SIMT_LDS` was count=16 cycles=451
- **removed on VMI:** `SIMT_STS` was count=8 cycles=172
- **removed on VMI:** `SIMT_FADD` was count=8 cycles=64
- **new on VMI:** `RV_VADD` count=4 cycles=28
- **removed on VMI:** `SIMT_S2R` was count=2 cycles=20
- **removed on VMI:** `SIMT_LEA` was count=1 cycles=9
- **removed on VMI:** `SIMT_END` was count=1 cycles=9
- **removed on VMI:** `SIMT_IADD_I` was count=1 cycles=8
- **removed on VMI:** `SIMT_SHFI` was count=1 cycles=7
- **new on VMI:** `RV_PSET` count=1 cycles=6

### sv1_e2048_t32 ↔ sv1v_e2048_t64 (mode 1)

| opcode | pipe | Simt call_count | Simt cycles | VMI call_count | VMI cycles | Δcount | Δcycles |
|--------|------|----------------:|------------:|---------------:|-----------:|-------:|--------:|
| `SIMT_LDS` | RVECLD (Simt) | 128 | 3584 | 0 | 0 | -128 | -3584 |
| `SIMT_STS` | RVECST (Simt) | 64 | 1344 | 0 | 0 | -64 | -1344 |
| `SIMT_IADD_I` | RVECEX (Simt) | 161 | 1288 | 0 | 0 | -161 | -1288 |
| `SIMT_FADD` | RVECEX (Simt) | 64 | 512 | 0 | 0 | -64 | -512 |
| `RV_VADD` | RVECEX (VMI/EX) | 0 | 0 | 32 | 224 | +32 | +224 |
| `SIMT_S2R` | RVECEX (Simt) | 2 | 20 | 0 | 0 | -2 | -20 |
| `SIMT_LEA` | RVECEX (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |
| `SIMT_END` | RVECLP (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |
| `SIMT_SHFI` | RVECEX (Simt) | 1 | 7 | 0 | 0 | -1 | -7 |
| `RV_PSET` | RVECEX (VMI/EX) | 0 | 0 | 1 | 6 | +1 | +6 |

EX totals: Simt count=422 cycles=6773; VMI count=33 cycles=230; Δcount=-389 Δcycles=-6543.

**Top diffs:**
- **removed on VMI:** `SIMT_LDS` was count=128 cycles=3584
- **removed on VMI:** `SIMT_STS` was count=64 cycles=1344
- **removed on VMI:** `SIMT_IADD_I` was count=161 cycles=1288
- **removed on VMI:** `SIMT_FADD` was count=64 cycles=512
- **new on VMI:** `RV_VADD` count=32 cycles=224
- **removed on VMI:** `SIMT_S2R` was count=2 cycles=20
- **removed on VMI:** `SIMT_LEA` was count=1 cycles=9
- **removed on VMI:** `SIMT_END` was count=1 cycles=9
- **removed on VMI:** `SIMT_SHFI` was count=1 cycles=7
- **new on VMI:** `RV_PSET` count=1 cycles=6

### sv3_m24_keep ↔ sv3v_m24_keep (mode 3)

| opcode | pipe | Simt call_count | Simt cycles | VMI call_count | VMI cycles | Δcount | Δcycles |
|--------|------|----------------:|------------:|---------------:|-----------:|-------:|--------:|
| `SIMT_LDS` | RVECLD (Simt) | 784 | 21296 | 0 | 0 | -784 | -21296 |
| `SIMT_FFMA` | RVECEX (Simt) | 768 | 8832 | 0 | 0 | -768 | -8832 |
| `SIMT_F2F` | RVECEX (Simt) | 400 | 3200 | 0 | 0 | -400 | -3200 |
| `SIMT_PRMT` | RVECEX (Simt) | 384 | 3072 | 0 | 0 | -384 | -3072 |
| `RV_VMUL` | RVECEX (VMI/EX) | 0 | 0 | 384 | 3072 | +384 | +3072 |
| `RV_VCVT_F2F` | RVECEX (VMI/EX) | 0 | 0 | 385 | 2695 | +385 | +2695 |
| `RV_VADD` | RVECEX (VMI/EX) | 0 | 0 | 384 | 2688 | +384 | +2688 |
| `SIMT_IADD_I` | RVECEX (Simt) | 322 | 2576 | 0 | 0 | -322 | -2576 |
| `RV_VDUPS` | RVECEX (VMI/EX) | 0 | 0 | 385 | 2310 | +385 | +2310 |
| `SIMT_STS` | RVECST (Simt) | 48 | 1008 | 0 | 0 | -48 | -1008 |
| `SIMT_MOV` | RVECEX (Simt) | 48 | 336 | 0 | 0 | -48 | -336 |
| `SIMT_LEA` | RVECEX (Simt) | 33 | 313 | 0 | 0 | -33 | -313 |
| `SIMT_IMAD_I` | RVECEX (Simt) | 16 | 160 | 0 | 0 | -16 | -160 |
| `SIMT_ISETP_I` | RVECEX (Simt) | 16 | 112 | 0 | 0 | -16 | -112 |
| `SIMT_BRANCH` | RVECLP (Simt) | 16 | 80 | 0 | 0 | -16 | -80 |
| `SIMT_S2R` | RVECEX (Simt) | 2 | 20 | 0 | 0 | -2 | -20 |
| `SIMT_MOVI` | RVECEX (Simt) | 2 | 14 | 0 | 0 | -2 | -14 |
| `SIMT_SHFI` | RVECEX (Simt) | 2 | 14 | 0 | 0 | -2 | -14 |
| `RV_PSET` | RVECEX (VMI/EX) | 0 | 0 | 2 | 12 | +2 | +12 |
| `SIMT_END` | RVECLP (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |

EX totals: Simt count=2842 cycles=41042; VMI count=1540 cycles=10777; Δcount=-1302 Δcycles=-30265.

**Top diffs:**
- **removed on VMI:** `SIMT_LDS` was count=784 cycles=21296
- **removed on VMI:** `SIMT_FFMA` was count=768 cycles=8832
- **removed on VMI:** `SIMT_F2F` was count=400 cycles=3200
- **removed on VMI:** `SIMT_PRMT` was count=384 cycles=3072
- **new on VMI:** `RV_VMUL` count=384 cycles=3072
- **new on VMI:** `RV_VCVT_F2F` count=385 cycles=2695
- **new on VMI:** `RV_VADD` count=384 cycles=2688
- **removed on VMI:** `SIMT_IADD_I` was count=322 cycles=2576
- **new on VMI:** `RV_VDUPS` count=385 cycles=2310
- **removed on VMI:** `SIMT_STS` was count=48 cycles=1008
- **removed on VMI:** `SIMT_MOV` was count=48 cycles=336
- **removed on VMI:** `SIMT_LEA` was count=33 cycles=313

### sv3_m32_keep ↔ sv3v_m32_keep (mode 3)

| opcode | pipe | Simt call_count | Simt cycles | VMI call_count | VMI cycles | Δcount | Δcycles |
|--------|------|----------------:|------------:|---------------:|-----------:|-------:|--------:|
| `SIMT_LDS` | RVECLD (Simt) | 1040 | 28208 | 0 | 0 | -1040 | -28208 |
| `SIMT_FFMA` | RVECEX (Simt) | 1024 | 11776 | 0 | 0 | -1024 | -11776 |
| `SIMT_IADD_I` | RVECEX (Simt) | 594 | 4752 | 0 | 0 | -594 | -4752 |
| `SIMT_F2F` | RVECEX (Simt) | 528 | 4224 | 0 | 0 | -528 | -4224 |
| `SIMT_PRMT` | RVECEX (Simt) | 512 | 4096 | 0 | 0 | -512 | -4096 |
| `RV_VMUL` | RVECEX (VMI/EX) | 0 | 0 | 512 | 4096 | +512 | +4096 |
| `RV_VCVT_F2F` | RVECEX (VMI/EX) | 0 | 0 | 513 | 3591 | +513 | +3591 |
| `RV_VADD` | RVECEX (VMI/EX) | 0 | 0 | 512 | 3584 | +512 | +3584 |
| `RV_VDUPS` | RVECEX (VMI/EX) | 0 | 0 | 513 | 3078 | +513 | +3078 |
| `SIMT_STS` | RVECST (Simt) | 64 | 1344 | 0 | 0 | -64 | -1344 |
| `SIMT_MOV` | RVECEX (Simt) | 64 | 448 | 0 | 0 | -64 | -448 |
| `SIMT_LEA` | RVECEX (Simt) | 33 | 313 | 0 | 0 | -33 | -313 |
| `SIMT_IADD` | RVECEX (Simt) | 16 | 128 | 0 | 0 | -16 | -128 |
| `SIMT_SHFI` | RVECEX (Simt) | 18 | 126 | 0 | 0 | -18 | -126 |
| `SIMT_ISETP_I` | RVECEX (Simt) | 16 | 112 | 0 | 0 | -16 | -112 |
| `SIMT_BRANCH` | RVECLP (Simt) | 16 | 80 | 0 | 0 | -16 | -80 |
| `SIMT_S2R` | RVECEX (Simt) | 2 | 20 | 0 | 0 | -2 | -20 |
| `SIMT_MOVI` | RVECEX (Simt) | 2 | 14 | 0 | 0 | -2 | -14 |
| `RV_PSET` | RVECEX (VMI/EX) | 0 | 0 | 2 | 12 | +2 | +12 |
| `SIMT_END` | RVECLP (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |

EX totals: Simt count=3930 cycles=55650; VMI count=2052 cycles=14361; Δcount=-1878 Δcycles=-41289.

**Top diffs:**
- **removed on VMI:** `SIMT_LDS` was count=1040 cycles=28208
- **removed on VMI:** `SIMT_FFMA` was count=1024 cycles=11776
- **removed on VMI:** `SIMT_IADD_I` was count=594 cycles=4752
- **removed on VMI:** `SIMT_F2F` was count=528 cycles=4224
- **removed on VMI:** `SIMT_PRMT` was count=512 cycles=4096
- **new on VMI:** `RV_VMUL` count=512 cycles=4096
- **new on VMI:** `RV_VCVT_F2F` count=513 cycles=3591
- **new on VMI:** `RV_VADD` count=512 cycles=3584
- **new on VMI:** `RV_VDUPS` count=513 cycles=3078
- **removed on VMI:** `SIMT_STS` was count=64 cycles=1344
- **removed on VMI:** `SIMT_MOV` was count=64 cycles=448
- **removed on VMI:** `SIMT_LEA` was count=33 cycles=313

### sv3_split_cm16 ↔ sv3v_split_cm16 (mode 3)

| opcode | pipe | Simt call_count | Simt cycles | VMI call_count | VMI cycles | Δcount | Δcycles |
|--------|------|----------------:|------------:|---------------:|-----------:|-------:|--------:|
| `SIMT_LDS` | RVECLD (Simt) | 1056 | 28608 | 0 | 0 | -1056 | -28608 |
| `SIMT_FFMA` | RVECEX (Simt) | 1024 | 11776 | 0 | 0 | -1024 | -11776 |
| `SIMT_F2F` | RVECEX (Simt) | 544 | 4352 | 0 | 0 | -544 | -4352 |
| `SIMT_PRMT` | RVECEX (Simt) | 512 | 4096 | 0 | 0 | -512 | -4096 |
| `RV_VMUL` | RVECEX (VMI/EX) | 0 | 0 | 512 | 4096 | +512 | +4096 |
| `RV_VCVT_F2F` | RVECEX (VMI/EX) | 0 | 0 | 513 | 3591 | +513 | +3591 |
| `RV_VADD` | RVECEX (VMI/EX) | 0 | 0 | 512 | 3584 | +512 | +3584 |
| `RV_VDUPS` | RVECEX (VMI/EX) | 0 | 0 | 513 | 3078 | +513 | +3078 |
| `SIMT_STS` | RVECST (Simt) | 64 | 1344 | 0 | 0 | -64 | -1344 |
| `SIMT_IADD_I` | RVECEX (Simt) | 98 | 784 | 0 | 0 | -98 | -784 |
| `SIMT_LEA` | RVECEX (Simt) | 66 | 626 | 0 | 0 | -66 | -626 |
| `SIMT_MOV` | RVECEX (Simt) | 69 | 483 | 0 | 0 | -69 | -483 |
| `SIMT_IADD` | RVECEX (Simt) | 34 | 274 | 0 | 0 | -34 | -274 |
| `SIMT_SHFI` | RVECEX (Simt) | 34 | 238 | 0 | 0 | -34 | -238 |
| `SIMT_ISETP_I` | RVECEX (Simt) | 32 | 224 | 0 | 0 | -32 | -224 |
| `SIMT_BRANCH` | RVECLP (Simt) | 34 | 170 | 0 | 0 | -34 | -170 |
| `SIMT_PLOP3` | RVECEX (Simt) | 4 | 28 | 0 | 0 | -4 | -28 |
| `SIMT_MOVI` | RVECEX (Simt) | 3 | 21 | 0 | 0 | -3 | -21 |
| `SIMT_S2R` | RVECEX (Simt) | 2 | 20 | 0 | 0 | -2 | -20 |
| `RV_PSET` | RVECEX (VMI/EX) | 0 | 0 | 2 | 12 | +2 | +12 |
| `SIMT_END` | RVECLP (Simt) | 1 | 9 | 0 | 0 | -1 | -9 |
| `SIMT_ISETP` | RVECEX (Simt) | 1 | 7 | 0 | 0 | -1 | -7 |

EX totals: Simt count=3578 cycles=53060; VMI count=2052 cycles=14361; Δcount=-1526 Δcycles=-38699.

**Top diffs:**
- **removed on VMI:** `SIMT_LDS` was count=1056 cycles=28608
- **removed on VMI:** `SIMT_FFMA` was count=1024 cycles=11776
- **removed on VMI:** `SIMT_F2F` was count=544 cycles=4352
- **removed on VMI:** `SIMT_PRMT` was count=512 cycles=4096
- **new on VMI:** `RV_VMUL` count=512 cycles=4096
- **new on VMI:** `RV_VCVT_F2F` count=513 cycles=3591
- **new on VMI:** `RV_VADD` count=512 cycles=3584
- **new on VMI:** `RV_VDUPS` count=513 cycles=3078
- **removed on VMI:** `SIMT_STS` was count=64 cycles=1344
- **removed on VMI:** `SIMT_IADD_I` was count=98 cycles=784
- **removed on VMI:** `SIMT_LEA` was count=66 cycles=626
- **removed on VMI:** `SIMT_MOV` was count=69 cycles=483

### sv5_g16 ↔ sv5v_g16 (mode 2)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv6_g32 ↔ sv6v_g32 (mode 2)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv7_g64 ↔ sv7v_g64 (mode 2)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv7_g128 ↔ sv7v_g128 (mode 2)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv8_live ↔ sv8v_live (mode 3)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv8_spill_dist ↔ sv8v_spill_dist (mode 1)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv9_k1_keep ↔ sv9v_k1_keep (mode 3)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv9_k8_keep ↔ sv9v_k8_keep (mode 3)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv9_k8_remat_scores ↔ sv9v_k8_remat_scores (mode 1+3)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

### sv9_k8_remat_idx ↔ sv9v_k8_remat_idx (mode 3)

*(empty — awaiting opsim / instr_exe.csv on one or both sides)*

## Mapping-mode reminder

See [`ST_VMI_TWINS.md`](ST_VMI_TWINS.md). Do not bolt AABBCC / token_tile / vf_fuse onto a twin.


## Opsim status / blockers (2026-09-24 ~14:48 HKT)

| Result | Detail |
|--------|--------|
| **PASS** | `sv1v_e256_t64` 0.9 µs; `sv1v_e2048_t64` 1.25 µs (compare filled above) |
| **ABI** | SimdVF `lanes ∈ {64,128,256}` only → Simt T=32 twins use **lanes=64**, tags `*_t64` |
| **PASS SV3V (2026-10-05)** | `sv3v_m24_…_keep` 4.57 µs, `sv3v_m32_…_keep` 5.81 µs, `sv3v_m32_…_split_cm16` 5.84 µs — all `maxabs=0.0`; rewrite in [`ST_SV3V_FIX.md`](ST_SV3V_FIX.md) |
| **COMPILE_FAIL SV2V/SV4V** | `VerifyParallelToPTO`: `local.var` / non-continuous address (need the same UB-shared + lane-aligned rewrite SV3V got) |
| **COMPILE_FAIL SV5V–SV8V** | PTO layout: `Parallel(R,CG)` with CG=8 not divisible by lanes=64 (Case-3 G=16). No valid lane divides 8 |
| **COMPILE_FAIL SV9V** | Fragment layout inference (`idx_cand_frag`) under SimdVF |
| **Lib** | Overlay installed only during VMI run; **deps restored** after (`cmp` OK vs backup) |

Follow-up (not this pass): twin source ABI fixes for shared-only Acc/`m`, and Case-3 Parallel shape that stays fidelity-legal while vectorizable.

## Regenerator

```bash
python3 harvest_simt_vmi_compare.py --out reports/ST_VMI_COMPARE.md
```

