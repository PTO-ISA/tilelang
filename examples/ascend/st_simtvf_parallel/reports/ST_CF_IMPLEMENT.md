# ST-CF implement notes (Simt CF1–CF6 + VMI CF1V–CF6V)

**Date:** 2026-09-28 ~10:45 HKT  
**Oneshot Simt:** `oneshot_cf1_cf6.sh`  
**Oneshot VMI:** `oneshot_cf1v_cf6v.sh` (overlay + `target=pto`; trap restores deps lib)  
**Harvest:** `harvest_cf_summary.py`  
**Host:** pto-b10 Ascend950PR_9599 via sim_dsl.sh  
**PR:** https://github.com/WenboCodes/tilelang-deepseek/pull/272

## Sources

| ST | Kernel | Tags (primary) |
|----|--------|----------------|
| CF1 | kernels/cf1_pred_thresh_keep.py | cf1_e256_k{1,8}_t32_keep |
| CF1V | kernels/cf1v_pred_thresh_keep.py | cf1v_e256_k{1,8}_t64_keep |
| CF2 | kernels/cf2_remat_thresh_kill_shared.py | cf2_e256_k8_t32_remat |
| CF2V | kernels/cf2v_remat_thresh_kill_shared.py | cf2v_e256_k8_t64_remat |
| CF3 | kernels/cf3_remat_idx.py | cf3_e256_k8_t32_remat_idx |
| CF3V | kernels/cf3v_remat_idx.py | cf3v_e256_k8_t64_remat_idx |
| CF4 | kernels/cf4_nested_if.py | cf4_e256_t32_pfat{5,25} |
| CF4V | kernels/cf4v_nested_if.py | cf4v_e256_t64_pfat{5,25} |
| CF5 | kernels/cf5_div_ulp_branch.py | cf5_e256_t32_pnear{5,25} |
| CF5V | kernels/cf5v_div_ulp_branch.py | cf5v_e256_t64_pnear{5,25} |
| CF6 | kernels/cf6_newton_branch.py | cf6_e256_t32_pnear{5,25} |
| CF6V | kernels/cf6v_newton_branch.py | cf6v_e256_t64_pnear{5,25} |

Opsim gold: `run_opsim_generic.py` (`cf1_`…`cf6_` and `cf1v_`…`cf6v_`).

## Mask / Select mapping (VMI twins)

| Case | Simt CF | VMI mapping | Mode |
|------|---------|-------------|------|
| **CF1→CF1V** | `if scores[i]>thr: scores[i]=NEG` (KEEP) | `scores[i]=T.Select(scores[i]>thr, NEG, scores[i])` inside `SimdVF`+`Parallel` | **3** stay-alive scores |
| **CF2→CF2V** | remat scores; `if …: s[i]=NEG` | remat Parallel; `s[i]=T.Select(…)` kill-in-shared | **1** remat + publish |
| **CF3→CF3V** | `if i==v: scores[i]=NEG` | `scores[i]=T.Select(i==v, NEG, scores[i])` (remat `i`) | **3** scores; index remat |
| **CF4→CF4V** | nested if hi/lo/fat | nested `T.Select(x>hi, hi, T.Select(x<lo, lo, x*scale))` | **2** fused classify |
| **CF5→CF5V** | Parallel `if ax<eps: 0 else a/b` | **ABI:** `quot=a/b` in Kernel `serial` *outside* SimdVF (Div/Max not Parallel-whitelist; serial-in-VF → AIV scalar fail). VF Parallel: abs via `Select` + near-0 `Select` on quot | **1** remat quot + mask |
| **CF6→CF6V** | Parallel near-0 + Newton `serial(N_FAST)` | **ABI:** Newton body in Kernel `serial` outside SimdVF (Div/Sub/Max); VF Parallel publish only | **1** + thin Parallel store |

**Forbidden on twins:** AABBCC / `vf_fuse` / `token_tile`. Prefer `T.Select` over `T.if_then_else` (`VerifyParallelToPTO` rejects `tirx.if_then_else`). Lanes=64 primary (t32 known ABI fail).

## Summary performance table (Simt + VMI)

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
| CF1V | `cf1v_e256_k1_t64_keep` | VMI mask Select thresh kill KEEP (K=1) | PASS | 1.17 | 577 | 106 | 0.184 | 11 | — |
| CF1V | `cf1v_e256_k8_t64_keep` | VMI mask Select thresh kill KEEP (K=8) | PASS | 1.01 | 301 | 308 | 1.023 | 74 | — |
| CF2V | `cf2v_e256_k8_t64_remat` | VMI remat + kill-in-shared (Select) | PASS | 1.04 | 356 | 425 | 1.194 | 74 | — |
| CF3V | `cf3v_e256_k8_t64_remat_idx` | VMI remat_idx Select | PASS | 1.02 | 321 | 380 | 1.184 | 106 | — |
| CF4V | `cf4v_e256_t64_pfat5` | VMI nested Select; p_fat=5% | PASS | 0.93 | 147 | 136 | 0.925 | 29 | — |
| CF4V | `cf4v_e256_t64_pfat25` | VMI nested Select; p_fat=25% | PASS | 0.93 | 147 | 136 | 0.925 | 29 | — |
| CF5V | `cf5v_e256_t64_pnear5` | VMI div outside VF + near-0 Select; pnear=5% | PASS | 6.23 | 529 | 3211 | 6.070 | 24 | — |
| CF5V | `cf5v_e256_t64_pnear25` | VMI div outside VF + near-0 Select; pnear=25% | PASS | 6.23 | 529 | 3211 | 6.070 | 24 | — |
| CF6V | `cf6v_e256_t64_pnear5` | VMI Newton outside VF; pnear=5% | PASS | 12.35 | 463 | 6746 | 14.570 | 1 | — |
| CF6V | `cf6v_e256_t64_pnear25` | VMI Newton outside VF; pnear=25% | PASS | 12.35 | 463 | 6746 | 14.570 | 1 | — |

**Metric recipes** (same as `harvest_simt_vmi_compare.py` / `harvest_cf_summary.py`):
- **e2e µs** = `core0.veccore0` wall duration from opsim log
- **VF cyc** = `VF_SIMT.cycles` (Simt) or `VF`/`EXIPC`/`IFU` (VMI)
- **instr#** = body opcodes excl. framing + VF wrapper
- **IPC** = body_instr / VF cycles (proxy)
- **EX instr#** = pipe EX/RVECEX or `simt_*`
- **EXIPC** = native row if present; else `—`

## Twin compare notes

- **CF1–CF4:** VMI mask-Select twins competitive or faster (CF2V 1.04 vs Simt CF2 1.95; CF4V 0.93 vs 1.23–1.25). Lockstep Select; no divergent early-exit win for Simt on these micros at E=256.
- **CF5–CF6:** Simt **wins hard** (1.18 / 1.37 µs vs VMI 6.23 / 12.35 µs) because VMI must hoist Div/Newton to Kernel serial outside SimdVF (Parallel value whitelist rejects Div/Sub/Max; serial-in-VF hits AIV "Unsupported scalar instruction").
- **Deps lib:** restored after each VMI oneshot (`GetPredicate(Var)`).

## Compile / ABI blockers (documented)

| Issue | Effect | Mitigation used |
|-------|--------|-----------------|
| `T.if_then_else` → `tirx.if_then_else` | VerifyParallelToPTO reject | Use `T.Select` |
| `Div` / `Sub` / `Max` in Parallel store value | VerifyParallelToPTO reject | CF5/CF6: compute outside SimdVF |
| `serial(Div)` *inside* SimdVF | bisheng AIV "Unsupported scalar instruction" | Kernel-scope serial outside VF |
| `lanes=32` | known COMPILE_FAIL on many twins | Primary matrix lanes=64 |
| Immutable local carry in Newton | `Immutable variable yi` | Carry via `y_work[i]` stores |

**Compile / opsim:** all 10 Simt + 10 VMI tags COMPILE_OK + PASS (2026-09-28).
