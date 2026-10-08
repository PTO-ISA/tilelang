# ST-SV8 — Case-3 end-to-end: reduce + reduced eltwise + bcast mul

**Date:** 2026-09-23 ~17:20 HKT (opsim GREEN Ascend950PR_9599)  
**Suite:** `/workspace/st_simtvf_parallel`  
**Kernel:** `kernels/sv8_case3_bcast.py`  
**Umbrella:** [`ST_CASE3_DESIGN.md`](ST_CASE3_DESIGN.md)  
**Legacy:** `kernels/sv8_block_reduce_32x32.py` (tags `sv8_32x32_t*`) — retained; gold fallback kept.

---

## 1. Case design principle

SV8 is the **full Case-3 pipeline**: group absmax → `sf_inv[R,CG]` → broadcast-multiply onto activations so `Out[i,j] = X[i,j] * sf_inv[i, j//G]` (fp16). Two arms contrast **how** the consumer sees the scale:

- **`live`**: keep `sf_inv` in fragment RF; consumer walks `Parallel(R,CG)+serial(G)` (same pattern as old SV6 remat — avoids `j//G` InverseAffine bugs).
- **`spill_dist`**: after computing `sf_inv` in frag, **layout-spill** into shared `scale_ub[R,C]` by writing each group scalar into G lanes, then consumer reloads `scale_ub[i,j]` (VL-friendly rehome). Documented as **layout-transform spill**, not capacity spill.

If expand-to-`[R,C]` cannot express under layout infer, fallback is weaker `shared[R,CG]` reload (document in opsim notes).

Defaults: **R=64, C=128, G=16, T=32**. Tags: `sv8_r…_live` / `sv8_r…_spill_dist`.

## 2. Expectation GPU vs NPU

| concern | GPU mapping | NPU SimtVF expectation |
|---------|-------------|------------------------|
| Live scales | Keep group scales in RF; index `j/G` | Live frag + `Parallel(R,CG)+serial(G)`; `alloc_var` pull before fp16 store (SV1B uint2 lesson) |
| Spill | Rare capacity spill to LDS | **Layout** spill: expand `[R,CG]→[R,C]` in shared so consumer is VL-aligned (`scale_ub[i,j]`) |
| Bcast mul | `out = x * sf_inv` | fp16 out via Cast; shared `x_ub` read in reduce + consumer |
| Reduce | Same as SV5 (G=16) | Same absmax loop; then either live use or expand |
| Cost delta | live vs spill ≈ LDS traffic | Expect spill_dist more MTE / shared traffic; live higher frag RF pressure |

## 3. Simt code slice

CLI: `python kernels/sv8_case3_bcast.py [R] [C] [G] [threads] [live|spill_dist]`  
Defaults `64 128 16 32 live`.

### Arm `live`

```python
with T.SimtVF(threads=threads):
    sf_inv = T.alloc_fragment((R, CG), "float32")
    for i, g in T.Parallel(R, CG):
        m = T.alloc_var("float32", init=0.0)
        for t in T.serial(G):
            vv = T.Cast("float32", x_ub[i, g * G + t])
            m = T.max(m, T.max(vv, -vv))
        m = T.max(m, T.float32(1e-6))
        sf_inv[i, g] = T.float32(1.0) / m
    for i, g in T.Parallel(R, CG):
        s = T.alloc_var("float32")
        s = sf_inv[i, g]
        for t in T.serial(G):
            j = g * G + t
            out_ub[i, j] = T.Cast("float16", T.Cast("float32", x_ub[i, j]) * s)
```

### Arm `spill_dist` (layout-transform spill)

```python
scale_ub = T.alloc_shared((R, C), "float32")  # expanded layout
with T.SimtVF(threads=threads):
    sf_inv = T.alloc_fragment((R, CG), "float32")
    # … same reduce → sf_inv …
    for i, g in T.Parallel(R, CG):
        s = T.alloc_var("float32")
        s = sf_inv[i, g]
        for t in T.serial(G):
            scale_ub[i, g * G + t] = s          # expand to G lanes
    for i, j in T.Parallel(R, C):
        s = T.alloc_var("float32")
        s = scale_ub[i, j]                      # VL-friendly reload
        out_ub[i, j] = T.Cast("float16", T.Cast("float32", x_ub[i, j]) * s)
```

## 4. Simt performance (opsim Ascend950PR_9599)

**GREEN** — both arms PASS; `spill_dist` layout-expand compiled (no fallback to shared`[R,CG]` needed).

| tag | PASS | wall µs | IPC_proxy | Acc/RF map (CCE) | notes |
|-----|------|--------:|----------:|------------------|-------|
| sv8_r64_c128_g16_t32_live | **PASS** | **17.04** | **0.141** | `launch_bounds=32`; CCE `float sf_inv[16]` live; VF_SIMT 15.78 µs / 28402 cyc / body_instr=3996 | live RF consumer; maxabs=0 |
| sv8_r64_c128_g16_t32_spill_dist | **PASS** | **18.23** | **0.138** | `launch_bounds=32`; CCE `float sf_inv[16]` + UB expand `scale_ub[R,C]` fp32; VF_SIMT 16.97 µs / 30545 cyc / body_instr=4220 | layout spill; +1.19 µs vs live |

Gold: seed 8, rtol/atol `2e-2` (fp16), Out `[R,C]` fp16. Legacy `sv8_32x32_t*` still supported in gold.

### RF / shared / dumps

| tag | sf_inv elems/thread | shared scale | consumer | dumps |
|-----|--------------------:|--------------|----------|-------|
| …_live | **16** fp32 (`64*8/32`) | none (beyond x/out) | live frag via Parallel+serial | `sources/sv8_r64_c128_g16_t32_live_{tir,source,lowered}.txt` |
| …_spill_dist | transient 16 fp32 | `scale_ub[64,128]` fp32 | reload full-C layout | `sources/sv8_r64_c128_g16_t32_spill_dist_{tir,source,lowered}.txt` |

Delta live→spill_dist: **+1.19 µs** wall / +2143 cyc VF — layout expand + shared reload tax; IPC_proxy nearly flat (~0.14).

## 5. VMI / SimdVF twin

**Implemented** (sources + gold + oneshot). Opsim **BLOCKED** this pass (Simt oneshot held pto-b10).

| item | value |
|------|-------|
| kernel | `kernels/sv8v_case3_bcast.py` |
| tags | `sv8v_r64_c128_g16_t32_live`, `…_spill_dist` |
| oneshot | `oneshot_sv5v_sv6v_sv8v.sh` |
| gold | same as SV8 (seed 8) |

### Code slice — `live`

```python
with T.SimdVF(lanes=lanes):
    # sf_inv[R,CG] in shared (SimdVF-green)
    for i, g in T.Parallel(R, CG):
        # … absmax → sf_inv[i,g] …
    for i, g in T.Parallel(R, CG):
        s = T.alloc_var("float32"); s = sf_inv[i, g]
        for t in T.serial(G):
            j = g * G + t
            out_ub[i, j] = T.Cast("float16", T.Cast("float32", x_ub[i, j]) * s)
```

### Code slice — `spill_dist`

```python
scale_ub = T.alloc_shared((R, C), "float32")  # expand layout
with T.SimdVF(lanes=lanes):
    # … reduce → sf_inv …
    for i, g in T.Parallel(R, CG):
        s = sf_inv[i, g]
        for t in T.serial(G):
            scale_ub[i, g * G + t] = s
    for i, j in T.Parallel(R, C):
        out_ub[i, j] = T.Cast("float16", T.Cast("float32", x_ub[i, j]) * scale_ub[i, j])
```

| tag | compile | opsim | notes |
|-----|---------|-------|-------|
| sv8v_…_live | pending | **BLOCKED** | overlay + Simt idle |
| sv8v_…_spill_dist | pending | **BLOCKED** | layout-transform spill |


Box mirror of dumps: `reports/sources/*_r64*_{tir,source,lowered}.txt` (from `st_simtvf_sv5_sv6_sv8_reports.tgz`).

## Artifact paths

| kind | path pattern |
|------|----------------|
| SO | `/tmp/st_simtvf_parallel/so/sv8_r64_c128_g16_t32_{live,spill_dist}.so` |
| sources / logs | `sources/*sv8_r64*`, `logs/compile_sv8_*.log`, `opsim_sv8_*.log` |
| summary / tgz | `SUMMARY_sv5_sv6_sv8_raw.txt`, `/tmp/st_simtvf_sv5_sv6_sv8_reports.tgz` |

## Knobs

| knob | values | cliff |
|------|--------|-------|
| arm | live \| spill_dist | RF keep vs layout-spill MTE |
| G | **16** (match SV5) | group width / expand traffic |
| T | 32 | elems/thread for sf_inv |

## Blockers (if any)

None for SimtVF Case-3 tags — expand `scale_ub[i, g*G+t] = sf_inv[i,g]` compiled cleanly; no shared`[R,CG]` fallback required.
