# ST-SV1 / ST-SV2 — report 1 (stream eltwise + live RF eltwise/bcast)

**Date:** 2026-09-23 ~14:50 HKT  
**Suite:** `/workspace/st_simtvf_parallel`  
**Scope:** Case A (SV1 measured) + Case B (SV2 both arms PASS on opsim 2026-09-23 ~15:10 HKT). Other STs deferred.

## Suite contract (applies to every ST)

- **Sensitivity target:** state what RF / Parallel / remat behavior this ST is probing (one sentence).
- **Turning-point knobs:** which parameters (E, T, arm, scale grain, …) move the design across a performance cliff; list expected cliff.
- **Artifacts required:** TileLang source slice / TIR or Parallel+fragment view / thread-block RF map / µs+IPC (opsim Ascend950PR_9599 when available).

---

## Case A — ST-SV1 Stream Parallel eltwise

### Sensitivity target

Pure Parallel streak eltwise RF occupancy vs `E` and `T`: `t3 = t1 + t2` with no loop-carry and no bcast. Measures how folded fragment layout (`t1[E/T]` etc.) and VF IPC scale when elems/thread grows from comfort (8) to heavy (64).

### Turning-point parameters

| knob | values | expected cliff |
|------|--------|----------------|
| E | 256, 2048, (optional 8192) | elems/thread = E/T crosses RF comfort (8→64→256) |
| threads T | 32, 128 | T128 shrinks elems/thread but may change mapping / any AllReduce path |

### Source slice

Kernel: `kernels/sv1_stream_eltwise.py`

```python
@T.prim_func
def main(
    A: T.Tensor((E,), "float32"),
    B: T.Tensor((E,), "float32"),
    C: T.Tensor((E,), "float32"),
):
    with T.Kernel(1):
        a_ub = T.alloc_shared((E,), "float32")
        b_ub = T.alloc_shared((E,), "float32")
        c_ub = T.alloc_shared((E,), "float32")
        T.copy(A, a_ub)
        T.copy(B, b_ub)
        with T.SimtVF(threads=threads):
            t1 = T.alloc_fragment((E,), "float32")
            t2 = T.alloc_fragment((E,), "float32")
            t3 = T.alloc_fragment((E,), "float32")
            for i in T.Parallel(E):
                t1[i] = a_ub[i]
                t2[i] = b_ub[i]
                t3[i] = t1[i] + t2[i]
                c_ub[i] = t3[i]
        T.copy(c_ub, C)
```

CLI: `python kernels/sv1_stream_eltwise.py [E] [threads]` → tag `sv1_e{E}_t{threads}` (default E=256 T=32).

### TIR / Parallel view

- Shared stage: `a_ub` / `b_ub` / `c_ub` size `E` (UB load/store outside SimtVF).
- Inside `T.SimtVF(threads=T)`: three `alloc_fragment((E,), float32)` — `t1`, `t2`, `t3`.
- Single `T.Parallel(E)` streak: load → add → store; no reduce, no loop-carried KEEP, no scale bcast.
- Lowered CCE (pto-b10 harvest): `launch_bounds=T`; per-thread arrays fold to `t1[E/T]` (and likewise `t2`, `t3`). Cited from suite SUMMARY / RESULTS (remote sources under `/tmp/st_simtvf_parallel/sources/` on laptop host; not re-pulled this turn — Shell machineId laptop not available on this executor).

### Thread-block register mapping

| threads | E | elems/thread | fragments live |
|--------:|--:|-------------:|----------------|
| 32 | 256 | 8 | t1, t2, t3 each `[8]` |
| 32 | 2048 | 64 | t1, t2, t3 each `[64]` |
| 128 (optional) | 256 | 2 | t1, t2, t3 each `[2]` |
| 128 (optional) | 2048 | 16 | t1, t2, t3 each `[16]` |

At baseline T=32: live fp32 RF ≈ `3 × (E/T)` scalars per thread (plus address temps).

### Measured (opsim Ascend950PR_9599)

From `reports/RESULTS.md` / `SUMMARY.md` (2026-09-23, lib `/tmp/libtilelang.so.deps_backup_ab`):

| tag | µs | VF notes | layout |
|-----|---:|----------|--------|
| sv1_e256_t32 | **1.21** | VF_SIMT ~0.4 µs, cyc=727, body_instr=141, **IPC_proxy≈0.194** | `launch_bounds=32`, `t1[8]` |
| sv1_e2048_t32 | **2.58** | VF_SIMT ~1.46 µs, cyc=2625, body_instr=525, **IPC_proxy≈0.200** | `launch_bounds=32`, `t1[64]` |

Wall time scales ~2.1× when elems/thread goes 8→64; IPC_proxy stays ~0.19–0.20 (stream add is not IPC-bound at these sizes).

### What would flip the cliff

- Push E to 8192 @ T=32 → elems/thread=256; expect RF pressure / spill or longer VF body before IPC rises.
- T=128 @ large E: fewer elems/thread but different launch_bounds; watch whether Parallel fold still keeps three live vectors or remats.
- Adding a live bcast operand (→ Case B / SV5) changes the story from pure streak RF to scale lifetime.

---

## Case B — ST-SV2 Live RF eltwise + scale bcast (32×32 quant-style)

### Sensitivity target

GPU story: a 32×32 quant slice keeps per-row (or per-group) scale coeffs in RF so eltwise consumers never reload them from shared.  
NPU SimtVF equivalent: after row absmax, `scale[R]` stays in **fragment RF** (folded VL) while `out = x / scale[i]` runs — contrast later with a `reload` arm (shared publish + per-consumer load) and with SV6 group-scale remat.

### Turning-point parameters

| knob | values | expected cliff |
|------|--------|----------------|
| R×C | 32×32 baseline; optional 64×64 | RF live = `x[R,C]` + `scale[R]` footprint (fp32) |
| threads | 32, 128 | elems/thread for x (=RC/T) and how many scales each thread owns |
| arm | `frag_live` vs `reload` | reload tax (shared traffic) vs live RF pressure |
| scale grain | per-row (this ST) vs G=16 group (SV6) | more scales → more RF or more remat |

### Source slice

Kernel: `kernels/sv2_eltwise_bcast_rf.py`  
CLI: `python kernels/sv2_eltwise_bcast_rf.py [R] [C] [threads] [arm]`  
Default `32 32 32 frag_live` → tag `sv2_r32_c32_t32_frag_live`.

**Primary arm `frag_live`:**

```python
@T.prim_func
def main(
    X: T.Tensor((R, C), "float16"),
    Out: T.Tensor((R, C), "float16"),
):
    with T.Kernel(1):
        x_ub = T.alloc_shared((R, C), "float16")
        out_ub = T.alloc_shared((R, C), "float16")
        T.copy(X, x_ub)
        with T.SimtVF(threads=threads):
            x = T.alloc_fragment((R, C), "float32")
            scale = T.alloc_fragment((R,), "float32")
            for i in T.Parallel(R):
                m = T.alloc_var("float32", init=0.0)
                for j in T.serial(C):
                    vv = T.Cast("float32", x_ub[i, j])
                    x[i, j] = vv
                    m = T.max(m, T.max(vv, -vv))  # abs; avoid T.abs
                scale[i] = T.max(m, T.float32(1e-6))
            for i, j in T.Parallel(R, C):
                out_ub[i, j] = T.Cast("float16", x[i, j] / scale[i])  # scale live in frag
        T.copy(out_ub, Out)
```

**Optional arm `reload`:** same reduce into fragment, then `scale_s[i] = scale[i]`; consumer does `s = scale_s[i]` before divide (shared reload tax). Separate `@T.prim_func` body (SV6 lesson).

### TIR / Parallel view (intended)

1. Shared: `x_ub`/`out_ub` fp16 `[R,C]`; reload arm adds `scale_s` fp32 `[R]`.
2. SimtVF fragments: `x` fp32 `[R,C]`, `scale` fp32 `[R]`.
3. Phase A — `Parallel(R)` + `serial(C)`: cast+store into `x`, mutable absmax via `alloc_var` + `m = T.max(m, T.max(vv,-vv))`, write `scale[i]`.
4. Phase B — `Parallel(R, C)`: eltwise divide; **frag_live** reads `scale[i]` from fragment; **reload** reads `scale_s[i]` from shared.
5. No `T.reduce_max` here (row absmax needs abs); pattern mirrors SV6 alloc_var style rather than SV8 plain `reduce_max`.

### Thread-block register mapping (predicted → confirmed below)

At R=C=32 T=32: predicted `x[32]` + `scale[1]` per thread — **confirmed in CCE** (see Measured). T128 / 64×64 still the knobs for the RF cliff.

### Measured (opsim Ascend950PR_9599, 2026-09-23 ~15:10 HKT)

| tag | status | µs | VF_SIMT µs | cyc | body_instr | IPC_proxy |
|-----|--------|---:|----------:|----:|-----------:|----------:|
| sv2_r32_c32_t32_frag_live | **PASS** | **4.93** | 4.10 | 7379 | 753 | **0.102** |
| sv2_r32_c32_t32_reload | **PASS** | **4.24** | 3.38 | 6077 | 595 | **0.098** |

At 32×32 T32, **reload is ~14% faster wall** than frag_live (4.24 vs 4.93 µs). CCE shows reload’s store loop is `#pragma unroll`’d; frag_live is a plain serial `s = scale[0]` then scalar half store. IPC_proxy is ~0.10 for both (absmax+normalize dominated, not IPC-bound).

### Thread-block register mapping (from lowered CCE)

Both arms, `launch_bounds=32`, one row per thread (`threadIdx.x * 32 + j`):

| arm | per-thread RF | scale path | store |
|-----|---------------|------------|-------|
| frag_live | `float x[32]`, `float scale[1]`, `float m`, `float s` | `s = scale[0]` (live fragment) | scalar `((half)(x[j]/s))` to UB |
| reload | same `x[32]`/`scale[1]`/`m`/`s` | `scale` → UB float, then `s = UB[tid]` each iter | scalar half store; store loop `#pragma unroll` |

Elems/thread for x: **32** (=1024/32). Scales/thread: **1**.

### Compile notes (frontend + harness)

1. Direct `out_ub = Cast(x/scale[i])` under `Parallel(R,C)` (or even serial Cast on live `scale[i]`) vectorizes into illegal `uint2{half,half,half,half}` → bisheng `-Wc++11-narrowing`.
2. Workaround: reload-shaped `s = T.alloc_var(...); s = scale[i]; Cast(x/s)` **plus** harness patch `_fix_uint2_half_pack` (same class as `int2→make_int2`) so any remaining pack is rewritten to half2 bitcasts.
3. T128 frag_live left out of oneshot for now (same pack risk / odd warp gating in earlier dump).

### Status

| item | state |
|------|--------|
| Kernel | `kernels/sv2_eltwise_bcast_rf.py` |
| Opsim | **both arms PASS** |
| Artifacts | laptop `st_simtvf_sv2_reports.tgz`; remote `/tmp/st_simtvf_parallel/sources/sv2_*_source.txt` |
| Contrast | SV6 group-scale @ R64 C128 G16 still ~19–22 µs (heavier grain) |

### What would flip the cliff next

- Force unroll on frag_live store (or match reload’s store shape more closely) to see if live RF can beat reload.
- T32 → T128; 32×32 → 64×64 (x elems/thread 32→128).
- Scale grain G=16 (bridges into SV6).

---

## Report series note

This is **report 1** of the ST series (SV1 + SV2 only). Subsequent STs (SV2…SV8 deep-dives) stay one-by-one later; do not expand this file into a full suite dump.
