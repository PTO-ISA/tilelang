# Private storage in PTO Parallel

Inside `T.SimdVF`, ordinary `T.alloc_var` and `T.alloc_local((1,), dtype)`
allocated in a Parallel unit describe independent mutable state for each
logical iteration. They use the existing language API; explicit `T.vmi`
allocations retain their existing contracts.

```python
with T.SimdVF(lanes=64):
    bias = T.alloc_local((1,), "float32")
    bias[0] = 3.1
    for i in T.Parallel(n):
        acc = T.alloc_var("float32")
        tmp = T.alloc_local((1,), "float32")
        acc = acc + a_ub[i]  # acc starts at zero for every iteration
        tmp[0] = acc
        old = tmp[0]
        tmp[0] = acc + 7.0
        out_ub[i] = old * 2.0 + tmp[0] + bias[0]
```

The result is `3 * a_ub[i] + 7 + bias[0]`. Binding `old` takes a snapshot;
subsequent stores to `tmp[0]` do not change it. The external `bias` remains
a typed scalar, read at its original position and broadcast when necessary.
External ordinary storage cannot be written inside a Parallel unit.

## Ownership and lowering

The frontend records ordinary allocations in the active SIMD frame. At frame
completion it annotates the root of each complete adjacent Parallel nest with
`tl.parallel_private_allocs`, an internal array of allocation data variables.
Only allocations associated with the innermost loop belong to that unit.
An allocation between two adjacent Parallel loops belongs to the enclosing
allocation block and is external to the complete unit. Explicit initializers
remain statements at their original positions.

`ParallelStorageQuery` associates actual allocations, owners and constant slots
by data-variable identity. Early loop cloning remaps the annotation through the
same variable map as allocation declarations and uses. Layout inference,
lane-use analysis, verification and vectorization share this query:

- Private slot reads are lane-varying even though their slot index is zero.
- External ordinary slot reads are scalar and lane-uniform.
- UB accesses retain address-based continuity analysis and determine the
  vectorized axis of a two-dimensional unit.

Vectorization creates mutable VMI storage immediately before the converted unit:
`alloc_var` becomes a `data` slot, and `alloc_local` becomes a `data_array` with
its original shape. Element dtypes gain the VF lane count. Reads become Bind
snapshots and writes replace the slot's vector value. There is no table that
substitutes a slot with its most recent expression.

Every chunk, including every row of a two-dimensional unit, initializes private
`alloc_var` with a typed zero broadcast. Explicit initializers execute afterward
in source order. `alloc_local` has no implicit initialization. Tail stores use
the existing mask; inputs must provide readable storage for complete vector
loads. For a singleton selected dimension, a uniform output address is accepted
because the tail mask enables only lane zero.

## Types and first-version limits

The first version accepts only rank-one shape `(1,)` ordinary storage, with
constant in-bounds slots. `CheckStorageV1` centralizes that restriction, while
`ResolveSlot` retains generic capacity and constant-offset representation for
future arrays with more slots. Arbitrary views, dynamic slots, private address
escapes, cross-unit private state and internal control flow are unsupported.

Storage capability and scalar-to-vector broadcast capability are separate.
Verify and Vectorize share `CheckBroadcastDType`: broadcasts support 8-, 16-
and 32-bit integers, float16, bfloat16 and float32. This applies to implicit
zero initialization, scalar slot assignments, mixed scalar/vector arithmetic
and scalar values stored to vector outputs.

Private FP8 `alloc_var` is rejected even if immediately overwritten, because
its required zero initialization cannot use the current broadcast instruction.
FP8 `alloc_local` may receive supported vector loads or casts. Assigning a
scalar FP8 value, or directly broadcasting an external FP8 scalar, is rejected.
The existing operation and cast contracts still apply. Bool broadcasts are
also unsupported.

Ordinary scalar local-array stores in PTO codegen preserve the element type at
the assignment: literals use `pto.const(value, dtype=...)`; other expressions
use `pto.cast`. This includes float16/bfloat16 rounding at storage time and
also applies outside Parallel. SIMT backing storage and VMI vector arrays use
their existing codegen paths.

The existing `VerifyBufferInit` configuration and warning behavior are retained.
Parallel verification adds no definite-initialization rejection. This does
**not** promise that uninitialized reads compile through PTODSL/PTOAS or have a
defined result. Initialize every `alloc_local` slot before reading it; this
implementation introduces neither an undef value nor automatic zero filling.

## Integration with Parallel control flow

Internal control flow remains outside this change. The control-flow proposal
in [PR #11](https://github.com/PTO-ISA/tilelang/pull/11), reviewed at
`0ae08678d01410c5f8d625af631d860b57176880`, does not automatically enable
conditional private-slot writes: its scope checks admit only UB, and its
conditional destination pairing requires a lane-dependent address. A private
slot such as `tmp[0]` has a constant index but represents independent lane state.

Integration must use `ParallelStorageQuery` in both verification and branch
legalization to recognize private destinations by allocation and slot, while
continuing to reject writes to external ordinary storage. The query must refer
to the current loop tree after legalization, preserving allocation ownership.
Scalar arms introduced by Select lowering must also use `CheckBroadcastDType`.

The control-flow legalizer fills a missing store arm with a read of the old
destination. For a private slot this must read its previous vector value;
`alloc_local` must already be initialized on that path. Integration tests need
two-arm assignments, one-arm updates, read-modify-write and a bound old value
used after a conditional overwrite, including multiple chunks and tails.
Branch-local declarations and Bind statements also need explicit admission in
the branch legalizer; this change does not promise support for them. These are
integration requirements, not completed combined-backend validation.

## Validation

Host tests in `testing/ascend/language/pto/test_pto_parallel_private_alloc.py`
cover ownership, cloning, mixed-core projection, two-dimensional layout,
allocation markers, snapshot order, scalar store typing, malformed metadata,
unsupported shapes/slots, cross-unit accesses, broadcast admission and the
initialization verification boundary.

`test_pto_parallel_private_alloc_run.py` requires a working NPU runtime. It
compiles through PTODSL/PTOAS and checks scalar broadcasts, narrow scalar
storage, FP8 vector storage/casts, snapshots, per-chunk zero initialization,
multiple units, two-dimensional layouts and tails. Numeric tests use independent
CPU references, nonuniform inputs and explicit padding/output sentinels.
Mandatory cases must be collected and executed; skips and timeouts do not
constitute successful validation.

Related regression suites cover Parallel structure/verification, pipeline
routing, early unrolling, explicit VMI value provenance and the cast matrix.
Record the TileLang commit, loaded Python/native paths, PTOAS revision and
artifact hashes with the results; use separate build and cache directories
when testing a different TileLang checkout.
