"""PTODSL tracing helpers for TileLang PTO SIMT codegen."""

from __future__ import annotations

from ptodsl import pto
from ptodsl._surface_values import unwrap_surface_value, wrap_surface_value
from ptodsl._types import (
    _integer_signedness,
    _restore_integer_signedness,
    _strip_integer_signedness,
)
from ptoas.mlir.dialects import llvm
from ptoas.mlir.dialects import pto as _pto
from ptoas.mlir.ir import F16Type, F32Type, IntegerType

__all__ = [
    "scalar_div",
    "scalar_rsqrt",
    "simt_allreduce_sum",
    "simt_allreduce_max",
    "simt_allreduce_min",
    "vectorize_binary_f32x2",
    "vectorize_unary_f32x2",
]


def _vector_lane(value, index):
    raw_index = unwrap_surface_value(pto.const(index, dtype=pto.i32))
    return wrap_surface_value(llvm.ExtractElementOp(unwrap_surface_value(value), raw_index).res)


def vectorize_unary_f32x2(op, value):
    return pto.Vec(
        pto.f32,
        2,
        init=(
            op(_vector_lane(value, 0)),
            op(_vector_lane(value, 1)),
        ),
    )


def vectorize_binary_f32x2(op, lhs, rhs):
    return pto.Vec(
        pto.f32,
        2,
        init=(
            op(_vector_lane(lhs, 0), _vector_lane(rhs, 0)),
            op(_vector_lane(lhs, 1), _vector_lane(rhs, 1)),
        ),
    )


def scalar_div(lhs, rhs):
    return lhs / rhs


def scalar_rsqrt(value):
    return 1.0 / pto.sqrt(value)


# ── SIMT cross-workitem all-reduce ─────────────────────────────────────────────
#
# All-reduce ops are emitted **inline** at the current insertion point.
# Three reducer variants: ``simt_allreduce_sum``, ``simt_allreduce_max``,
# ``simt_allreduce_min``.
#
# Dispatch tree (compile-time, since *threads* / *scale* are Python ints)::
#
#     threads <= scale                                       →  identity
#     threads ≤ 32,  pow2(threads), pow2(scale)              →  warp_reduce
#     threads ≤ 32                                           →  ub_reduce
#     threads > 32,  pow2(threads), scale≤32, pow2(scale)   →  cross_warp_reduce
#     otherwise                                              →  ub_reduce (fallback)
#
# The low-level redux and shuffle operations accept signless integer carriers
# only, while PTODSL models integer signedness on the values themselves. Wrap
# every call site so authored si32/ui32 values are stripped at the boundary and
# the reduced result is restored to the authored dtype afterward.

def _redux_integer_compat(op, value):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return op(value)
    signedness = _integer_signedness(raw_value.type)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = op(signless_value, signedness=signedness)
    return wrap_surface_value(
        _restore_integer_signedness(unwrap_surface_value(result), raw_value.type)
    )


def _redux_add(value):
    # redux_add is sign-agnostic: the op dispatches between ReduxAddIOp and
    # ReduxAddFOp by value type and takes no signedness keyword, so call it
    # without one on integer carriers.
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return pto.redux_add(value)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = pto.redux_add(signless_value)
    return wrap_surface_value(
        _restore_integer_signedness(unwrap_surface_value(result), raw_value.type)
    )


def _redux_max(value):
    return _redux_integer_compat(pto.redux_max, value)


def _redux_min(value):
    return _redux_integer_compat(pto.redux_min, value)


def _shuffle_bfly(value, offset):
    raw_value = unwrap_surface_value(value)
    if not IntegerType.isinstance(raw_value.type):
        return pto.shuffle_bfly(value, offset)
    signless_value = wrap_surface_value(_strip_integer_signedness(raw_value))
    result = pto.shuffle_bfly(signless_value, offset)
    return wrap_surface_value(
        _restore_integer_signedness(unwrap_surface_value(result), raw_value.type)
    )


def _is_pow2(n: int) -> bool:
    """Compile-time power-of-two check."""
    return n > 0 and (n & (n - 1)) == 0


def _validate_scratch_buffer(scratch, *, value_type, reducer: str, dtype: str,
                             threads: int, scale: int, thread_offset: int) -> None:
    context = f"all_reduce {reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset}"
    raw_scratch = unwrap_surface_value(scratch)
    try:
        scratch_type = _pto.PtrType(raw_scratch.type)
    except Exception as exc:
        raise TypeError(f"{context} requires a UB scratch buffer pointer, got {raw_scratch.type}") from exc
    if scratch_type.element_type != value_type:
        raise TypeError(
            f"{context} scratch element type mismatch: expected {value_type}, got {scratch_type.element_type}"
        )
    memory_space = getattr(scratch_type, "memory_space", None)
    vec_attr = _pto.AddressSpaceAttr.get(_pto.AddressSpace.VEC)
    memory_space_value = getattr(memory_space, "value", memory_space)
    scratch_type_text = str(scratch_type)
    if (
        memory_space != vec_attr
        and memory_space_value != _pto.AddressSpace.VEC
        and ", ub>" not in scratch_type_text
        and ", vec>" not in scratch_type_text
    ):
        raise TypeError(f"{context} requires a UB scratch buffer, got {scratch_type}")


# ── reducer dispatch tables ────────────────────────────────────────────────────

_REDUCER_IDENTITY = {
    "sum": {"f32": 0.0, "f16": 0.0, "si32": 0, "ui32": 0},
    "max": {
        "f32": float("-inf"), "f16": float("-inf"),
        "si32": -(2 ** 31), "ui32": 0,
    },
    "min": {
        "f32": float("inf"), "f16": float("inf"),
        "si32": 2 ** 31 - 1, "ui32": 2 ** 32 - 1,
    },
}

_REDUCER_COMBINE = {
    "sum": lambda a, b: a + b,
    "max": pto.max,
    "min": pto.min,
}

_REDUCER_REDUX = {
    "sum": _redux_add,
    "max": _redux_max,
    "min": _redux_min,
}

_REDUCER_IDENTITY_DTYPE = {
    "f32": pto.float32, "f16": pto.float16,
    "si32": pto.si32, "ui32": pto.ui32,
}


# ── butterfly  ──────────────────────────────────────────────────────────────────

def _emit_butterfly(v, *, threads: int, scale: int, reducer: str):
    """Unrolled butterfly shuffle reduce."""
    combine = _REDUCER_COMBINE[reducer]
    cur = threads
    while cur > scale:
        offset = cur // 2
        v = combine(v, _shuffle_bfly(v, offset))
        cur //= 2
    return v


# ── warp_hw_reduce  ────────────────────────────────────────────────────────────

def _emit_warp_hw_reduce(x, *, threads: int, lane_in_warp, dtype: str, reducer: str):
    """Warp-level hardware reduce with group masking."""
    redux_fn = _REDUCER_REDUX[reducer]
    groups = 32 // threads

    if groups == 1:
        return redux_fn(x)

    c_identity = pto.const(
        _REDUCER_IDENTITY[reducer][dtype],
        dtype=_REDUCER_IDENTITY_DTYPE[dtype],
    )
    my_group = lane_in_warp // threads

    for g in range(groups):
        in_group = my_group == g
        masked = pto.select(in_group, x, c_identity)
        reduced = redux_fn(masked)
        x = pto.select(in_group, reduced, x)
    return x


# ── warp_reduce  ───────────────────────────────────────────────────────────────

def _emit_warp_reduce(x, *,
                      dtype, threads, scale, thread_offset, reducer):
    """Single-warp all-reduce."""
    extent = threads // scale
    if extent <= 1:
        return x

    if thread_offset:
        lane_in_warp = (pto.get_tid_x() - thread_offset) & 31
    else:
        lane_in_warp = pto.get_laneid()

    if extent >= 16 and scale == 1:
        return _emit_warp_hw_reduce(
            x, threads=threads,
            lane_in_warp=lane_in_warp, dtype=dtype, reducer=reducer,
        )
    return _emit_butterfly(x, threads=threads, scale=scale, reducer=reducer)


# ── cross_warp_reduce  ─────────────────────────────────────────────────────────

def _simt_warp_indexing(*, thread_offset):
    """Derive global tx, warp id, and warp-local lane id for this work-item."""
    tid_x = pto.get_tid_x()
    if thread_offset:
        tx = tid_x - thread_offset
        wid = tx // 32
        lid = tx & 31
    else:
        tx = tid_x
        wid = tx // 32
        lid = pto.get_laneid()
    return tx, wid, lid


def _warp_partial_reduce(x, *, scale: int, reducer: str):
    """Reduce the warp-local value down to ``scale`` partials per warp."""
    if scale == 1:
        return _REDUCER_REDUX[reducer](x)
    return _emit_butterfly(x, threads=32, scale=scale, reducer=reducer)


def _write_warp_partials(warp_val, scratch, *, wid, lid, scale: int) -> None:
    """Have each warp's ``scale`` leaders store their partial result."""
    is_writer = lid < scale
    with pto.if_(is_writer) as br:
        with br.then_:
            slot = wid * scale + lid
            pto.store(warp_val, scratch, slot)


def _combine_warp_partials(lid, scratch, *, num_warps: int, scale: int,
                           reducer: str, identity):
    """Combine every warp's partials on warp 0 (leader lanes only).

    The caller wraps the emitted ops in the runtime leader-warp region; the
    ``num_warps``/``scale``/``reducer`` dispatch is resolved at trace time.
    """
    if scale == 1:
        loaded = pto.select(
            lid < num_warps,
            pto.load(scratch, lid),
            identity,
        )
        return _REDUCER_REDUX[reducer](loaded)

    total = scale * num_warps
    if total <= 32:
        loaded = pto.select(
            lid < total,
            pto.load(scratch, lid),
            identity,
        )
        return _emit_butterfly(
            loaded, threads=total, scale=scale, reducer=reducer,
        )

    combine = _REDUCER_COMBINE[reducer]
    is_reducer = lid < scale
    reduced = identity
    my_slot = lid % scale
    for w in range(num_warps):
        idx_val = w * scale + my_slot
        loaded_v = pto.load(scratch, idx_val)
        reduced = combine(reduced, loaded_v)
    return pto.select(is_reducer, reduced, identity)


def _leader_warp_reduce(
    tx, lid, scratch, *, num_warps: int, scale: int, reducer: str, identity
):
    """Let warp 0 combine every warp's partial into the cross-warp result."""
    is_leader_warp = tx < 32
    with pto.if_(is_leader_warp) as br:
        with br.then_:
            br.assign(stage4_result=_combine_warp_partials(
                lid,
                scratch,
                num_warps=num_warps,
                scale=scale,
                reducer=reducer,
                identity=identity,
            ))
        with br.else_:
            br.assign(stage4_result=identity)

    return br.stage4_result


def _broadcast_warp_result(partial_reduced, scratch, *, tx, scale: int):
    """Store the final partial and load the value broadcast to all lanes."""
    is_global_leader = tx < scale
    with pto.if_(is_global_leader) as br5:
        with br5.then_:
            pto.store(partial_reduced, scratch, tx)

    pto.syncthreads()
    result = pto.load(scratch, tx % scale)
    pto.syncthreads()
    return result


def _emit_cross_warp_reduce(x, scratch, *,
                            dtype, threads, scale, thread_offset, reducer):
    """Cross-warp all-reduce (threads > 32)."""
    num_warps = threads // 32
    c_identity = pto.const(
        _REDUCER_IDENTITY[reducer][dtype],
        dtype=_REDUCER_IDENTITY_DTYPE[dtype],
    )
    tx, wid, lid = _simt_warp_indexing(thread_offset=thread_offset)
    warp_val = _warp_partial_reduce(x, scale=scale, reducer=reducer)
    _write_warp_partials(warp_val, scratch, wid=wid, lid=lid, scale=scale)

    pto.syncthreads()

    partial_reduced = _leader_warp_reduce(
        tx,
        lid,
        scratch,
        num_warps=num_warps,
        scale=scale,
        reducer=reducer,
        identity=c_identity,
    )

    return _broadcast_warp_result(
        partial_reduced, scratch, tx=tx, scale=scale,
    )


# ── ub_reduce  ─────────────────────────────────────────────────────────────────

def _emit_ub_reduce(x, scratch, *,
                    dtype, threads, scale, thread_offset, reducer):
    """UB-scratch all-reduce (fallback for non-pow2 or general case)."""
    combine = _REDUCER_COMBINE[reducer]

    # ── thread indexing ──────────────────────────────────────────────────
    tid_x = pto.get_tid_x()
    tx = (tid_x - thread_offset) if thread_offset else tid_x
    group = tx // threads
    lane = tx % threads

    # ── each lane writes x → scratch[tx] ─────────────────────────────────
    pto.store(x, scratch, tx)
    pto.syncthreads()

    # ── reducers sequentially combine ────────────────────────────────────
    is_reducer = pto.cmp(lane, scale, "lt")
    with pto.if_(is_reducer) as br:
        with br.then_:
            group_offset = group * threads
            first_elem = group_offset + lane
            acc = pto.load(scratch, first_elem)

            carry_loop = pto.for_(scale, threads, step=scale).carry(acc=acc)
            with carry_loop:
                prev = carry_loop.acc
                elem = first_elem + carry_loop.iv
                loaded = pto.load(scratch, elem)
                carry_loop.update(acc=combine(prev, loaded))
            acc = carry_loop.final("acc")

            br.assign(flag=acc)
        with br.else_:
            br.assign(flag=x)

    flag = br.flag
    pto.syncthreads()

    # ── per-class leader writes back ─────────────────────────────────────
    is_leader = pto.cmp(lane, scale, "lt")
    with pto.if_(is_leader) as br5:
        with br5.then_:
            pto.store(flag, scratch, group * threads + lane)

    # ── broadcast ────────────────────────────────────────────────────────
    pto.syncthreads()
    result = pto.load(scratch, group * threads + (tx % scale))
    pto.syncthreads()

    return result


# ── public API  ────────────────────────────────────────────────────────────────

def _check_params(*, threads, scale, thread_offset):
    """Validate allreduce parameters (compile-time checks)."""
    for name, val in (("threads", threads), ("scale", scale),
                       ("thread_offset", thread_offset)):
        if not isinstance(val, int):
            raise ValueError(
                f"all_reduce: '{name}' must be a Python int, "
                f"got {type(val).__name__}"
            )
    if threads < 1:
        raise ValueError(f"all_reduce: threads must be >= 1, got {threads}")
    if scale < 1:
        raise ValueError(f"all_reduce: scale must be >= 1, got {scale}")
    if thread_offset < 0:
        raise ValueError(
            f"all_reduce: thread_offset must be >= 0, got {thread_offset}"
        )
    if threads % scale != 0:
        raise ValueError(
            f"all_reduce requires threads % scale == 0; "
            f"got threads={threads}, scale={scale}"
        )


def _simt_allreduce(value, *, threads, scale, thread_offset, scratch, reducer):
    """Unified allreduce dispatch tree."""
    _check_params(threads=threads, scale=scale, thread_offset=thread_offset)

    if threads <= scale:
        return value

    raw_value = unwrap_surface_value(value)
    if raw_value.type == F32Type.get():
        dtype = "f32"
    elif raw_value.type == F16Type.get():
        dtype = "f16"
    elif raw_value.type == IntegerType.get_signed(32):
        dtype = "si32"
    elif raw_value.type == IntegerType.get_unsigned(32):
        dtype = "ui32"
    else:
        raise NotImplementedError(f"all_reduce: unsupported dtype {raw_value.type}")

    args = dict(dtype=dtype, threads=threads, scale=scale,
                thread_offset=thread_offset, reducer=reducer)

    if threads <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_warp_reduce(value, **args)

    if scratch is None:
        raise ValueError(
            f"all_reduce {reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset} "
            "requires a UB scratch buffer"
        )
    _validate_scratch_buffer(
        scratch,
        value_type=raw_value.type,
        reducer=reducer,
        dtype=dtype,
        threads=threads,
        scale=scale,
        thread_offset=thread_offset,
    )

    if threads <= 32:
        return _emit_ub_reduce(value, scratch, **args)

    if scale <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_cross_warp_reduce(value, scratch, **args)

    return _emit_ub_reduce(value, scratch, **args)


def simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Sum reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale,
                           thread_offset=thread_offset, scratch=scratch, reducer="sum")


def simt_allreduce_max(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Max reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale,
                           thread_offset=thread_offset, scratch=scratch, reducer="max")


def simt_allreduce_min(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Min reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale,
                           thread_offset=thread_offset, scratch=scratch, reducer="min")
