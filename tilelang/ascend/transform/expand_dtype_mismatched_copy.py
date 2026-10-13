"""ExpandDtypeMismatchedCopy — expand the PTO TopK i32->i64 index copy.

The Ascend frontend spells ``T.copy(ub_i32, gm_i64)`` as a
``tl.tileop.ascend_copy`` over two ``tl.region`` operands (see
``tilelang/ascend/language/copy_op.py``).  Ascend DMA cannot cast, so on the
PTO route this pass runs after ``InsertNd2Nz`` and before ``VFChecker`` and
rewrites the widening copy into

  1. an int64 ``shared.dyn`` scratch allocation;
  2. a bare ``SIMD_VF`` block (no ``T.Parallel``) that widens elementwise.
     K in ``{2,4,8,64}`` uses the signed VMI path -- ``vcmps`` selects the
     -1/0 high half, ``vintlv`` interleaves the int32 halves into the scratch
     through a same-storage int32 view; below K=64 a compiler-owned 64-element
     pad is filled by two split loops (copy ``[0,K)``, zero ``[K,64)``) so no
     out-of-range load is ever guarded by a ``Select``.  Every other legal
     static K keeps the signed serial ``Cast``.  No zext proof is attempted:
     an element-wise assume never widens the accepted signedness;
  3. a same-dtype ``tl.tileop.ascend_copy`` (int64 scratch -> the original
     int64 GM destination region), which the C++ side parses as an
     ``AscendCopyNode``.

Admission is deliberately narrow: only ``int32`` shared (``shared`` /
``shared.dyn``, implicit-contiguous or provably unit stride) -> ``int64``
``global`` copies with static, equal, one-dimensional regions are rewritten.
Region bounds must be provable either directly or from the lexical loop
ranges that enclose the copy (``for tile in serial(0, 2): copy(dst[tile*8 :
tile*8+8])`` with capacity 16 is in bounds); unprovable or out-of-bounds
regions still get a specific ``ValueError``.

Diagnostics apply only to that target pattern.  Other copies (GM source,
non-GM destination, other dtype pairs, same dtype, scalar stores) are left
untouched for their existing handlers, regardless of extra operands.  A
target copy with a third operand (e.g. an MX scale) is rejected, never
truncated.

Copy annotations are preserved with source-referencing metadata remapped onto
the scratch: the original source region base is subtracted, so a
``BufferRegion``/``tl.region``/``BufferLoad`` that described ``src[3:11]``
describes ``scratch[0:8]`` afterwards.  ``Buffer``, ``data`` Var, and nested
``Map``/``Array`` containers are remapped recursively.  Metadata kinds the
pass cannot interpret are rejected with a diagnostic instead of being copied
blindly; see ``issues/issue-int64-task1-opaque-metadata.md``.
"""

from __future__ import annotations

from tvm import arith, tirx
from tvm import ir as tvm_ir
from tvm import runtime as tvm_runtime
from tvm.tirx import (
    Bind,
    BufferLoad,
    BufferRegion,
    BufferStore,
    Call,
    Evaluate,
    For,
    ForKind,
    SBlock,
    SeqStmt,
)
from tvm.tirx.stmt_functor import ir_transform
from tvm.tirx.transform import prim_func_pass

_COPY_OP = "tl.tileop.ascend_copy"
_REGION_OP = "tl.region"
_ACCESS_PTR_OP = "tl.access_ptr"
_VF_HINTS = ("SIMD_VF", "SIMT_VF")
_SCRATCH_SCOPE = "shared.dyn"
_LEAF_TYPES = (tirx.IntImm, tirx.FloatImm, tirx.StringImm)
# VMI widen admission (Task 2): every other legal static K uses the serial
# signed fallback.
_WIDEN_VMI_K = (2, 4, 8, 64)
_WIDEN_VL = 64


def _op_name(node):
    op = getattr(node, "op", None)
    return getattr(op, "name", None)


def _scope(buf) -> str:
    scope = getattr(buf, "scope", "")
    return scope() if callable(scope) else scope


def _is_gm(buf) -> bool:
    return _scope(buf) in ("", "global")


def _is_shared(buf) -> bool:
    return _scope(buf) in ("shared", "shared.dyn")


def _is_target_pair(src_buf, dst_buf) -> bool:
    return str(src_buf.dtype) == "int32" and str(dst_buf.dtype) == "int64"


def _region_parts(arg):
    """Return ``(buffer, load, extents)`` for a ``tl.region`` call, else None."""
    if not (isinstance(arg, Call) and _op_name(arg) == _REGION_OP):
        return None
    load = arg.args[0]
    if not isinstance(load, BufferLoad):
        return None
    extents = list(arg.args[2:]) if len(arg.args) > 2 else list(load.buffer.shape)
    return load.buffer, load, extents


def _static_extent(ext):
    return int(ext.value) if isinstance(ext, tirx.IntImm) else None


def _unit_stride(buf) -> bool:
    strides = getattr(buf, "strides", None)
    if strides is None or len(strides) == 0:
        return True
    if len(strides) != 1:
        return False
    stride = strides[0]
    return bool(arith.Analyzer().can_prove_equal(stride, tirx.const(1, stride.dtype)))


def _reject(msg: str):
    raise ValueError(f"ExpandDtypeMismatchedCopy: {msg}")


def _container_kind(value):
    """Recognize the runtime ``Map``/``Array`` containers (tvm_ffi)."""
    name = type(value).__name__
    if name == "Map" or isinstance(value, dict):
        return "map"
    if name == "Array" or isinstance(value, (list, tuple)):
        return "array"
    return None


def _refs_source(value, src_buf) -> bool:
    """Does *value* reference the int32 source in any recognized form?

    Identity is not enough: an alias that shares the source ``data`` Var, a
    nested index/bound expression containing ``src[...]``, and container keys
    all count.
    """
    if isinstance(value, tirx.Buffer):
        if value.same_as(src_buf) or value.data.same_as(src_buf.data):
            return True
        if any(_refs_source(expr, src_buf) for expr in value.shape):
            return True
        strides = getattr(value, "strides", None)
        if strides is not None and any(_refs_source(expr, src_buf) for expr in strides):
            return True
        return _refs_source(value.elem_offset, src_buf)
    if isinstance(value, tirx.Var):
        return value.same_as(src_buf.data)
    if isinstance(value, BufferRegion):
        if _refs_source(value.buffer, src_buf):
            return True
        return any(
            _refs_source(r.min, src_buf) or _refs_source(r.extent, src_buf) for r in value.region
        )
    if isinstance(value, Call):
        return any(_refs_source(arg, src_buf) for arg in value.args)
    if isinstance(value, tirx.PrimExpr):
        hits = []

        def visit(node):
            if isinstance(node, tirx.Var):
                if node.same_as(src_buf.data):
                    hits.append(node)
            elif isinstance(node, BufferLoad) and _refs_source(node.buffer, src_buf):
                hits.append(node)

        tirx.stmt_functor.post_order_visit(value, visit)
        return bool(hits)
    kind = _container_kind(value)
    if kind == "map":
        return any(
            _refs_source(key, src_buf) or _refs_source(val, src_buf) for key, val in value.items()
        )
    if kind == "array":
        return any(_refs_source(item, src_buf) for item in value)
    return False


def _unsupported_source_ref(value, what: str):
    raise ValueError(
        f"ExpandDtypeMismatchedCopy: metadata {what} still references the int32 "
        f"source and cannot be remapped (kind '{type(value).__name__}'); "
        "refusing to copy it blindly (see issue-int64-task1-opaque-metadata)"
    )


def _shift_index(index, src_buf, src_base, simplify):
    """Region-relative index: subtract the original source region base."""
    if _refs_source(index, src_buf):
        _unsupported_source_ref(index, "region bound")
    return simplify(index - src_base)


def _static_int(expr):
    return int(expr.value) if isinstance(expr, tirx.IntImm) else None


def _require_scratch_window(min_expr, extent, scratch_k, what: str):
    """The remapped window ``[min, min + extent)`` must lie inside the int64
    scratch ``[0, K)``; otherwise reject instead of emitting metadata that
    describes elements the scratch never holds (no enlargement, no clipping)."""
    if scratch_k is None:
        raise ValueError(
            f"ExpandDtypeMismatchedCopy: metadata {what} cannot be checked "
            "against a non-static int64 scratch; refusing to emit unproven "
            "metadata (see issue-int64-task1-opaque-metadata)"
        )
    extent_int = _static_int(extent)
    analyzer = arith.Analyzer()
    fits = False
    if extent_int is not None:
        fits = bool(
            analyzer.can_prove(min_expr >= 0)
            and analyzer.can_prove(min_expr + tirx.const(extent_int, min_expr.dtype) <= scratch_k)
        )
    if not fits:
        raise ValueError(
            f"ExpandDtypeMismatchedCopy: metadata {what} remaps to window "
            f"[{min_expr}, +{extent}) outside the int64 scratch [0, {scratch_k}); "
            "refusing to emit out-of-range metadata "
            "(see issue-int64-task1-opaque-metadata)"
        )


def _remap_value(value, src_buf, scratch, src_base, simplify):
    """Re-point one metadata value (or Map key) at the scratch.

    Supported source references are shifted by the original source region
    base, so the metadata describes the same copied elements.  A reference in
    a form this pass does not understand (alias storage, nested index
    expressions, opaque calls) is rejected instead of being copied blindly.
    """
    if value is None or isinstance(value, (bool, int, float, str)) or isinstance(value, _LEAF_TYPES):
        return value
    scratch_k = _static_int(scratch.shape[0])
    if isinstance(value, tirx.Buffer):
        if value.same_as(src_buf):
            return scratch
        if value.data.same_as(src_buf.data):
            _unsupported_source_ref(value, "alias buffer sharing the source storage")
        if _refs_source(value, src_buf):
            _unsupported_source_ref(
                value, "buffer whose shape/strides/elem_offset reference the source"
            )
        return value
    if isinstance(value, tirx.Var):
        return scratch.data if value.same_as(src_buf.data) else value
    if isinstance(value, BufferRegion):
        if not _refs_source(value, src_buf):
            return value
        if not value.buffer.same_as(src_buf):
            _unsupported_source_ref(
                value, "region over another buffer with a nested source reference"
            )
        for r in value.region:
            if _refs_source(r.extent, src_buf):
                _unsupported_source_ref(r.extent, "region extent")
        regions = []
        for r in value.region:
            mapped_min = _shift_index(r.min, src_buf, src_base, simplify)
            _require_scratch_window(mapped_min, r.extent, scratch_k, "region")
            regions.append(tvm_ir.Range.from_min_extent(mapped_min, r.extent))
        return BufferRegion(scratch, regions)
    if isinstance(value, Call) and _op_name(value) == _REGION_OP:
        if not _refs_source(value, src_buf):
            return value
        parts = _region_parts(value)
        if parts is None or not parts[0].same_as(src_buf):
            _unsupported_source_ref(value, "region call with a nested source reference")
        extents = list(parts[2])
        for ext in extents:
            if _refs_source(ext, src_buf):
                _unsupported_source_ref(ext, "region extent")
        indices = list(parts[1].indices)
        indices[0] = _shift_index(indices[0], src_buf, src_base, simplify)
        for extra in indices[1:]:
            if _refs_source(extra, src_buf):
                _unsupported_source_ref(extra, "region index")
        window_extent = extents[0] if extents else src_buf.shape[0]
        _require_scratch_window(indices[0], window_extent, scratch_k, "region call")
        return Call("handle", value.op, [BufferLoad(scratch, indices)] + list(value.args[1:]))
    if isinstance(value, Call):
        if _refs_source(value, src_buf):
            _unsupported_source_ref(value, f"opaque call 'Call({_op_name(value)})'")
        raise ValueError(
            "ExpandDtypeMismatchedCopy: unsupported metadata kind "
            f"'Call({_op_name(value)})' on a widened copy; it is not copied "
            "blindly (see issue-int64-task1-opaque-metadata)"
        )
    if isinstance(value, BufferLoad):
        if not _refs_source(value, src_buf):
            return value
        if not value.buffer.same_as(src_buf):
            _unsupported_source_ref(
                value, "load over another buffer with a nested source reference"
            )
        indices = list(value.indices)
        indices[0] = _shift_index(indices[0], src_buf, src_base, simplify)
        for extra in indices[1:]:
            if _refs_source(extra, src_buf):
                _unsupported_source_ref(extra, "load index")
        _require_scratch_window(indices[0], tirx.const(1, "int32"), scratch_k, "load")
        return BufferLoad(scratch, indices)
    kind = _container_kind(value)
    if kind == "map":
        return tvm_runtime.convert(
            {
                _remap_value(key, src_buf, scratch, src_base, simplify): _remap_value(
                    val, src_buf, scratch, src_base, simplify
                )
                for key, val in value.items()
            }
        )
    if kind == "array":
        return tvm_runtime.convert(
            [_remap_value(item, src_buf, scratch, src_base, simplify) for item in value]
        )
    if _refs_source(value, src_buf):
        _unsupported_source_ref(value, "value")
    raise ValueError(
        "ExpandDtypeMismatchedCopy: unsupported metadata kind "
        f"'{type(value).__name__}' on a widened copy; it is not copied blindly "
        "(see issue-int64-task1-opaque-metadata)"
    )


def _loop_constraints(state):
    """Active lexical loop domains, as analyzer constraints."""
    constraints = []
    for loop in state["loops"]:
        low = _static_extent(loop.min)
        extent = _static_extent(loop.extent)
        if low is None or extent is None:
            continue
        var = loop.loop_var
        constraints.append((var >= low) & (var < low + extent))
    return constraints


def _prove(analyzer, constraints, predicate) -> bool:
    """Prove *predicate* under the active lexical loop constraints."""

    def prove_at(index):
        if index >= len(constraints):
            return bool(analyzer.can_prove(predicate))
        with analyzer.constraint_scope(constraints[index]):
            return prove_at(index + 1)

    try:
        return prove_at(0)
    except Exception:  # noqa: BLE001 - unprovable is a rejection, not a crash
        return False


def _emit_serial_widen(src_buf, src_index, scratch, extent: int):
    """Signed serial widen baseline; the reverse index keeps the legacy
    MergeUB-style alias analysis safe (no overlapping R/W at the same index)."""
    i = tirx.Var("cast_i", "int32")
    rev = tirx.const(extent - 1, "int32") - i
    loaded = BufferLoad(src_buf, [src_index + rev])
    casted = tirx.Cast(scratch.dtype, loaded)
    store = tirx.BufferStore(scratch, casted, [rev])
    return For(
        i,
        tirx.const(0, "int32"),
        tirx.const(extent, "int32"),
        ForKind.SERIAL,
        store,
        annotations={"tl.ub_force_loop_live": tirx.const(1, "int32")},
    )


def _vmi(op_tail: str, dtype, args, annotations=None):
    return Call(dtype, tvm_ir.Op.get(f"tl.vmi.{op_tail}"), list(args), annotations or {})


def _element_ptr(buf, index, extent, rw_mask):
    """``tl.access_ptr`` over one buffer element; *extent* counts elements."""
    load = BufferLoad(buf, [index])
    return Call(
        "handle",
        tvm_ir.Op.get(_ACCESS_PTR_OP),
        [load, tirx.const(int(extent), "int32"), tirx.const(rw_mask, "int32")],
    )


def _emit_vmi_widen(src_buf, src_index, scratch, extent, pending_allocs):
    """Signed widen via ``vcmps``/``vsel`` + ``vintlv`` (admission {2,4,8,64}).

    The high int32 half is 0 or -1 selected from ``lo < 0`` (sext; no zext
    proof is attempted).  K < 64 first materialises a compiler-owned 64-element
    pad through two split loops -- a copy loop over ``[0, K)`` that only reads
    valid source elements, then a zero fill over ``[K, 64)`` -- so no
    out-of-range load is ever guarded by a ``Select``.  The interleaved halves
    are stored through an int32 view of the int64 scratch (no si64 VReg);
    K = 64 writes 128 halves via two stores and exercises both ``vintlv``
    outputs.
    """
    n = extent
    vl = _WIDEN_VL
    vec_dt = f"int32x{vl}"
    mask_dt = f"boolx{vl}"
    size_ann = {"size": tirx.const(vl, "int32")}

    scratch_i32 = tirx.decl_buffer(
        [2 * n],
        dtype="int32",
        name=f"{scratch.name}_i32view",
        data=scratch.data,
        scope=_SCRATCH_SCOPE,
    )

    pad = None
    if n < vl:
        pad = tirx.decl_buffer(
            [vl],
            dtype=src_buf.dtype,
            name=f"{src_buf.name}_widen_pad",
            scope=_SCRATCH_SCOPE,
        )
        pending_allocs.append(pad)

    steps = []
    steps.append(
        ("mask", lambda e: _vmi("create_mask", mask_dt, [tirx.const(vl, "int32")], size_ann))
    )
    if pad is None:
        steps.append(("src_ptr", lambda e: _element_ptr(src_buf, src_index, vl, 1)))
    else:
        steps.append(("src_ptr", lambda e: _element_ptr(pad, tirx.const(0, "int32"), vl, 1)))
    steps.append(
        ("lo", lambda e: _vmi("vload", vec_dt, [e["src_ptr"], tirx.const(0, "int32")], size_ann))
    )
    steps.append(
        (
            "cmp",
            lambda e: _vmi(
                "vcmps",
                mask_dt,
                [e["lo"], tirx.const(0, "int32"), e["mask"], tirx.StringImm("lt")],
            ),
        )
    )
    steps.append(("ones", lambda e: _vmi("vbrc", vec_dt, [tirx.const(-1, "int32")], size_ann)))
    steps.append(("zeros", lambda e: _vmi("vbrc", vec_dt, [tirx.const(0, "int32")], size_ann)))
    steps.append(("hi", lambda e: _vmi("vsel", vec_dt, [e["cmp"], e["ones"], e["zeros"]])))
    steps.append(("pair", lambda e: _vmi("vintlv", vec_dt, [e["lo"], e["hi"], e["mask"]])))
    steps.append(("a0", lambda e: _vmi("pair_get", vec_dt, [e["pair"], tirx.const(0, "int32")])))
    steps.append(("a1", lambda e: _vmi("pair_get", vec_dt, [e["pair"], tirx.const(1, "int32")])))
    store_lanes = 2 * n
    steps.append(
        (
            "store_mask",
            lambda e: _vmi(
                "create_mask",
                mask_dt,
                [tirx.const(min(store_lanes, vl), "int32")],
                size_ann,
            ),
        )
    )
    steps.append(
        ("dst_ptr0", lambda e: _element_ptr(scratch_i32, tirx.const(0, "int32"), min(store_lanes, vl), 2))
    )
    if store_lanes > vl:
        steps.append(
            (
                "store_mask1",
                lambda e: _vmi(
                    "create_mask",
                    mask_dt,
                    [tirx.const(store_lanes - vl, "int32")],
                    size_ann,
                ),
            )
        )
        steps.append(
            ("dst_ptr1", lambda e: _element_ptr(scratch_i32, tirx.const(vl, "int32"), store_lanes - vl, 2))
        )

    env_vars = {}

    class _Env:
        def __getitem__(self, key):
            return env_vars[key]

    fake = _Env()
    computed = []
    for name, maker in steps:
        val = maker(fake)
        var = tirx.Var(name, val.dtype)
        env_vars[name] = var
        computed.append((var, val))

    stmts = [Bind(var, val) for var, val in computed]
    stmts.append(
        Evaluate(
            _vmi(
                "vstore",
                "void",
                [
                    env_vars["a0"],
                    env_vars["dst_ptr0"],
                    tirx.const(0, "int32"),
                    env_vars["store_mask"],
                ],
            )
        )
    )
    if store_lanes > vl:
        stmts.append(
            Evaluate(
                _vmi(
                    "vstore",
                    "void",
                    [
                        env_vars["a1"],
                        env_vars["dst_ptr1"],
                        tirx.const(0, "int32"),
                        env_vars["store_mask1"],
                    ],
                )
            )
        )

    prefix = []
    if pad is not None:
        i = tirx.Var("widen_pad_i", "int32")
        prefix.append(
            For(
                i,
                tirx.const(0, "int32"),
                tirx.const(n, "int32"),
                ForKind.SERIAL,
                BufferStore(pad, BufferLoad(src_buf, [src_index + i]), [i]),
            )
        )
        z = tirx.Var("widen_zero_i", "int32")
        prefix.append(
            For(
                z,
                tirx.const(n, "int32"),
                tirx.const(vl - n, "int32"),
                ForKind.SERIAL,
                BufferStore(pad, tirx.const(0, "int32"), [z]),
            )
        )
    return SeqStmt(prefix + stmts)


def _rewrite_copy(value, state):
    if not (isinstance(value, Call) and _op_name(value) == _COPY_OP):
        return None
    if len(value.args) < 2:
        return None
    src = _region_parts(value.args[0])
    dst = _region_parts(value.args[1])
    if src is None or dst is None:
        return None
    src_buf, src_load, src_ext = src
    dst_buf, dst_load, dst_ext = dst
    # Scope eligibility comes first: only the target ub -> GM pattern is this
    # pass's rewrite/rejection surface.  GM sources, non-GM destinations and
    # other dtype pairs stay untouched, extra operands included.
    if not _is_target_pair(src_buf, dst_buf):
        return None
    if not _is_shared(src_buf):
        return None
    if not _is_gm(dst_buf):
        return None
    if len(value.args) != 2:
        _reject(
            "a target widening copy must have exactly two region operands; "
            "extra operands (e.g. an MX scale) are not truncated"
        )
    if state["vf_depth"] > 0:
        _reject(
            "the widening copy is already inside a SIMD_VF/SIMT_VF block; it "
            "must be expanded before the VF is formed"
        )
    if not _unit_stride(src_buf):
        _reject("source must be implicitly contiguous or provably unit stride")
    if len(src_ext) != 1 or len(dst_ext) != 1:
        _reject("only 1-D regions are supported in the first version")
    src_k = _static_extent(src_ext[0])
    dst_k = _static_extent(dst_ext[0])
    if src_k is None or dst_k is None:
        _reject("region extents must be static (compile-time constants)")
    if src_k <= 0 or dst_k <= 0:
        _reject("region extents must be positive")
    if src_k != dst_k:
        _reject(f"source and destination extents must match, got {src_k} and {dst_k}")

    analyzer = arith.Analyzer()
    constraints = _loop_constraints(state)
    for label, buf, load, k in (
        ("source", src_buf, src_load, src_k),
        ("destination", dst_buf, dst_load, dst_k),
    ):
        base = load.indices[0]
        fits = base + tirx.const(k, base.dtype) <= buf.shape[0]
        if not (_prove(analyzer, constraints, base >= 0) and _prove(analyzer, constraints, fits)):
            _reject(f"{label} region [{base}, {base} + {k}) exceeds the buffer bounds")

    scratch = tirx.decl_buffer(
        [src_k],
        dtype=dst_buf.dtype,
        name=f"{src_buf.name}_as_wide",
        scope=_SCRATCH_SCOPE,
    )
    state["alloc"][-1].append(scratch)

    use_vmi = src_k in _WIDEN_VMI_K
    if use_vmi:
        widen = _emit_vmi_widen(
            src_buf, src_load.indices[0], scratch, src_k, state["alloc"][-1]
        )
    else:
        widen = _emit_serial_widen(src_buf, src_load.indices[0], scratch, src_k)
    vf = SBlock(
        [],
        [BufferRegion(src_buf, [tvm_ir.Range.from_min_extent(src_load.indices[0], src_ext[0])])],
        [BufferRegion(scratch, [tvm_ir.Range.from_min_extent(tirx.const(0, "int32"), src_ext[0])])],
        "SIMD_VF",
        widen,
        None,
        [],
        [],
        {
            "tl.simdvf_lanes": tirx.const(128, "int64"),
            "tl.widen_vmi": tirx.const(1 if use_vmi else 0, "int32"),
        },
    )

    scratch_load = BufferLoad(scratch, [tirx.const(0, "int32")])
    src_region = Call(
        "handle", tvm_ir.Op.get(_REGION_OP), [scratch_load, tirx.const(1, "int32"), src_ext[0]]
    )
    dst_region = Call(
        "handle", tvm_ir.Op.get(_REGION_OP), [dst_load, tirx.const(2, "int32")] + list(dst_ext)
    )
    annotations = getattr(value, "annotations", None)
    new_annotations = None
    if annotations:
        simplify = arith.Analyzer().simplify
        new_annotations = {
            key: _remap_value(val, src_buf, scratch, src_load.indices[0], simplify)
            for key, val in annotations.items()
        }
    new_copy = Call(
        "handle",
        tvm_ir.Op.get(_COPY_OP),
        [src_region, dst_region],
        annotations=new_annotations,
    )
    return SeqStmt([vf, Evaluate(new_copy)])


def ExpandDtypeMismatchedCopy():
    """Expand integer-widening ``tl.tileop.ascend_copy`` (i32 UB -> i64 GM)."""

    def pass_fn(func, mod, ctx):
        state: dict = {"alloc": [], "vf_depth": 0, "loops": []}

        def pre(node):
            if isinstance(node, SBlock):
                state["alloc"].append([])
                if node.name_hint in _VF_HINTS:
                    state["vf_depth"] += 1
            elif isinstance(node, For):
                state["loops"].append(node)
            return None

        def post(node):
            if isinstance(node, Evaluate):
                if state["alloc"]:
                    rewritten = _rewrite_copy(node.value, state)
                    return rewritten if rewritten is not None else node
                return node
            if isinstance(node, For):
                state["loops"].pop()
                return node
            if isinstance(node, SBlock):
                if node.name_hint in _VF_HINTS:
                    state["vf_depth"] -= 1
                pending = state["alloc"].pop()
                if pending:
                    return SBlock(
                        node.iter_vars,
                        node.reads,
                        node.writes,
                        node.name_hint,
                        node.body,
                        node.init,
                        list(node.alloc_buffers) + pending,
                        node.match_buffers,
                        node.annotations,
                    )
                return node
            return node

        return func.with_body(ir_transform(func.body, pre, post))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.ExpandDtypeMismatchedCopy")
