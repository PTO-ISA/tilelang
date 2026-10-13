"""ExpandDtypeMismatchedCopy expands PTO i32 UB -> i64 GM ``tl.tileop.ascend_copy``.

The pass runs after ``InsertNd2Nz`` and before ``VFChecker`` on the PTO route.
It consumes the op exactly as the Ascend frontend emits it
(``tl.tileop.ascend_copy`` over ``tl.region`` pairs) and rewrites the widening
copy into a bare ``SIMD_VF`` widen plus a same-dtype ``tl.tileop.ascend_copy``
(int64 scratch -> original int64 GM destination).  Task 1 delivers the signed
serial baseline only: no VMI path, no assume/zext optimisation, no padding.
"""

from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region


def _expand(mod):
    factory = getattr(transform, "ExpandDtypeMismatchedCopy", None)
    assert factory is not None, (
        "ExpandDtypeMismatchedCopy must be exported from tilelang.ascend.transform"
    )
    return factory()(mod)


def _copy_module(
    k=8,
    *,
    src_base=0,
    dst_base=3,
    dst_shape=16,
    src_shape=None,
    src_strides=None,
    src_scope="shared.dyn",
    dst_scope="global",
    src_dtype="int32",
    dst_dtype="int64",
    src_extent=None,
    dst_extent=None,
    extra_region=None,
    annotations=None,
    in_vf=False,
):
    src = tirx.decl_buffer(
        (src_shape if src_shape is not None else k,),
        src_dtype,
        name="idx_ub",
        scope=src_scope,
        strides=src_strides,
    )
    dst = tirx.decl_buffer((dst_shape,), dst_dtype, name="idx_out", scope=dst_scope)
    src_reg = region(src, 1, [src_base], [src_extent if src_extent is not None else k])
    dst_reg = region(dst, 2, [dst_base], [dst_extent if dst_extent is not None else k])
    args = [src_reg, dst_reg]
    if extra_region is not None:
        args.append(extra_region)
    if callable(annotations):
        annotations = annotations(src, dst)
    call = tirx.Call(
        "handle", tvm.ir.Op.get("tl.tileop.ascend_copy"), args, annotations=annotations
    )
    body = tirx.Evaluate(call)
    if in_vf:
        reads = [tirx.BufferRegion(src, [tvm.ir.Range(src_base, k)])]
        body = tirx.SBlock([], reads, [], "SIMD_VF", body)
    return kernel(body, buffers=[src, dst]), src, dst


def _buffer(reg_call):
    return reg_call.args[0].buffer


def _min_index(reg_call):
    return reg_call.args[0].indices[0]


def _extents(reg_call):
    return [int(x) for x in reg_call.args[2:]]


def test_expand_ascend_copy_i32_i64():
    mod, src, dst = _copy_module(k=8, dst_base=3)
    after = _expand(mod)

    copies = calls(after, "tl.tileop.ascend_copy")
    assert len(copies) == 1, (
        "the widening copy must become exactly one same-dtype ascend_copy"
    )
    call = copies[0]
    src_reg, dst_reg = call.args[0], call.args[1]
    scratch, out = _buffer(src_reg), _buffer(dst_reg)

    # Both ends of the MTE copy are int64.
    assert str(scratch.dtype) == "int64"
    assert str(out.dtype) == "int64"
    # The MTE copy reads the int64 scratch, not the int32 source.
    assert not scratch.same_as(src), "the MTE copy must read the widen scratch"
    assert out.data.same_as(dst.data), "the GM destination must be the original buffer"
    # Destination region offset/extent unchanged.
    assert int(_min_index(dst_reg)) == 3
    assert _extents(dst_reg) == [8]
    # Scratch region starts at zero with the same extent.
    assert int(_min_index(src_reg)) == 0
    assert _extents(src_reg) == [8]
    assert scratch.scope() in ("shared", "shared.dyn")

    # The scratch is declared (as an int64 allocation) and attached to a block.
    allocs = [buf for block in nodes(after, tirx.SBlock) for buf in (block.alloc_buffers or [])]
    assert any(buf.same_as(scratch) for buf in allocs), "int64 scratch must be allocated"

    # Exactly one bare SIMD_VF widen with the right read/write footprints.
    vf = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "SIMD_VF"]
    assert len(vf) == 1, "the widen must be a single bare SIMD_VF"
    block = vf[0]
    assert any(r.buffer.same_as(src) for r in block.reads), "the VF must read the int32 source"
    assert any(r.buffer.same_as(scratch) for r in block.writes), "the VF must write the scratch"

    # No T.Parallel: the widen must not depend on Parallel int64 support.
    fors = nodes(block.body, tirx.For)
    assert fors, "the serial baseline keeps a loop"
    assert all(f.kind != tirx.ForKind.PARALLEL for f in fors), (
        "the widen must not contain a T.Parallel loop"
    )
    casts = [c for c in nodes(block.body, tirx.Cast) if str(c.dtype) == "int64"]
    assert casts, "the serial widen must cast to int64"
    stores = [s for s in nodes(block.body, tirx.BufferStore) if s.buffer.same_as(scratch)]
    assert stores, "the serial widen must store into the int64 scratch"


def test_expand_preserves_ascend_copy_annotations():
    def anns(src, _dst):
        return {
            "data_select": tirx.const(1, "int32"),
            "src_view": tirx.BufferRegion(src, [tvm.ir.Range(0, 8)]),
        }

    mod, src, _ = _copy_module(k=8, annotations=anns)
    after = _expand(mod)
    call = calls(after, "tl.tileop.ascend_copy")[0]
    annotations = call.annotations
    assert annotations is not None
    # Scalar metadata is carried over verbatim.
    assert int(annotations["data_select"]) == 1
    # Source-referencing metadata is remapped onto the int64 scratch.
    view = annotations["src_view"]
    assert isinstance(view, tirx.BufferRegion)
    assert view.buffer.same_as(_buffer(call.args[0]))
    assert [int(r.extent) for r in view.region] == [8]
    assert not view.buffer.same_as(src)


@pytest.mark.parametrize(
    "make_kwargs,match",
    [
        pytest.param(lambda: {"src_strides": (2,), "src_shape": 16}, "stride", id="stride-2"),
        pytest.param(lambda: {"in_vf": True}, "SIMD_VF", id="inside-vf"),
        pytest.param(lambda: {"k": 0}, "positive", id="zero-extent"),
        pytest.param(
            lambda: {"src_extent": tirx.Var("k_dyn", "int32")}, "static", id="dynamic-extent"
        ),
        pytest.param(lambda: {"src_extent": 8, "dst_extent": 7}, "extent", id="extent-mismatch"),
        pytest.param(lambda: {"dst_base": 10}, "bounds", id="destination-oob"),
    ],
)
def test_expand_copy_input_contract(make_kwargs, match):
    mod, _, _ = _copy_module(**make_kwargs())
    with pytest.raises(ValueError, match=match):
        _expand(mod)


def test_expand_rejects_extra_operands_without_truncating():
    mod, _, _ = _copy_module(k=8)
    # Append a third (MX scale) region to the already built call.
    call = calls(mod, "tl.tileop.ascend_copy")[0]
    scale = tirx.decl_buffer((8,), "float32", name="scale", scope="shared.dyn")
    scale_region = region(scale, 1, [0], [8])
    widened = tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.ascend_copy"),
            list(call.args) + [scale_region],
            annotations=call.annotations,
        )
    )
    root = next(b for b in nodes(mod, tirx.SBlock) if b.name_hint == "tilelang_root")
    patched = kernel(widened, buffers=list(root.alloc_buffers) + [scale])
    with pytest.raises(ValueError, match="operand"):
        _expand(patched)


def test_expand_rejects_rank_2():
    src = tirx.decl_buffer((4, 8), "int32", name="idx_ub", scope="shared.dyn")
    dst = tirx.decl_buffer((4, 8), "int64", name="idx_out", scope="global")
    src_reg = region(src, 1, [0, 0], [4, 8])
    dst_reg = region(dst, 2, [0, 0], [4, 8])
    call = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.ascend_copy"), [src_reg, dst_reg])
    patched = kernel(tirx.Evaluate(call), buffers=[src, dst])
    with pytest.raises(ValueError, match="1-D"):
        _expand(patched)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"src_scope": "global"}, id="gm-source"),
        pytest.param({"dst_scope": "shared.dyn"}, id="non-gm-destination"),
        pytest.param({"dst_dtype": "int32"}, id="same-dtype"),
        pytest.param({"dst_dtype": "int16"}, id="narrowing-pair"),
    ],
)
def test_expand_leaves_non_target_copies_alone(kwargs):
    mod, _, _ = _copy_module(**kwargs)
    after = _expand(mod)
    tvm.ir.assert_structural_equal(after, mod)


def test_expand_leaves_scalar_store_path_alone():
    dst = tirx.decl_buffer((8,), "int64", name="idx_out", scope="global")
    i = tirx.Var("i", "int32")
    body = tirx.BufferStore(dst, tirx.const(0, "int64"), [i])
    mod = kernel(body, buffers=[dst])
    after = _expand(mod)
    tvm.ir.assert_structural_equal(after, mod)


# ---------------------------------------------------------------------------
# Fix round (independent review B1/B2 + admission coverage)
# ---------------------------------------------------------------------------


def test_expand_remaps_metadata_relative_to_the_source_region():
    """A non-zero source base must be subtracted, not merely re-buffered."""

    mod, src, _ = _copy_module(
        src_base=3,
        src_shape=16,
        annotations=lambda src, dst: {
            "src_view": tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(3, 8)]),
            "src_region": region(src, 1, [3], [8]),
        },
    )
    after = _expand(mod)
    call = calls(after, "tl.tileop.ascend_copy")[0]
    scratch = _buffer(call.args[0])
    assert int(_min_index(call.args[0])) == 0 and _extents(call.args[0]) == [8]

    view = call.annotations["src_view"]
    assert view.buffer.same_as(scratch)
    assert int(view.region[0].min) == 0 and int(view.region[0].extent) == 8

    remapped = call.annotations["src_region"]
    assert remapped.args[0].buffer.same_as(scratch)
    assert int(remapped.args[0].indices[0]) == 0
    assert [int(x) for x in remapped.args[2:]] == [8]


def test_expand_remaps_nested_source_references():
    """Buffer / data Var / BufferLoad / nested container references are all
    re-pointed, so nothing still refers to the stale int32 source."""

    def anns(src, _dst):
        return {
            "source_buffer": src,
            "source_data": src.data,
            "source_load": tirx.BufferLoad(src, [3]),
            "source_nested": tvm.runtime.convert(
                {"view": tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(3, 8)])}
            ),
            "source_array": tvm.runtime.convert([tirx.BufferLoad(src, [3])]),
        }

    mod, src, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    after = _expand(mod)
    call = calls(after, "tl.tileop.ascend_copy")[0]
    scratch = _buffer(call.args[0])

    assert call.annotations["source_buffer"].same_as(scratch)
    assert call.annotations["source_data"].same_as(scratch.data)
    load = call.annotations["source_load"]
    assert load.buffer.same_as(scratch) and int(load.indices[0]) == 0
    nested = call.annotations["source_nested"]
    assert nested["view"].buffer.same_as(scratch)
    assert int(nested["view"].region[0].min) == 0
    array = call.annotations["source_array"]
    assert array[0].buffer.same_as(scratch) and int(array[0].indices[0]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"src_scope": "global"}, id="gm-source"),
        pytest.param({"dst_scope": "shared.dyn"}, id="non-gm-destination"),
    ],
)
def test_expand_leaves_non_target_copies_with_extra_operands(kwargs):
    """The extra-operand diagnostic is part of the target-pattern contract;
    non-target copies (GM source / non-GM destination) stay untouched."""

    scale = tirx.decl_buffer((8,), "float32", name="scale", scope="shared.dyn")
    mod, _, _ = _copy_module(extra_region=region(scale, 1, [0], [8]), **kwargs)
    after = _expand(mod)
    tvm.ir.assert_structural_equal(after, mod)


def test_expand_proves_capacity_from_enclosing_loops():
    """A copy at ``dst[tile*8 : tile*8+8]`` inside ``serial(0, 2)`` with
    destination capacity 16 is provably in bounds from its lexical loop."""

    tile = tirx.Var("tile", "int32")
    mod, src, dst = _copy_module(dst_base=tile * 8, dst_shape=16)
    call = calls(mod, "tl.tileop.ascend_copy")[0]
    body = tirx.For(tile, 0, 2, tirx.ForKind.SERIAL, tirx.Evaluate(call))
    after = _expand(kernel(body, buffers=[src, dst]))
    rewritten = calls(after, "tl.tileop.ascend_copy")[0]
    assert str(_buffer(rewritten.args[0]).dtype) == "int64"
    assert _extents(rewritten.args[1]) == [8]


def test_expand_rejects_unprovable_loop_capacity():
    """The same shape with loop range [0, 4) is not provably in bounds and
    must still be rejected -- the proof must not simply be dropped."""

    tile = tirx.Var("tile", "int32")
    mod, src, dst = _copy_module(dst_base=tile * 8, dst_shape=16)
    call = calls(mod, "tl.tileop.ascend_copy")[0]
    body = tirx.For(tile, 0, 4, tirx.ForKind.SERIAL, tirx.Evaluate(call))
    with pytest.raises(ValueError, match="destination region"):
        _expand(kernel(body, buffers=[src, dst]))


def test_expand_accepts_explicit_and_simplifiable_unit_stride():
    v = tirx.Var("v", "int32")
    for stride in (1, v - v + 1):
        mod, _, _ = _copy_module(src_strides=(stride,), src_shape=16, src_base=3)
        after = _expand(mod)
        assert str(_buffer(calls(after, "tl.tileop.ascend_copy")[0].args[0]).dtype) == "int64"


def test_expand_rejects_unknown_stride():
    v = tirx.Var("v", "int32")
    mod, _, _ = _copy_module(src_strides=(v,), src_shape=16)
    with pytest.raises(ValueError, match="stride"):
        _expand(mod)


@pytest.mark.parametrize(
    "make_kwargs,match",
    [
        pytest.param(lambda: {"src_base": 1}, "source region", id="source-oob"),
        pytest.param(lambda: {"src_base": -1}, "source region", id="negative-source-base"),
        pytest.param(
            lambda: {"src_shape": tirx.Var("capacity", "int32")},
            "source region",
            id="unknown-source-capacity",
        ),
    ],
)
def test_expand_rejects_source_capacity_violations(make_kwargs, match):
    mod, _, _ = _copy_module(**make_kwargs())
    with pytest.raises(ValueError, match=match):
        _expand(mod)


def test_expand_rejects_simt_vf_nested_target_copy():
    mod, src, dst = _copy_module()
    call = calls(mod, "tl.tileop.ascend_copy")[0]
    patched = kernel(tirx.SBlock([], [], [], "SIMT_VF", tirx.Evaluate(call)), buffers=[src, dst])
    with pytest.raises(ValueError, match="SIMT_VF"):
        _expand(patched)


def test_expand_preserves_base_and_elem_offset_addressing():
    src = tirx.decl_buffer((16,), "int32", name="src", scope="shared.dyn", elem_offset=5, strides=(1,))
    dst = tirx.decl_buffer((24,), "int64", name="dst", scope="global", elem_offset=7, strides=(1,))
    call = tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.ascend_copy"),
        [region(src, 1, [3], [8]), region(dst, 2, [4], [8])],
    )
    after = _expand(kernel(tirx.Evaluate(call), buffers=[src, dst]))

    vf = next(b for b in nodes(after, tirx.SBlock) if b.name_hint == "SIMD_VF")
    assert int(vf.reads[0].region[0].min) == 3 and int(vf.reads[0].region[0].extent) == 8
    assert int(vf.writes[0].region[0].min) == 0 and int(vf.writes[0].region[0].extent) == 8
    assert not any(realize.block.same_as(vf) for realize in nodes(after, tirx.SBlockRealize))

    out_call = calls(after, "tl.tileop.ascend_copy")[0]
    out_load = out_call.args[1].args[0]
    assert out_load.buffer.same_as(dst)
    assert int(out_load.indices[0]) == 4 and int(out_load.buffer.elem_offset) == 7

    loop = nodes(vf.body, tirx.For)[0]
    load = nodes(loop.body, tirx.BufferLoad)[0]
    assert load.buffer.same_as(src) and int(load.buffer.elem_offset) == 5
    first = tirx.stmt_functor.substitute(load.indices[0], {loop.loop_var: tirx.const(0, "int32")})
    assert int(tvm.arith.Analyzer().simplify(first)) == 3 + 8 - 1


def test_expand_rejects_unsupported_metadata_kinds():
    """Metadata kinds the pass cannot remap are rejected, never copied
    blindly (see issues/issue-int64-task1-opaque-metadata)."""

    def anns(src, _dst):
        opaque = tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.access_ptr"),
            [tirx.BufferLoad(src, [0]), tirx.const(8, "int32"), tirx.const(1, "int32")],
        )
        return {"opaque_ref": opaque}

    mod, _, _ = _copy_module(annotations=anns)
    with pytest.raises(ValueError, match="metadata"):
        _expand(mod)


# ---------------------------------------------------------------------------
# Re-review round: Map keys, nested index references, alias buffers
# (review/int64-task1-rereview-metadata-probes.py, 7 cases)
# ---------------------------------------------------------------------------


def test_expand_remaps_source_references_used_as_map_keys():
    """Map keys are traversed like values: a source Buffer / data Var key is
    re-pointed at the scratch, never left stale."""

    def anns(src, _dst):
        return {
            "buffer_key": tvm.runtime.convert({src: tirx.const(1, "int32")}),
            "data_key": tvm.runtime.convert({src.data: tirx.const(1, "int32")}),
        }

    mod, src, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    after = _expand(mod)
    call = calls(after, "tl.tileop.ascend_copy")[0]
    scratch = _buffer(call.args[0])

    (buffer_key,) = list(call.annotations["buffer_key"].keys())
    assert buffer_key.same_as(scratch)
    assert not buffer_key.same_as(src)
    (data_key,) = list(call.annotations["data_key"].keys())
    assert data_key.same_as(scratch.data)


@pytest.mark.parametrize(
    "make_annotation",
    [
        pytest.param(
            lambda src, dst: {
                "meta": tvm.runtime.convert(
                    {
                        tirx.Call(
                            "handle",
                            tvm.ir.Op.get("tl.access_ptr"),
                            [
                                tirx.BufferLoad(src, [3]),
                                tirx.const(8, "int32"),
                                tirx.const(1, "int32"),
                            ],
                        ): tirx.const(1, "int32")
                    }
                )
            },
            id="opaque-source-key",
        ),
        pytest.param(
            lambda src, dst: {"meta": tirx.BufferLoad(dst, [tirx.BufferLoad(src, [3])])},
            id="foreign-load-with-source-index",
        ),
        pytest.param(
            lambda src, dst: {
                "meta": tirx.BufferRegion(
                    dst, [tvm.ir.Range.from_min_extent(tirx.BufferLoad(src, [3]), 1)]
                )
            },
            id="foreign-region-with-source-min",
        ),
        pytest.param(
            lambda src, dst: {"meta": region(dst, 2, [tirx.BufferLoad(src, [3])], [1])},
            id="foreign-tl-region-with-source-index",
        ),
        pytest.param(
            lambda src, dst: {
                "meta": tirx.decl_buffer((16,), "int32", data=src.data, name="alias", scope="shared.dyn")
            },
            id="alias-buffer-sharing-source-data",
        ),
    ],
)
def test_expand_rejects_source_references_it_cannot_remap(make_annotation):
    """A value that still references the int32 source in a form outside the
    supported remap set is rejected, never copied blindly."""

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=make_annotation)
    with pytest.raises(ValueError, match="source|metadata"):
        _expand(mod)


# ---------------------------------------------------------------------------
# Final re-review round: remapped metadata must stay inside the scratch window
# (review/int64-task1-rereview-concurrent-metadata-probes.py, 2 range cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_annotation",
    [
        pytest.param(
            lambda src, dst: {"meta": tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(0, 16)])},
            id="region-before-window",
        ),
        pytest.param(
            lambda src, dst: {"meta": tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(11, 1)])},
            id="region-past-window",
        ),
        pytest.param(lambda src, dst: {"meta": tirx.BufferLoad(src, [2])}, id="load-before-window"),
        pytest.param(lambda src, dst: {"meta": region(src, 1, [2], [8])}, id="tl-region-before-window"),
    ],
)
def test_expand_rejects_metadata_windows_outside_the_scratch(make_annotation):
    """Metadata describing source elements that were not materialized in the
    scratch (mapped window outside [0, K)) is rejected, not emitted
    out-of-range (no scratch enlargement, no silent clipping)."""

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=make_annotation)
    with pytest.raises(ValueError, match="scratch"):
        _expand(mod)


def test_expand_accepts_metadata_at_the_window_boundary():
    """The last copied source element (base + K - 1) maps to scratch K - 1."""

    def anns(src, _dst):
        last = 3 + 8 - 1
        return {
            "region": tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(last, 1)]),
            "load": tirx.BufferLoad(src, [last]),
        }

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    after = _expand(mod)
    call = calls(after, "tl.tileop.ascend_copy")[0]
    assert int(call.annotations["region"].region[0].min) == 7
    assert int(call.annotations["load"].indices[0]) == 7


def test_expand_rejects_source_referencing_region_extent():
    """A tl.region extent that itself references the source is a nested
    source reference and must be rejected, not carried into the scratch
    region call."""

    def anns(src, _dst):
        return {"meta": region(src, 1, [3], [tirx.BufferLoad(src, [4])])}

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    with pytest.raises(ValueError, match="source|metadata"):
        _expand(mod)


def test_expand_rejects_expression_containing_source_load():
    """A generic expression value holding a source BufferLoad is rejected by
    the recursive source scan."""

    def anns(src, _dst):
        return {"meta": tirx.BufferLoad(src, [3]) + tirx.const(1, "int32")}

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    with pytest.raises(ValueError, match="source|metadata"):
        _expand(mod)


# ---------------------------------------------------------------------------
# Closing round: nested source data Var in expressions, source references in
# foreign Buffer fields (review/int64-task1-final-rereview-metadata-probes.py)
# ---------------------------------------------------------------------------


def test_expand_rejects_foreign_load_with_source_data_var_index():
    """A ``src.data`` Var nested inside a foreign load's index expression is a
    source reference and must be rejected (no pointer-expression remapping)."""

    def anns(src, dst):
        external = tirx.Var("external_ptr", "handle")
        index = tirx.Cast("int32", tirx.EQ(src.data, external))
        return {"meta": tirx.BufferLoad(dst, [index])}

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    with pytest.raises(ValueError, match="source|metadata"):
        _expand(mod)


def test_expand_rejects_source_load_with_source_data_var_index():
    """The same expression on the source buffer itself is rejected too
    (previously only by the window guard, which is incidental)."""

    def anns(src, _dst):
        external = tirx.Var("external_ptr", "handle")
        index = tirx.Cast("int32", tirx.EQ(src.data, external))
        return {"meta": tirx.BufferLoad(src, [index])}

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    with pytest.raises(ValueError, match="source|metadata"):
        _expand(mod)


@pytest.mark.parametrize("field", ["shape", "strides", "elem_offset"])
def test_expand_rejects_foreign_buffer_fields_referencing_source(field):
    """Every inspected Buffer field (shape / strides / elem_offset) is scanned:
    a foreign buffer carrying a source reference in any of them is rejected,
    not returned as a safe metadata leaf."""

    def anns(src, _dst):
        ref = tirx.BufferLoad(src, [3])
        if field == "shape":
            buf = tirx.decl_buffer((ref,), "int32", name="metadata_buf", scope="shared.dyn")
        elif field == "strides":
            buf = tirx.decl_buffer(
                (8,), "int32", name="metadata_buf", scope="shared.dyn", strides=(ref,)
            )
        else:
            buf = tirx.decl_buffer(
                (8,), "int32", name="metadata_buf", scope="shared.dyn", elem_offset=ref
            )
        return {"meta": buf}

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    with pytest.raises(ValueError, match="source|metadata"):
        _expand(mod)


def test_expand_leaves_foreign_buffer_without_source_fields_alone():
    """A foreign buffer with clean fields is not a source reference and keeps
    its existing pass-through behavior (no widened rejection surface)."""

    def anns(src, _dst):
        return {
            "meta": tirx.decl_buffer(
                (8,), "int32", name="metadata_buf", scope="shared.dyn", elem_offset=2
            )
        }

    mod, _, _ = _copy_module(src_base=3, src_shape=16, annotations=anns)
    after = _expand(mod)
    call = calls(after, "tl.tileop.ascend_copy")[0]
    assert call.annotations["meta"].elem_offset is not None
    assert int(call.annotations["meta"].elem_offset) == 2


@pytest.mark.pto
def test_frontend_ascend_copy_enters_the_pass():
    """The real Ascend frontend emits the widening ``tl.tileop.ascend_copy``
    that reaches the pass, and the PTO pipeline completes once it is expanded.

    The frontend module captured at the pass entry is the actual
    ``T.copy(i32 UB -> i64 GM)`` IR, not a hand-built lookalike.
    """
    import tilelang.ascend.language as T
    from tilelang.ascend.target import normalize_pto_target
    from tilelang.engine.lower import lower as _lower

    factory = getattr(transform, "ExpandDtypeMismatchedCopy", None)
    assert factory is not None, (
        "ExpandDtypeMismatchedCopy must be exported for the PTO pipeline"
    )
    captured = {}
    original = factory

    def capturing_factory():
        inner = original()

        def run(mod):
            captured.setdefault("pre", mod)
            return inner(mod)

        return run

    @T.prim_func
    def main(A: T.Tensor((8,), "int32"), Out: T.Tensor((8,), "int64")):
        with T.Kernel(1):
            ub = T.alloc_shared((8,), "int32")
            T.copy(A, ub)
            T.copy(ub, Out)

    transform.ExpandDtypeMismatchedCopy = capturing_factory
    try:
        target = normalize_pto_target("pto")
        with target:
            _lower(main, target=target)
    finally:
        transform.ExpandDtypeMismatchedCopy = original

    pre = captured.get("pre")
    assert pre is not None, "the pass must run inside the PTO pipeline"
    widening = []
    for call in calls(pre, "tl.tileop.ascend_copy"):
        try:
            if (
                str(call.args[0].args[0].buffer.dtype) == "int32"
                and str(call.args[1].args[0].buffer.dtype) == "int64"
            ):
                widening.append(call)
        except (AttributeError, IndexError):
            continue
    assert widening, "the frontend must produce the int32->int64 ascend_copy consumed by the pass"
