"""Verify complete indexed offsets using the expression lowering contract."""

import pytest

from test_pto_parallel_gather_scatter import ascend_transform, assert_vector_axis, calls, indexed_kernel, lower_case, tvm
from test_pto_parallel_indexed_control_flow import replace_unit_body
from tilelang.ascend.target import normalize_pto_target
import tilelang.ascend.language as T


def scalar_select_coordinate_cast_kernel(extent=256):
    @T.prim_func
    def main(A: T.Tensor((512,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((512,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    y[i] = x[T.Cast("uint32", idx[i] + T.Select(idx[0] > 0, 0, 1))]
            T.copy(y, C)

    return main


@pytest.fixture(scope="module")
def source_module():
    _, captured = lower_case("indexed_offset_expression_base", indexed_kernel())
    return captured["VerifyParallelToPTO_before"]


@pytest.mark.pto
def test_cast_input_may_include_a_scalar_select_coordinate(source_module):
    def make_body(i, payload, index, output):
        scalar = tvm.tirx.Select(index[0] > 0, 0, 1)
        offset = tvm.tirx.Cast("uint32", index[i] + scalar)
        return tvm.tirx.BufferStore(output, payload[offset], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        assert tvm.ir.structural_equal(mod, verified), "verification must not rewrite scalar coordinates or source casts"
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert len(calls(converted, "tl.vmi.vgather")) == 1
    assert len(calls(converted, "tl.vmi.vload")) >= 1
    assert len(calls(converted, "tl.vmi.vinterpret_cast")) == 1, "the explicit same-width source Cast must survive"


@pytest.mark.pto
def test_complete_offset_checks_references_introduced_by_view_metadata(source_module):
    def make_body(i, payload, index, output):
        ghost = tvm.tirx.Var("undefined_view_origin", "int32")
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=ghost, scope=payload.scope())
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="undefined_view_origin.*never defined"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
def test_uniform_store_checks_operations_introduced_by_view_metadata(source_module):
    def make_body(i, payload, index, output):
        origin = tvm.tirx.call_extern("int32", "opaque_origin")
        view = tvm.tirx.decl_buffer(output.shape, output.dtype, data=output.data, elem_offset=origin, scope=output.scope())
        return tvm.tirx.BufferStore(view, payload[i], [0])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="complete indexed offset.*unsupported call"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["predicate", "conditional_gather"])
def test_metadata_index_reads_keep_their_own_admission_rules(source_module, kind):
    def make_body(i, payload, index, output):
        if kind == "predicate":
            origin = tvm.tirx.BufferLoad(index, [i], predicate=i < 150)
        else:
            origin = tvm.tirx.Select(index[0] > 0, index[index[i]], index[i])
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=origin, scope=payload.scope())
        if kind == "conditional_gather":
            return tvm.tirx.Evaluate(view[index[i]])
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    diagnostic = (
        "access predicate that is not provably true"
        if kind == "predicate"
        else "requires gather.*Select true value|unsupported indexed offset node.*Select"
    )
    with normalize_pto_target("pto"), pytest.raises(Exception, match=diagnostic):
        ascend_transform.VerifyParallelToPTO()(mod)


@T.prim_func
def nested_gather_kernel(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                y[i] = x[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[i]]]]]]]]]]]]]]]
        T.copy(y, C)


@pytest.mark.pto
def test_nested_gather_offsets_preserve_each_read_once():
    _, captured = lower_case("nested_gather_full_pipeline", nested_gather_kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, "tl.vmi.vload")) == 1
    assert len(calls(converted, "tl.vmi.vgather")) == 14


@pytest.mark.pto
def test_bound_external_view_origin_survives_output_reference_check(source_module):
    base = tvm.tirx.Var("external_view_origin", "int32")

    def make_body(i, payload, index, output):
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=base, scope=payload.scope())
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    functions = {}
    for name, func in mod.functions.items():
        functions[name] = tvm.tirx.PrimFunc([*func.params, base], func.body, func.ret_type, func.buffer_map, func.attrs)
    mod = tvm.IRModule(functions)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert len(calls(converted, "tl.vmi.vgather")) == 1


@T.prim_func
def nested_b16_rebased_gather_kernel(A: T.Tensor((70000,), "uint16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((128,), "uint16")):
    with T.Kernel(1):
        raw = T.alloc_shared((70000,), "uint16")
        view = tvm.tirx.decl_buffer((256,), "uint16", data=raw.data, elem_offset=66000, scope="shared.dyn")
        idx = T.alloc_shared((128,), "uint16")
        y = T.alloc_shared((128,), "uint16")
        T.copy(A, raw)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                y[i] = view[view[view[view[view[view[view[view[view[view[view[view[view[view[idx[i]]]]]]]]]]]]]]]
        T.copy(y, C)


@pytest.mark.pto
def test_nested_b16_gathers_verify_metadata_without_rewalking_source_indices():
    _, captured = lower_case("nested_b16_rebased_gather", nested_b16_rebased_gather_kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    gathered = calls(converted, "tl.vmi.vgather")
    assert len(gathered) == 14
    assert all(int(call.args[1]) == 66000 for call in gathered)
    assert len(calls(converted, "tl.vmi.vload")) == 1


def nd_to_nz_kernel(access):
    @T.prim_func
    def main(A: T.Tensor((16, 32), "float16"), C: T.Tensor((2, 16, 16), "float16")):
        with T.Kernel(1):
            nd = T.alloc_shared((512,), "float16")
            nz = T.alloc_shared((512,), "float16")
            T.copy(A, nd)
            if access == "scatter":
                T.copy(C, nz)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(512):
                    if access == "gather":
                        nz[i] = nd[(i % 256 // 16) * 32 + (i // 256) * 16 + i % 16]
                    else:
                        nz[(i % 32 // 16) * 256 + (i // 32) * 16 + i % 16] = nd[i]
            T.copy(nz, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("access", ["gather", "scatter"])
def test_nd_to_nz_indexes_are_vectorized(access):
    _, captured = lower_case(f"simd_parallel_nd_to_nz_{access}", nd_to_nz_kernel(access))
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, f"tl.vmi.v{access}")) == 1


def floor_divmod_kernel(dtype, divisor, lanes, operand_mode="repeated", *, extent=None):
    extent = 2 * lanes if extent is None else extent
    outputs = 1 if operand_mode == "remainder_only" else 2

    @T.prim_func
    def main(A: T.Tensor((extent,), dtype), B: T.Tensor((extent,), dtype), C: T.Tensor((outputs, extent), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((extent,), dtype)
            d = T.alloc_shared((extent,), dtype)
            y = T.alloc_shared((outputs, extent), dtype)
            x_view = T.view(x, shape=(extent,))
            T.copy(A, x)
            if divisor is None:
                T.copy(B, d)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(extent):
                    if divisor is not None:
                        y[0, i] = x[i] // T.cast(divisor, dtype)
                        y[1, i] = x[i] % T.cast(divisor, dtype)
                    elif operand_mode == "remainder_only":
                        y[0, i] = x[i] % d[i]
                    elif operand_mode == "bound":
                        lhs = x[i]
                        rhs = d[i]
                        y[0, i] = lhs // rhs
                        y[1, i] = lhs % rhs
                    else:
                        y[0, i] = x[i] // d[i]
                        if operand_mode == "overwritten":
                            x_view[i] = x[i] + 1
                        y[1, i] = x[i] % d[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("operand_mode", ["repeated", "bound", "overwritten"])
def test_floor_divmod_shares_the_loop_for_unchanged_operands(operand_mode):
    artifact, _ = lower_case(f"simd_parallel_floor_divmod_{operand_mode}", floor_divmod_kernel("int32", None, 64, operand_mode))
    source = artifact.kernel_source
    if operand_mode == "overwritten":
        assert "tl.vmi_integer_divmod(" not in source
        assert source.count("tl.vmi_integer_floor(") == 2
    else:
        assert source.count("tl.vmi_integer_divmod(") == 1
        assert "tl.vmi_integer_floor(" not in source


def varying_floor_offset_kernel(kind, divisor):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    if kind == "gather_floordiv":
                        y[i] = x[i // divisor]
                    elif kind == "gather_floormod":
                        y[i] = x[idx[i] + i % divisor]
                    elif kind == "gather_chunk_floordiv":
                        view = T.view(x, shape=(2, 128), strides=(128, 1))
                        y[i] = view[i // divisor, idx[i]]
                    elif kind == "scatter_floordiv":
                        y[idx[i] // divisor] = x[i]
                    else:
                        assert kind == "stride_floormod"
                        view = T.view(x, shape=(2, 128), strides=(i % divisor, 1))
                        y[i] = view[1, 0]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "kind,divisor,access",
    [
        ("gather_floordiv", 2, "gather"),
        ("gather_floordiv", 3, "gather"),
        ("gather_floormod", 3, "gather"),
        ("gather_chunk_floordiv", 128, "gather"),
        ("scatter_floordiv", 3, "scatter"),
        ("stride_floormod", 3, "gather"),
    ],
)
def test_varying_floor_offsets_are_vectorized(kind, divisor, access):
    _, captured = lower_case(f"varying_floor_{kind}_{divisor}", varying_floor_offset_kernel(kind, divisor))
    assert len(calls(captured["VectorizeParallelToPTO_after"], f"tl.vmi.v{access}")) == 1


def scalar_floor_coordinate_kernel(kind):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((128,), "int32"), C: T.Tensor((4, 256), "float32")):
        with T.Kernel(4) as bx:
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((128,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C[bx, :], y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if kind == "gather":
                        y[i] = x[idx[i] + bx % 2]
                    else:
                        y[idx[i] + bx // 2] = x[i]
            T.copy(y, C[bx, :])

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_scalar_floor_coordinates_stay_scalar_in_indexed_offsets(kind):
    # `bx % 2` / `bx // 2` do not depend on the lane, so they are preserved as
    # scalar coordinates and enter the instruction once as a broadcast —
    # the same treatment compiler-generated buffer-version coordinates
    # receive. Simplification may rewrite the operator (e.g. to a bitwise
    # form), so the contract checked here is the scalar dtype of the single
    # broadcast, not the surviving node type.
    _, captured = lower_case(f"scalar_floor_{kind}", scalar_floor_coordinate_kernel(kind))
    converted = captured["VectorizeParallelToPTO_after"]
    operation = "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter"
    assert len(calls(converted, operation)) == 1
    assert not calls(converted, "tl.vmi.vcvt")
    broadcasts = calls(converted, "tl.vmi.vbrc")
    assert len(broadcasts) == 1, "the scalar floor coordinate is broadcast once"
    assert str(broadcasts[0].args[0].dtype) == "int32", "the floor coordinate stays a scalar"


@T.prim_func
def chunk_division_offset(A: T.Tensor((512,), "int32"), I: T.Tensor((256,), "int32"), C: T.Tensor((4, 256), "int32")):
    with T.Kernel(1):
        x = T.alloc_shared((512,), "int32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((4, 256), "int32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j] = x[idx[j] + T.truncdiv(j, 128)]
        T.copy(y, C)


@pytest.mark.pto
def test_candidate_rejects_integer_division_with_a_vector_operand():
    _, captured = lower_case("chunk_division_offset", chunk_division_offset)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 0)
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, "tl.vmi.vscatter")) == 1
    assert not calls(converted, "tl.vmi.vgather")
    assert not calls(converted, "tl.vmi.vdiv")
