"""PTO Parallel must use the current scheduler slot for every vector address."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.target import normalize_pto_target
from tvm import tirx

from testing.ascend._ir import kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.pto
@pytest.mark.parametrize("stages", [1, 2, 3, 7])
def test_persistent_expand_compiles(stages):
    n = T.dynamic("n")

    @T.prim_func
    def expand(X: T.Tensor((n, 1280), "bfloat16"), Y: T.Tensor((n, 4, 1280), "bfloat16")):
        with T.Kernel(4) as core:
            x = T.alloc_shared((256,), "bfloat16")
            y = T.alloc_shared((4, 256), "bfloat16")
            T.annotate_buffer_versions({x: stages, y: stages})
            for row, col in T.Persistent([n, 5], 4, core, group_size=1, num_stages=stages):
                T.copy(X[row, col * 256], x)
                with T.SimdVF(lanes=128):
                    for m, j in T.Parallel(4, 256):
                        y[m, j] = x[j]
                T.copy(y, Y[row, 0, col * 256])

    compiled = tilelang.compile(expand, target="pto")
    assert compiled.get_kernel_source()


@pytest.mark.parametrize("mode", ["guarded", "alias_store", "escape", "local_allocation", "explicit_vf", "ascendc"])
def test_version_slot_scope(mode):
    x = tirx.decl_buffer((64,), "int32", name="x", scope="shared.dyn")
    counter = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    # An unrelated local with the same name must never be captured as a slot.
    user_index = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    tile, lane = tirx.Var("tile", "int32"), tirx.Var("lane", "int32")
    kind = tirx.ForKind.SERIAL if mode == "explicit_vf" else tirx.ForKind.PARALLEL
    body = tirx.For(lane, 0, 64, kind, tirx.BufferStore(x, x[user_index[0]], [lane]))
    if mode == "alias_store":
        alias = tirx.decl_buffer((1,), "int32", data=counter.data, scope="local.var")
        body = seq(body, tirx.BufferStore(alias, 1, [0]))
    elif mode == "escape":
        body = seq(body, tirx.Evaluate(tirx.call_extern("int32", "opaque", counter.data)))
    vf = tirx.SBlock([], [], [], "SIMD_VF", body, alloc_buffers=[counter] if mode == "local_allocation" else [])
    pred = tile < 3
    vf = tirx.SBlockRealize([], pred, vf)
    advance = unit(tirx.BufferStore(counter, counter[0] + 1, [0]))
    owner = tirx.For(
        tile,
        0,
        4,
        tirx.ForKind.SERIAL,
        seq(unit(tirx.BufferStore(x, tile, [0])), unit(vf), advance),
        annotations={
            "multi_buffer_eligible": [x.data],
            "tl.multi_buffer_counter_map": {x.data: counter},
            "tl.storage_epoch_guard_map": {x.data: tirx.const(True, "bool")},
        },
    )
    before = kernel(
        seq(unit(tirx.BufferStore(counter, 0, [0])), unit(owner, core=None)),
        buffers=[x, counter, user_index],
        annotations={"tl.buffer_versions_map": {x.data: 3}},
    )
    if mode != "ascendc":
        before = tvm.IRModule({"main": before["main"].with_attr("target", normalize_pto_target("pto"))})
    after = transform.MaterializeMultiBuffer()(before)
    bindings = nodes(after, tirx.Bind)
    if mode != "guarded":
        assert not bindings
        return
    (binding,) = bindings
    assert tvm.arith.Analyzer().can_prove_equal(binding.value, counter[0] % 3)
    (guard,) = [g for g in nodes(after, tirx.IfThenElse) if nodes(g.then_case, tirx.Bind)]
    assert tvm.ir.structural_equal(guard.condition, pred)
    first, second = guard.then_case.seq
    assert first.same_as(binding) and isinstance(second, tirx.SBlockRealize)
    (store,) = nodes(second.block, tirx.BufferStore)
    assert store.indices[0].same_as(binding.var) and store.value.indices[0].same_as(binding.var)
    assert store.value.indices[1].buffer.data.same_as(user_index.data)


def test_two_vfs_refresh_independent_clocks():
    x, y = [tirx.decl_buffer((64,), "float32", name=n, scope="shared.dyn") for n in ("x", "y")]
    read, write = [tirx.decl_buffer((1,), t, name="epoch", scope="local.var") for t in ("int32", "int64")]
    tile = tirx.Var("tile", "int32")
    active = tile < 3

    def vf():
        lane = tirx.Var("lane", "int32")
        body = tirx.For(lane, 0, 64, tirx.ForKind.PARALLEL, tirx.BufferStore(y, x[lane], [lane]))
        return unit(tirx.SBlock([], [], [], "SIMD_VF", body), guard=active)

    step = unit(tirx.BufferStore(read, read[0] + 1, [0]), guard=active)
    body = seq(unit(tirx.BufferStore(x, 0.0, [0])), unit(tirx.BufferStore(y, 0.0, [0])), vf(), step, vf())
    owner = tirx.For(
        tile,
        0,
        4,
        tirx.ForKind.SERIAL,
        body,
        annotations={
            "multi_buffer_eligible": [x.data, y.data],
            "tl.multi_buffer_counter_map": {x.data: read, y.data: write},
            "tl.storage_epoch_guard_map": {x.data: active, y.data: active},
            "tl.write_counter_update_guard_map": {y.data: active},
        },
    )
    before = kernel(
        seq(unit(tirx.BufferStore(read, 0, [0])), unit(tirx.BufferStore(write, tirx.const(0, "int64"), [0])), unit(owner, core=None)),
        buffers=[x, y, read, write],
        annotations={"tl.buffer_versions_map": {x.data: 2, y.data: 3}},
    )
    before = tvm.IRModule({"main": before["main"].with_attr("target", normalize_pto_target("pto"))})
    result = transform.MaterializeMultiBuffer()(before)
    groups = [g for g in nodes(result, tirx.IfThenElse) if nodes(g, tirx.Bind)]
    assert len(groups) == 2
    all_slots = []
    for g in groups:
        bindings = nodes(g, tirx.Bind)
        assert len(bindings) == 2
        (store,) = nodes(g, tirx.BufferStore)
        for address, counter, modulus in [(store, write, 3), (store.value, read, 2)]:
            (binding,) = [b for b in bindings if b.var.same_as(address.indices[0])]
            expected = tirx.Cast(binding.var.dtype, counter[0] % modulus)
            assert tvm.arith.Analyzer().can_prove_equal(binding.value, expected)
            assert all(not binding.var.same_as(old) for old in all_slots)
            all_slots.append(binding.var)
