"""PTO reducer state, horizontal finalization and one-lane consumers."""

import pytest
import tilelang.ascend.language as T
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower

pytestmark = pytest.mark.pto


def reducer_kernel(
    n=257, lanes=128, dtype="float32", op="sum", seed=False, arithmetic=False, conditional=False, private=False, uniform=False
):
    padded = (n + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((padded,), dtype), Out: T.Tensor((padded,), dtype), Result: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((padded,), dtype)
            b = T.alloc_shared((padded,), dtype)
            out = T.alloc_shared((lanes,), dtype)
            T.copy(A, a)
            T.copy(Out, b)
            T.copy(Result, out)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_reducer((1,), dtype, op=op)
                result = T.alloc_local((1,), dtype)
                if seed:
                    T.reducer_init(acc, T.Cast(dtype, 7))
                else:
                    T.reducer_init(acc)
                for i in T.Parallel(n):
                    if private:
                        value = T.alloc_var(dtype, init=T.Cast(dtype, 1))
                        scratch = T.alloc_local((1,), dtype)
                        scratch[0] = a[i] + value
                        T.reducer_update(acc[0], scratch[0])
                    elif conditional:
                        if a[i] > T.Cast(dtype, 0):
                            T.reducer_update(acc[0], a[i])
                    else:
                        T.reducer_update(acc[0], a[0 if uniform else i])
                T.finalize_reducer(acc, result)
                alias = result[0]
                out[0] = alias
                for i in T.Parallel(n):
                    if arithmetic:
                        b[i] = a[i] * (alias / T.Cast(dtype, 4096))
                    else:
                        b[i] = a[i] * alias
            T.copy(b, Out)
            T.copy(out, Result)

    return main


def result_consumer_kernel(mode, dtype="float32"):
    cast_dtype = "float16" if dtype == "float32" else "float32"
    out_dtype = cast_dtype if mode.startswith("cast") else dtype

    @T.prim_func
    def main(A: T.Tensor((384,), dtype), Out: T.Tensor((384,), out_dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((384,), dtype)
            out = T.alloc_shared((384,), out_dtype)
            T.copy(A, a)
            T.copy(Out, out)
            with T.SimdVF(lanes=128):
                acc = T.alloc_reducer((1,), dtype, op="sum")
                result = T.alloc_local((1,), dtype)
                T.reducer_init(acc)
                for i in T.Parallel(257):
                    T.reducer_update(acc[0], a[i])
                T.finalize_reducer(acc, result)
                alias = result[0]
                alias2 = alias
                if mode == "while_direct" or mode == "while_alias":
                    keep = T.alloc_var("bool", init=True)
                    while keep:
                        if mode == "while_alias":
                            out[0] = alias2
                        else:
                            out[0] = result[0]
                        keep = False
                elif mode == "if_result":
                    if result[0] > 0:
                        out[0] = T.Cast(dtype, 1)
                elif mode == "while_result":
                    while (alias2 > 0) & (a[0] > 0):
                        out[0] = T.Cast(dtype, 1)
                elif mode == "ub_condition":
                    out[0] = result[0]
                    if out[0] > 0:
                        out[1] = result[0]
                elif mode == "cast_alias":
                    for i in T.Parallel(257):
                        out[i] = T.Cast(cast_dtype, alias2)
                else:
                    for i in T.Parallel(257):
                        out[i] = T.Cast(cast_dtype, result[0])
            T.copy(out, Out)

    return main


@pytest.mark.parametrize("mode,wrapper", [("if_result", "If"), ("while_result", "While")])
def test_region_condition_requires_scalar(mode, wrapper):
    with pytest.raises(Exception, match=rf"VerifyParallelToPTO.*region-level {wrapper} condition.*depends on VMI value"):
        _lower(result_consumer_kernel(mode))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("mode", ["cast_direct", "cast_alias"])
def test_one_lane_result_cast_rejected(dtype, mode):
    with pytest.raises(Exception, match="VerifyParallelToPTO.*casting a one-lane VMI value"):
        _lower(result_consumer_kernel(mode, dtype))


def _lower(f):
    target = normalize_pto_target("pto")
    with target:
        return lower(f, target=target)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_reducer_codegen(dtype, op):
    source = _lower(reducer_kernel(dtype=dtype, op=op)).kernel_source
    assert {"sum": "pto.vmi.vcadd", "max": "pto.vmi.vcmax", "min": "pto.vmi.vcmin"}[op] in source
    assert "group=1" in source
    assert "pto.vmi.vsel" in source


@pytest.mark.parametrize("seed", [False, True])
@pytest.mark.parametrize("dtype", ["int32", "uint32"])
@pytest.mark.parametrize("op", ["min", "max"])
def test_integer_extrema_reducer_codegen(seed, dtype, op, monkeypatch):
    from tilelang import tvm
    from testing.ascend._ir import calls, nodes

    kernel = reducer_kernel(n=65, lanes=64, dtype=dtype, op=op, seed=seed)
    after = _capture_stages(kernel, monkeypatch)["VectorizeParallelToPTO"]
    binds = nodes(after, tvm.tirx.Bind)
    (compare,) = calls(after, "tl.vmi.vcmp")
    assert [str(arg.dtype) for arg in compare.args[:2]] == [f"{dtype}x64"] * 2
    (horizontal,) = calls(after, "tl.vmi.vc" + op)
    assert str(horizontal.args[0].dtype) == f"{dtype}x64"
    assert int(horizontal.annotations["group"]) == 1
    result = next(bind.var for bind in binds if bind.value.same_as(horizontal))
    combines = calls(after, "tl.vmi.v" + op)
    if seed:
        (combine,) = combines
        assert combine.args[0].same_as(result)
        assert [str(arg.dtype) for arg in combine.args[:2]] == [dtype, dtype]
        seed_broadcast = next(bind.value for bind in binds if bind.var.same_as(combine.args[1]))
        assert seed_broadcast.op.name == "tl.vmi.vbrc"
        seed_load = next(bind.value for bind in binds if bind.var.same_as(seed_broadcast.args[0]))
        seed_store = next(store for store in nodes(after, tvm.tirx.BufferStore) if store.buffer.same_as(seed_load.buffer))
        assert int(seed_store.value) == 7
        result = next(bind.var for bind in binds if bind.value.same_as(combine))
    else:
        assert not combines
    assert any(store.value.same_as(result) for store in nodes(after, tvm.tirx.BufferStore))


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
@pytest.mark.parametrize("op", ["min", "max"])
def test_integer_extrema_reducer_uniform_input(dtype, op):
    kernel = reducer_kernel(n=65, lanes=64, dtype=dtype, op=op, uniform=True)
    assert f"pto.vmi.vc{op}" in _lower(kernel).kernel_source


@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_integer_reducer_single_input(lanes):
    kernel = reducer_kernel(n=1, lanes=lanes, dtype="uint32", op="max")
    assert "pto.vmi.vcmax" in _lower(kernel).kernel_source


def integer_reducer_consumer_kernel(mode):
    @T.prim_func
    def main(A: T.Tensor((64,), "uint32"), Out: T.Tensor((64,), "uint32")):
        with T.Kernel(1):
            src = T.alloc_shared((64,), "uint32")
            out = T.alloc_shared((64,), "uint32")
            T.copy(A, src)
            with T.SimdVF(lanes=64):
                acc = T.alloc_reducer((1,), "uint32", op="max")
                result = T.alloc_local((1,), "uint32")
                T.reducer_init(acc)
                for i in T.Parallel(64):
                    T.reducer_update(acc[0], src[i])
                T.finalize_reducer(acc, result)
                for i in T.Parallel(64):
                    if mode == "alias":
                        saved = T.bind(result[0])
                        out[i] = src[i] * saved
                    elif mode == "bitwise_alias":
                        saved = T.bind(result[0])
                        out[i] = src[i] | (saved & T.uint32(0x807FFFFF))
                    elif mode == "max_alias":
                        saved = T.bind(result[0])
                        out[i] = T.max(saved, T.uint32(7))
                    else:
                        out[i] = T.max(result[0], T.uint32(7))
            T.copy(out, Out)

    return main


@pytest.mark.parametrize("mode, operation", [("alias", "vmul"), ("max", "vsel"), ("max_alias", "vsel"), ("bitwise_alias", "vor")])
def test_integer_reducer_result_consumer(mode, operation, monkeypatch):
    from tilelang import tvm
    from testing.ascend._ir import calls, nodes

    after = _capture_stages(integer_reducer_consumer_kernel(mode), monkeypatch)["VectorizeParallelToPTO"]
    producers = {bind.var: bind.value for bind in nodes(after, tvm.tirx.Bind)}
    (horizontal,) = calls(after, "tl.vmi.vcmax")
    (finalize,) = [
        store for store in nodes(after, tvm.tirx.BufferStore) if store.value in producers and producers[store.value].same_as(horizontal)
    ]
    (output,) = calls(after, "tl.vmi.vstore")
    consumer = producers[output.args[0]]
    assert consumer.op.name == f"tl.vmi.{operation}"
    broadcast = producers[consumer.args[1]]
    assert broadcast.op.name == "tl.vmi.vbrc" and int(broadcast.annotations["group"]) == 1
    source = producers[broadcast.args[0]]
    if mode == "bitwise_alias":
        assert source.op.name == "tl.vmi.vand" and str(source.dtype) == "uint32"
        assert int(producers[source.args[2]].args[0]) == 1
        source = producers[source.args[0]]
    if mode.endswith("alias"):
        assert isinstance(source, tvm.tirx.Var)
        source = producers[source]
    assert isinstance(source, tvm.tirx.BufferLoad) and source.buffer.same_as(finalize.buffer)


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_integer_sum_reducer_rejected(dtype):
    with pytest.raises(Exception, match="PTO reducer requires float16/float32 or int32/uint32 with min/max"):
        _lower(reducer_kernel(dtype=dtype, op="sum"))


def test_seed_and_one_lane_arithmetic():
    source = _lower(reducer_kernel(seed=True, arithmetic=True)).kernel_source
    assert "pto.vmi.vdiv" in source


@pytest.mark.parametrize("lanes", [64, 128, 256])
@pytest.mark.parametrize("extent", ["one", "short", "full", "tail"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_reducer_layout_width(lanes, extent, dtype, op, monkeypatch):
    from tilelang import tvm
    from tilelang.layout.partial_fragment import PartialFragment
    from testing.ascend._ir import calls, nodes

    n = {"one": 1, "short": 17, "full": lanes, "tail": 2 * lanes + 1}[extent]
    stages = _capture_stages(reducer_kernel(n=n, lanes=lanes, dtype=dtype, op=op), monkeypatch)
    vf = next(b for b in nodes(stages["AscendLayoutInference"], tvm.tirx.SBlock) if b.name_hint == "SIMD_VF")
    (partial,) = [layout for buf, layout in vf.annotations["layout_map"].items() if "reducer" in buf.scope()]
    assert isinstance(partial, PartialFragment) and int(partial.combine_size) == lanes
    after = stages["VectorizeParallelToPTO"]
    producers = {b.var: b.value for b in nodes(after, tvm.tirx.Bind)}
    (horizontal,) = calls(after, {"sum": "tl.vmi.vcadd", "max": "tl.vmi.vcmax", "min": "tl.vmi.vcmin"}[op])
    mask = producers[horizontal.args[1]]
    assert mask.op.name == "tl.vmi.create_mask" and int(mask.args[0]) == lanes
    accumulator = producers[horizontal.args[0]].buffer
    writes = [store for store in nodes(after, tvm.tirx.BufferStore) if store.buffer.same_as(accumulator)]
    init = producers[writes[0].value]
    assert init.op.name == "tl.vmi.vbrc" and int(init.annotations["size"]) == lanes


def unsupported_kernel(updates=1, arithmetic="none", seed=False):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), Out: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                acc = T.alloc_reducer((1,), "float32", op="sum")
                result = T.alloc_local((1,), "float32")
                if arithmetic == "seed":
                    T.reducer_init(acc, a[0] + 1)
                elif seed:
                    T.reducer_init(acc, T.float32(7))
                else:
                    T.reducer_init(acc)
                if updates > 0:
                    for i in T.Parallel(128):
                        T.reducer_update(acc[0], a[i])
                        if updates > 1:
                            T.reducer_update(acc[0], a[i])
                T.finalize_reducer(acc, result)
                if arithmetic == "store":
                    a[0] = result[0] / 4096
                elif arithmetic == "bind":
                    bad = result[0] / 4096
                    a[0] = bad
                else:
                    a[0] = result[0]
            T.copy(a, Out)

    return main


@pytest.mark.parametrize("updates,seed", [(0, False), (0, True), (2, False)])
def test_update_count_diagnostic(updates, seed):
    with pytest.raises(Exception, match="exactly one static update"):
        _lower(unsupported_kernel(updates=updates, seed=seed))


@pytest.mark.parametrize("arithmetic", ["seed", "store", "bind"])
def test_region_arithmetic_rejected(arithmetic):
    with pytest.raises(Exception, match="ordinary data arithmetic outside Parallel"):
        _lower(unsupported_kernel(arithmetic=arithmetic))


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_conditional_reducer(op):
    assert "pto.vmi.vcmp" in _lower(reducer_kernel(op=op, conditional=True)).kernel_source


def seed_snapshot_kernel(vmi_seed=False, repeat=1):
    @T.prim_func
    def main(A: T.Tensor((384,), "float32"), Result: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((384,), "float32")
            out = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                source = T.alloc_local((1,), "float32")
                result = T.alloc_local((1,), "float32")
                if vmi_seed:
                    first = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(first)
                    for j in T.Parallel(1):
                        T.reducer_update(first[0], a[j])
                    T.finalize_reducer(first, source)
                else:
                    source[0] = T.float32(7)
                for epoch in T.serial(repeat):
                    acc = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(acc, source[0])
                    source[0] = T.float32(19)
                    for i in T.Parallel(257):
                        T.reducer_update(acc[0], a[i])
                    T.finalize_reducer(acc, result)
                    out[epoch] = result[0]
            T.copy(out, Result)

    return main


@pytest.mark.parametrize("vmi_seed", [False, True])
def test_seed_snapshot_codegen(vmi_seed):
    source = _lower(seed_snapshot_kernel(vmi_seed)).kernel_source
    assert "pto.vmi.vcadd" in source


def test_repeated_epoch_codegen():
    assert "pto.vmi.vcadd" in _lower(seed_snapshot_kernel(repeat=2)).kernel_source


def mixed_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), Out: T.Tensor((256,), "float32"), Result: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            b = T.alloc_shared((256,), "float32")
            reduced = T.alloc_shared((1,), "float32")
            out = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    b[i] = a[i] * 2
                T.reduce_sum(b, reduced, dim=0)
                acc = T.alloc_reducer((1,), "float32", op="sum")
                result = T.alloc_local((1,), "float32")
                T.reducer_init(acc)
                for i in T.Parallel(256):
                    T.reducer_update(acc[0], b[i])
                T.finalize_reducer(acc, result)
                out[0] = result[0]
                for i in T.Parallel(256):
                    b[i] = a[i] * result[0] + reduced[0]
            T.copy(b, Out)
            T.copy(out, Result)

    return main


def test_mixed_direct_reduce():
    assert _lower(mixed_kernel()).kernel_source.count("pto.vmi.vcadd") == 2


def _capture_stages(f, monkeypatch):
    from tilelang import transform
    from tilelang.ascend import transform as ascend

    snapshots = {}
    for owner, name in [(ascend, "AscendLayoutInference"), (transform, "ReducerPlanAndMaterialize"), (ascend, "VectorizeParallelToPTO")]:
        if not hasattr(owner, name):
            continue
        original = getattr(owner, name)

        def factory(*args, _original=original, _name=name, **kwargs):
            run = _original(*args, **kwargs)

            def wrapped(mod):
                snapshots[_name + "_before"] = mod
                result = run(mod)
                snapshots[_name] = result
                return result

            return wrapped

        monkeypatch.setattr(owner, name, factory)
    _lower(f)
    return snapshots


@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_finalize_dataflow(monkeypatch, lanes):
    from tilelang import tvm
    from testing.ascend._ir import calls, nodes

    s = _capture_stages(reducer_kernel(lanes=lanes, seed=True, arithmetic=True), monkeypatch)
    layout_vf = next(b for b in nodes(s["AscendLayoutInference"], tvm.tirx.SBlock) if b.name_hint == "SIMD_VF")
    layouts = layout_vf.annotations["layout_map"]
    partials = [(b, layout) for b, layout in layouts.items() if "reducer" in b.scope()]
    assert len(partials) == 1 and "PartialFragment" in str(type(partials[0][1]))
    assert not any(b.name == "result" for b in layouts)
    before = s["VectorizeParallelToPTO_before"]
    after = s["VectorizeParallelToPTO"]
    vf = next(b for b in nodes(before, tvm.tirx.SBlock) if b.name_hint == "SIMD_VF")
    states = vf.annotations["tl.pto.reducer_states"]
    assert len(states) == 1
    assert any(b.data.same_as(states[0]) and b.scope() == "local" for b in vf.alloc_buffers)
    final = calls(before, "tl.tileop.pto_finalize_reducer")
    assert len(final) == 1 and [int(x) for x in final[0].annotations["plan_steps"]] == [lanes, 1]
    assert not calls(after, "tl.tileop.pto_finalize_reducer")
    assert all("tl.pto.reducer_states" not in b.annotations for b in nodes(after, tvm.tirx.SBlock))
    binds = nodes(after, tvm.tirx.Bind)
    producer = {b.var: b.value for b in binds}
    horizontal = calls(after, "tl.vmi.vcadd")[0]
    mask = producer[horizontal.args[1]]
    assert mask.op.name == "tl.vmi.create_mask" and int(mask.args[0]) == lanes
    div = calls(after, "tl.vmi.vdiv")[0]
    one_mask = producer[div.args[2]]
    assert int(one_mask.annotations["size"]) == 1 and int(one_mask.args[0]) == 1
    div_var = next(b.var for b in binds if b.value.same_as(div))
    broadcast = next(c for c in calls(after, "tl.vmi.vbrc") if c.args[0].same_as(div_var))
    assert int(broadcast.annotations["group"]) == 1
    assert int(broadcast.annotations["size"]) == lanes
    one_stores = [c for c in calls(after, "tl.vmi.vstore") if len(c.args) == 3]
    assert len(one_stores) == 1 and int(one_stores[0].args[1].args[1]) == 1


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing", "missing from tl.pto.reducer_states"),
        ("foreign", "allocation"),
        ("odd_plan", "width/scale pairs"),
        ("unsupported_plan", "only plan_steps"),
        ("empty_plan", "only plan_steps"),
    ],
)
def test_materialized_contract_diagnostics(monkeypatch, damage, match):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend
    from tvm.tirx.stmt_functor import ir_transform

    s = _capture_stages(reducer_kernel(), monkeypatch)
    mod = s["VectorizeParallelToPTO_before"]

    def rewrite(node):
        if isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            annotations = dict(node.annotations)
            if damage == "missing":
                annotations.pop("tl.pto.reducer_states")
            if damage == "foreign":
                annotations["tl.pto.reducer_states"] = [tvm.tirx.Var("foreign", "handle")]
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                annotations,
            )
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.pto_finalize_reducer" and damage.endswith("plan"):
            annotations = dict(node.annotations)
            annotations["plan_steps"] = [
                tvm.tirx.IntImm("int32", x) for x in {"odd_plan": [128], "unsupported_plan": [64, 1], "empty_plan": []}[damage]
            ]
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations=annotations)

    changed = tvm.IRModule({g: f.with_body(ir_transform(f.body, None, rewrite)) for g, f in mod.functions.items()})
    with pytest.raises(Exception, match=match):
        ascend.VerifyParallelToPTO()(changed)


def test_plan_element_type_checked(monkeypatch):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend
    from tvm.tirx.stmt_functor import ir_transform

    mod = _capture_stages(reducer_kernel(), monkeypatch)["VectorizeParallelToPTO_before"]

    def rewrite(node):
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.pto_finalize_reducer":
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations={"plan_steps": ["invalid", "invalid"]})

    changed = tvm.IRModule({g: f.with_body(ir_transform(f.body, None, rewrite)) for g, f in mod.functions.items()})
    with pytest.raises(Exception, match="plan_steps must be IntImm"):
        ascend.VerifyParallelToPTO()(changed)


def test_handwritten_vmi_without_lanes_unchanged():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128,), "float32")
            b = T.alloc_shared((128,), "float32")
            T.copy(A, a)
            with T.SimdVF():
                value = T.vmi.vload(a[0], size=128)
                T.vmi.vstore(value, b[0])
            T.copy(b, B)

    assert "pto.vmi.vload" in _lower(main).kernel_source


@pytest.mark.parametrize(
    "damage,match",
    [
        ("offset", "single local slot"),
        ("overlap", "independent"),
        ("dtype", "dtype mismatch"),
        ("access", "access_mask"),
    ],
)
def test_finalize_regions_checked(monkeypatch, damage, match):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend
    from testing.ascend._ir import region
    from tvm.tirx.stmt_functor import ir_transform

    mod = _capture_stages(reducer_kernel(), monkeypatch)["VectorizeParallelToPTO_before"]

    def rewrite(node):
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.pto_finalize_reducer":
            args = list(node.args)
            acc = args[0].args[0].buffer
            dst = args[1].args[0].buffer
            if damage == "offset":
                args[1] = region(dst, 2, [1], [1])
            elif damage == "overlap":
                args[1] = region(acc, 2)
            elif damage == "dtype":
                bad = tvm.tirx.decl_buffer((1,), "float16", "bad", scope="local")
                args[1] = region(bad, 2)
            else:
                args[1] = region(dst, 1)
            return tvm.tirx.Call(node.dtype, node.op, args, annotations=node.annotations)

    changed = tvm.IRModule({g: f.with_body(ir_transform(f.body, None, rewrite)) for g, f in mod.functions.items()})
    with pytest.raises(Exception, match=match):
        ascend.VerifyParallelToPTO()(changed)
