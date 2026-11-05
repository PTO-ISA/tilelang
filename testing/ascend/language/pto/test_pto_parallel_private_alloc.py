"""Ordinary Parallel locals: ownership, mutable snapshots and admission."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm, transform
from tilelang.ascend import transform as ascend
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower
from testing.ascend._ir import nodes

pytestmark = pytest.mark.pto
KEY = "tl.parallel_private_allocs"


def private_kernel(n=131, lanes=64, rows=1, column=False, dtype="int32"):
    p = (n + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((rows * p,), dtype), Seed: T.Tensor((rows * p,), dtype), Out: T.Tensor((rows * p,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((rows * p,), dtype)
            b = T.alloc_shared((rows * p,), dtype)
            T.copy(A, a)
            T.copy(Seed, b)
            with T.SimdVF(lanes=lanes):
                if rows == 1:
                    for i in T.Parallel(n):
                        acc = T.alloc_var(dtype)
                        tmp = T.alloc_local((1,), dtype)
                        other = T.alloc_local((1,), dtype)
                        acc = acc + a[i]
                        tmp[0] = acc
                        old = tmp[0]
                        tmp[0] = acc + T.Cast(dtype, 7)
                        other[0] = old * T.Cast(dtype, 2)
                        b[i] = other[0] + tmp[0]
                else:
                    for i, j in T.Parallel(n if column else rows, rows if column else n):
                        acc = T.alloc_var(dtype)
                        tmp = T.alloc_local((1,), dtype)
                        other = T.alloc_local((1,), dtype)
                        pos = j * p + i if column else i * p + j
                        acc = acc + a[pos]
                        tmp[0] = acc
                        old = tmp[0]
                        tmp[0] = acc + T.Cast(dtype, 7)
                        other[0] = old * T.Cast(dtype, 2)
                        b[pos] = other[0] + tmp[0]
            T.copy(b, Out)

    return main


def external_kernel(dtype="float32", mode=0):
    @T.prim_func
    def main(A: T.Tensor((192,), dtype), Out: T.Tensor((192,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((192,), dtype)
            b = T.alloc_shared((192,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                bias = T.alloc_local((1,), dtype)
                if mode == 0:
                    bias[0] = T.Cast(dtype, 3.1)
                elif mode == 1:
                    value = T.Cast(dtype, 3.1)
                    bias[0] = value
                else:
                    bias[0] = a[0]
                for i in T.Parallel(131):
                    tmp = T.alloc_local((1,), dtype)
                    tmp[0] = bias[0]
                    b[i] = tmp[0]
            T.copy(b, Out)

    return main


def _lower(f):
    target = normalize_pto_target("pto")
    with target:
        return lower(f, target=target)


def _capture(f, monkeypatch):
    captured = {}
    original = ascend.VectorizeParallelToPTO

    def factory():
        run = original()

        def wrapped(mod):
            captured["before"] = mod
            captured["after"] = run(mod)
            return captured["after"]

        return wrapped

    monkeypatch.setattr(ascend, "VectorizeParallelToPTO", factory)
    artifact = _lower(f)
    return captured, artifact


def test_private_ownership_and_snapshots(monkeypatch):
    f = private_kernel()
    roots = [x for x in nodes(f, tvm.tirx.For) if KEY in x.annotations]
    assert len(roots) == 1 and len(roots[0].annotations[KEY]) == 3
    captured, artifact = _capture(f, monkeypatch)
    old = list(roots[0].annotations[KEY])
    allocs = nodes(captured["after"], tvm.tirx.AllocBuffer)
    private = [a for a in allocs if "tl.vmi.alloc_kind" in a.annotations]
    assert sorted(str(a.annotations["tl.vmi.alloc_kind"]) for a in private) == ["data", "data_array", "data_array"]
    assert all(a.buffer.dtype == "int32x64" for a in private)
    assert all(tuple(int(s) for s in a.buffer.shape) == (1,) for a in private)
    assert not any(KEY in loop.annotations for loop in nodes(captured["after"], tvm.tirx.For))
    for buf in [a.buffer for a in allocs] + [l.buffer for l in nodes(captured["after"], tvm.tirx.BufferLoad)]:
        assert not any(buf.data.same_as(d) for d in old)
    # A scalar placeholder is not initialization; the actual zero is inside q.
    chunk = [x for x in nodes(captured["after"], tvm.tirx.For) if x.loop_var.name == "q"][0]
    zero_stores = [x for x in nodes(chunk.body, tvm.tirx.BufferStore) if x.buffer.scope() == "local.var" and x.buffer.dtype == "int32x64"]
    assert len(zero_stores) == 2
    # The old value is bound before the second write to tmp, not reloaded
    # when the later expression consumes it.
    sequence = list(chunk.body.seq)
    writes = [i for i, stmt in enumerate(sequence) if isinstance(stmt, tvm.tirx.BufferStore) and stmt.buffer.name == "tmp_v"]
    snapshots = [i for i, stmt in enumerate(sequence) if isinstance(stmt, tvm.tirx.Bind) and stmt.var.name == "old"]
    assert len(writes) == 2 and len(snapshots) == 1
    assert writes[0] < snapshots[0] < writes[1]
    assert "local_snapshot" in artifact.kernel_source
    assert "_tl_slot_0" in artifact.kernel_source


@pytest.mark.parametrize("column", [False, True])
def test_private_2d_layout(column):
    assert "pto.vmi.vstore" in _lower(private_kernel(rows=3, column=column)).kernel_source


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_external_array_keeps_scalar_type(dtype, mode):
    source = _lower(external_kernel(dtype, mode)).kernel_source
    assert "bias[0] = pto." in source
    assert {"float32": "pto.f32", "float16": "pto.f16", "bfloat16": "pto.bf16"}[dtype] in source
    assert "pto.vmi.vbrc" in source


def rejected_kernel(kind, dtype="float32", shape=(1,)):
    @T.prim_func
    def main(A: T.Tensor((64,), dtype), Out: T.Tensor((64,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((64,), dtype)
            b = T.alloc_shared((64,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                ext = T.alloc_local((1,), dtype)
                ext[0] = a[0]
                for i in T.Parallel(64):
                    if kind == "var":
                        acc = T.alloc_var(dtype)
                        acc = a[i]
                        b[i] = acc
                    elif kind == "external":
                        ext[0] = a[i]
                        b[i] = ext[0]
                    else:
                        tmp = T.alloc_local(shape, dtype)
                        if kind == "slot":
                            tmp[1] = a[i]
                        elif kind == "dynamic":
                            tmp[i] = a[i]
                        elif kind == "broadcast":
                            tmp[0] = ext[0]
                        elif len(shape) == 2:
                            tmp[0, 0] = a[i]
                        else:
                            tmp[0] = a[i]
                        if len(shape) == 2:
                            b[i] = tmp[0, 0]
                        else:
                            b[i] = tmp[0]
            T.copy(b, Out)

    return main


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_var_rejected_before_implicit_zero(dtype):
    with pytest.raises(Exception, match="broadcast dtype.*default zero initialization"):
        _lower(rejected_kernel("var", dtype))


@pytest.mark.parametrize(
    "kind,shape,reason",
    [
        ("external", (1,), "external ordinary storage write"),
        ("local", (2,), "requires shape"),
        ("local", (1, 1), "requires shape"),
        ("slot", (1,), "slot must be constant"),
        ("dynamic", (1,), "Local buffer.*indexed by T.Parallel"),
    ],
)
def test_ordinary_storage_rejects_invalid_access(kind, shape, reason):
    with pytest.raises(Exception, match=reason):
        _lower(rejected_kernel(kind, shape=shape))


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_array_copy_and_broadcast_admission(dtype):
    assert "pto.vmi.vload" in _lower(rejected_kernel("local", dtype)).kernel_source
    with pytest.raises(Exception, match="unsupported scalar-to-vector broadcast dtype"):
        _lower(rejected_kernel("broadcast", dtype))


def test_reserved_owner_key():
    with pytest.raises(ValueError, match="reserved"):
        T.Parallel(64, annotations={KEY: []})


def test_owner_survives_early_unroll():
    @T.prim_func
    def program(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            T.copy(A, a)
            for u in T.unroll(2, explicit=True):
                with T.SimdVF(lanes=64):
                    for i in T.Parallel(64):
                        tmp = T.alloc_local((1,), "float32")
                        tmp[0] = a[i] + T.Cast("float32", u)
                        b[i] = tmp[0]
            T.copy(b, B)

    mod = ascend.UnrollLoopSkipVF()(tvm.IRModule({"main": program}))
    roots = [x for x in nodes(mod, tvm.tirx.For) if KEY in x.annotations]
    assert len(roots) == 2
    owned_data = [x.annotations[KEY][0] for x in roots]
    assert not owned_data[0].same_as(owned_data[1])
    for root, data in zip(roots, owned_data):
        loads = [x for x in nodes(root.body, tvm.tirx.BufferLoad) if x.buffer.scope() == "local"]
        stores = [x for x in nodes(root.body, tvm.tirx.BufferStore) if x.buffer.scope() == "local"]
        assert loads and stores
        assert all(x.buffer.data.same_as(data) for x in loads + stores)
    assert "pto.vmi.vstore" in _lower(program).kernel_source


def test_layer_between_parallel_loops_is_external():
    @T.prim_func
    def program(A: T.Tensor((3, 64), "float32"), B: T.Tensor((3, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((3, 64), "float32")
            b = T.alloc_shared((3, 64), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(3):
                    scale = T.alloc_var("float32")
                    for j in T.Parallel(64):
                        tmp = T.alloc_local((1,), "float32")
                        tmp[0] = a[i, j] + scale
                        b[i, j] = tmp[0]
            T.copy(b, B)

    roots = [x for x in nodes(program, tvm.tirx.For) if KEY in x.annotations]
    assert len(roots) == 1 and len(roots[0].annotations[KEY]) == 1
    assert roots[0].annotations[KEY][0].name == "tmp"
    assert "pto.vmi.vstore" in _lower(program).kernel_source


def _raw_owned_module(metadata="valid", escape=False, standalone=False, initialized=True):
    tx = tvm.tirx
    data = tx.decl_buffer((1,), "float32", name="tmp", scope="local")
    ub = tx.decl_buffer((64,), "float32", name="out", scope="shared")
    i = tx.Var("i", "int32")
    stmts = []
    if standalone:
        stmts.append(tx.AllocBuffer(data))
    if initialized:
        stmts.append(tx.BufferStore(data, tx.FloatImm("float32", 2), [0]))
    stmts.append(tx.BufferStore(ub, tx.BufferLoad(data, [0]), [i]))
    annotations = {KEY: [data.data]}
    if metadata == "missing":
        annotations = {}
    elif metadata == "duplicate":
        annotations = {KEY: [data.data, data.data]}
    elif metadata == "unknown":
        annotations = {KEY: [tx.Var("ghost", "handle")]}
    elif metadata == "malformed":
        annotations = {KEY: 1}
    loop = tx.For(i, 0, 64, tx.ForKind.PARALLEL, (stmts[0] if len(stmts) == 1 else tx.SeqStmt(stmts)), annotations=annotations)
    body = [loop]
    if escape:
        body.append(tx.BufferStore(ub, tx.BufferLoad(data, [0]), [0]))
    block = tx.SBlock(
        [],
        [],
        [],
        "SIMD_VF",
        tx.AttrStmt("simdvf", "tl.simdvf_scope", 1, (body[0] if len(body) == 1 else tx.SeqStmt(body))),
        alloc_buffers=[ub] if standalone else [ub, data],
        annotations={"tl.simdvf_lanes": tx.IntImm("int64", 64)},
    )
    func = tx.PrimFunc([], block).with_attr("target", normalize_pto_target("pto")).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


@pytest.mark.parametrize(
    "metadata,reason",
    [
        ("duplicate", "duplicate private ownership"),
        ("unknown", "no actual allocation"),
        ("malformed", "expects Array"),
        ("missing", "external ordinary storage write"),
    ],
)
def test_raw_ownership_contract(metadata, reason):
    with pytest.raises(Exception, match=reason):
        mod = ascend.AscendLayoutInference()(_raw_owned_module(metadata))
        ascend.VerifyParallelToPTO()(mod)


def test_private_owner_cannot_escape():
    mod = ascend.AscendLayoutInference()(_raw_owned_module(escape=True))
    with pytest.raises(Exception, match="private owner violation"):
        ascend.VerifyParallelToPTO()(mod)


@pytest.mark.parametrize("standalone", [False, True])
def test_allocation_carriers_and_uninitialized_verify_boundary(standalone):
    mod = ascend.AscendLayoutInference()(_raw_owned_module(standalone=standalone, initialized=False))
    # The existing initialization pass remains warning-only, including when
    # the opt-out configuration is used. Neither path initializes the slot.
    checked = transform.VerifyBufferInit()(mod)
    tvm.ir.assert_structural_equal(checked, mod)
    with tvm.transform.PassContext(config={"tl.disable_buffer_init_check": True}):
        tvm.ir.assert_structural_equal(transform.VerifyBufferInit()(mod), mod)
    # This only promises Verify acceptance, not backend undef support.
    ascend.VerifyParallelToPTO()(mod)
    after = ascend.VectorizeParallelToPTO()(mod)
    allocs = [x for x in nodes(after, tvm.tirx.AllocBuffer) if "tl.vmi.alloc_kind" in x.annotations]
    assert len(allocs) == 1 and allocs[0].annotations["tl.vmi.alloc_kind"] == "data_array"
    assert not any(x.buffer.scope() == "local" for x in nodes(after, tvm.tirx.BufferStore))


def test_cross_unit_read_is_not_external_scalar():
    mod = _raw_owned_module()
    block = mod["main"].body
    root = block.body.body
    tx = tvm.tirx
    j = tx.Var("j", "int32")
    other = tx.For(j, 0, 64, tx.ForKind.PARALLEL, tx.BufferStore(block.alloc_buffers[0], tx.BufferLoad(block.alloc_buffers[1], [0]), [j]))
    new = tx.SBlock(
        [],
        [],
        [],
        "SIMD_VF",
        tx.AttrStmt("simdvf", "tl.simdvf_scope", 1, tx.SeqStmt([root, other])),
        alloc_buffers=block.alloc_buffers,
        annotations=block.annotations,
    )
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(new))
    mod = ascend.AscendLayoutInference()(mod)
    with pytest.raises(Exception, match="private owner violation"):
        ascend.VerifyParallelToPTO()(mod)


def test_failed_frontend_build_does_not_leak_candidates():
    with pytest.raises(Exception, match="deliberate construction failure"):

        @T.prim_func
        def broken():
            with T.Kernel(1), T.SimdVF(lanes=64):
                for _i in T.Parallel(64):
                    _tmp = T.alloc_local((1,), "float32")
                    raise ValueError("deliberate construction failure")

    f = private_kernel()
    roots = [x for x in nodes(f, tvm.tirx.For) if KEY in x.annotations]
    assert len(roots) == 1 and len(roots[0].annotations[KEY]) == 3


def test_explicit_vmi_not_recorded_as_ordinary():
    @T.prim_func
    def program():
        with T.Kernel(1), T.SimdVF(lanes=64):
            for _i in T.Parallel(64):
                _explicit = T.vmi.alloc_local((2,), T.vmi.vreg(64, "float32"))
                ordinary = T.alloc_local((1,), "float32")
                ordinary[0] = 1.0

    roots = [x for x in nodes(program, tvm.tirx.For) if KEY in x.annotations]
    assert len(roots) == 1 and len(roots[0].annotations[KEY]) == 1
    assert roots[0].annotations[KEY][0].name == "ordinary"


def test_singleton_private_store_uses_one_lane_mask():
    source = _lower(private_kernel(n=1)).kernel_source
    assert "pto.vmi.vstore" in source
    assert "pto.vmi.create_mask" in source


def test_direct_scalar_local_codegen_preserves_type():
    # No Parallel or promotion: this is also a regression for ordinary local arrays.
    tx = tvm.tirx
    build = tvm.ffi.get_global_func("target.build.tilelang_pto_without_compile")
    for dtype in ["float32", "float16", "bfloat16"]:
        buf = tx.decl_buffer((1,), dtype, name="bias", scope="local")
        value = tx.Var("value", dtype)
        body = tx.SeqStmt([tx.AllocBuffer(buf), tx.BufferStore(buf, tx.FloatImm(dtype, 3.1), [0]), tx.BufferStore(buf, value, [0])])
        target = normalize_pto_target("pto")
        f = tx.PrimFunc([value], body).with_attr("global_symbol", "main").with_attr("target", target)
        with target:
            source = build(tvm.IRModule({"main": f}), target).inspect_source()
        assert "bias[0] = pto.const(" in source
        assert "bias[0] = pto.cast(value," in source


def test_subcore_projection_preserves_private_owner(monkeypatch):
    @T.prim_func
    def program(A: T.Tensor((2, 64), "float32"), B: T.Tensor((2, 64), "float32")):
        with T.MixedKernel(1) as (_, sid):
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            T.copy(A[sid, :], a)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(64):
                    tmp = T.alloc_local((1,), "float32")
                    tmp[0] = a[i] + 1.0
                    b[i] = tmp[0]
            if sid == 0:
                T.copy(b, B[0, :])
            else:
                T.copy(b, B[1, :])

    captured, _ = _capture(program, monkeypatch)
    before = next(iter(captured["before"].functions.values()))
    roots = [x for x in nodes(before, tvm.tirx.For) if KEY in x.annotations]
    assert roots
    all_data = [buf.data for block in nodes(before, tvm.tirx.SBlock) for buf in block.alloc_buffers]
    for root in roots:
        for data in root.annotations[KEY]:
            assert any(data.same_as(actual) for actual in all_data)
            assert any(x.buffer.data.same_as(data) for x in nodes(root.body, tvm.tirx.BufferStore))


def test_private_bool_storage_rejected():
    with pytest.raises(Exception, match="unsupported private allocation dtype"):
        _lower(rejected_kernel("local", "bool"))


def test_runtime_uniform_slot_rejected_before_codegen():
    tx = tvm.tirx
    mod = _raw_owned_module()
    b = mod["main"].body
    root = b.body.body
    idx = tx.Var("slot", "int32")
    tmp = b.alloc_buffers[1]
    body = tx.SeqStmt([tx.BufferStore(tmp, tx.FloatImm("float32", 1), [idx]), root.body.seq[-1]])
    loop = tx.For(root.loop_var, 0, 64, tx.ForKind.PARALLEL, body, annotations=root.annotations)
    block = tx.SBlock(
        [], [], [], "SIMD_VF", tx.AttrStmt("simdvf", "tl.simdvf_scope", 1, loop), alloc_buffers=b.alloc_buffers, annotations=b.annotations
    )
    f = tx.PrimFunc([idx], block).with_attr("target", normalize_pto_target("pto"))
    mod = ascend.AscendLayoutInference()(tvm.IRModule({"main": f}))
    with pytest.raises(Exception, match="slot must be constant"):
        ascend.VerifyParallelToPTO()(mod)
