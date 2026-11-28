"""Local indices keep the value and lane dependence of their defining load."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from test_pto_parallel_gather_scatter import capture_passes
from testing.ascend._ir import calls, nodes
from tilelang.ascend import transform as ascend
from tilelang.ascend.target import normalize_pto_target


def local_index_kernel(source, access, dtype="int32"):
    @T.prim_func
    def main(A: T.Tensor((1024,), dtype), I: T.Tensor((256,), "int32"), C: T.Tensor((1024,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((1024,), dtype)
            idx_ub = T.alloc_shared((256,), "int32")
            out = T.alloc_shared((1024,), dtype)
            T.copy(A, a)
            T.copy(I, idx_ub)
            T.copy(C, out)
            with T.SimdVF(lanes=128):
                i = T.alloc_var("int32", init=0)
                i = idx_ub[0]
                for j in T.Parallel(256):
                    if source == "direct":
                        out[j] = a[idx_ub[i]]
                    elif source == "external":
                        out[(i % 3) * 256 + j] = a[(i % 3) * 256 + j]
                    elif source in ("copy", "branch", "loop"):
                        base = T.bind(idx_ub[0])
                        p = T.alloc_local((1,), "int32")
                        p[0] = base
                        if source == "copy":
                            q = T.alloc_var("int32")
                            alias = T.bind(p[0])
                            q = alias
                            p[0] = j
                            out[j] = a[q + j]
                        elif source == "branch":
                            if j < 150:
                                p[0] = idx_ub[1]
                                out[j] = a[p[0] + j]
                        else:
                            for k in T.serial(2):
                                p[0] = base
                                out[k * 256 + j] = a[p[0] + j]
                            out[512 + j] = a[p[0] + j]
                    else:
                        idx = T.alloc_var("int32")
                        if source in ("private_uniform", "snapshot"):
                            idx = idx_ub[i]
                        else:
                            idx = idx_ub[j]
                        if source == "snapshot":
                            idx_ub[j] = j
                        if access == "guarded":
                            if j < 150:
                                out[j] = a[idx]
                        elif access == "read":
                            out[j] = a[idx]
                        else:
                            out[idx] = a[j]
            T.copy(out, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "source,access,gathers,scatters",
    [
        ("direct", "read", 0, 0),
        ("external", "copy", 0, 0),
        ("private_uniform", "read", 0, 0),
        ("private_varying", "read", 1, 0),
        ("private_varying", "write", 0, 1),
        ("copy", "read", 0, 0),
        ("branch", "read", 0, 0),
        ("loop", "read", 0, 0),
    ],
)
def test_local_index_compiles(source, access, gathers, scatters, monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    with capture_passes() as captured:
        tilelang.compile(local_index_kernel(source, access), target="pto")
    ir = captured["VectorizeParallelToPTO_after"]
    assert len(calls(ir, "tl.vmi.vgather")) == gathers
    assert len(calls(ir, "tl.vmi.vscatter")) == scatters
    if source == "external":
        assert len(calls(ir, "tl.vmi.vload")) == len(calls(ir, "tl.vmi.vstore")) == 1


@pytest.mark.pto
@pytest.mark.parametrize("access", ["read", "guarded"])
def test_uniform_private_index_into_b16(access, monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    with capture_passes() as captured:
        tilelang.compile(local_index_kernel("private_uniform", access, "float16"), target="pto")
    ir = captured["VectorizeParallelToPTO_after"]
    assert not calls(ir, "tl.vmi.vgather")
    assert not calls(ir, "tl.vmi.vscatter")


@pytest.mark.pto
def test_private_index_observes_source_before_overwrite(monkeypatch):
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    with capture_passes() as captured:
        tilelang.compile(local_index_kernel("snapshot", "read"), target="pto")
    ir = captured["VectorizeParallelToPTO_after"]
    (chunk,) = [loop for loop in nodes(ir, tirx.For) if loop.loop_var.name == "q"]
    seq = list(chunk.body.seq)
    (saved,) = [
        stmt
        for stmt in seq
        if isinstance(stmt, tirx.Bind) and isinstance(stmt.value, tirx.BufferLoad) and stmt.value.buffer.name == "idx_ub"
    ]
    (overwrite,) = [
        stmt
        for stmt in seq
        if isinstance(stmt, tirx.Evaluate) and calls(stmt, "tl.vmi.vstore") and stmt.value.args[1].args[0].buffer.name == "idx_ub"
    ]
    (read,) = [
        stmt for stmt in seq if isinstance(stmt, tirx.Bind) and isinstance(stmt.value, tirx.BufferLoad) and stmt.value.buffer.name == "a"
    ]
    assert read.value.indices[0].same_as(saved.var)
    assert seq.index(saved) < seq.index(overwrite) < seq.index(read)


@pytest.mark.pto
def test_private_index_cannot_escape_through_a_bound_pointer():
    idx = tirx.decl_buffer((1,), "int32", scope="local", name="idx")
    table = tirx.decl_buffer((256,), "int32", scope="shared", name="table")
    ptr = tirx.Var("alias_ptr", idx.data.type_annotation)
    alias = tirx.decl_buffer((1,), "int32", data=ptr, scope="local")
    j = tirx.Var("j", "int32")
    body = tirx.SeqStmt(
        [
            tirx.BufferStore(idx, 3, [0]),
            tirx.Bind(ptr, idx.data),
            tirx.BufferStore(alias, 11, [0]),
            tirx.BufferStore(table, table[idx[0] + j], [j]),
        ]
    )
    loop = tirx.For(j, 0, 128, tirx.ForKind.PARALLEL, body, annotations={"tl.parallel_private_allocs": [idx.data]})
    vf = tirx.SBlock([], [], [], "SIMD_VF", loop, alloc_buffers=[idx, table], annotations={"tl.simdvf_lanes": 128})
    func = tirx.PrimFunc([], vf).with_attr("target", normalize_pto_target("pto"))
    with pytest.raises(tvm.error.InternalError, match="private address escape"):
        ascend.VerifyParallelToPTO()(tvm.IRModule({"main": func}))


@pytest.mark.pto
@pytest.mark.parametrize("x_slots,y_slots", [(1, 1), (2, 2), (3, 3), (7, 7), (2, 3)])
def test_persistent_expand_compiles(x_slots, y_slots, monkeypatch):
    stages = max(x_slots, y_slots)
    n = T.dynamic("n")

    @T.prim_func
    def expand(X: T.Tensor((n, 1280), "bfloat16"), Y: T.Tensor((n, 4, 1280), "bfloat16")):
        with T.Kernel(4) as core:
            x = T.alloc_shared((256,), "bfloat16")
            y = T.alloc_shared((4, 256), "bfloat16")
            T.annotate_buffer_versions({x: x_slots, y: y_slots})
            for row, col in T.Persistent([n, 5], 4, core, group_size=1, num_stages=stages):
                T.copy(X[row, col * 256], x)
                with T.SimdVF(lanes=128):
                    for m, j in T.Parallel(4, 256):
                        y[m, j] = x[j]
                T.copy(y, Y[row, 0, col * 256])

    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    with capture_passes() as captured:
        tilelang.compile(expand, target="pto")
    ir = captured["VectorizeParallelToPTO_after"]["expand"]
    (load,) = calls(ir, "tl.vmi.vload")
    (store,) = calls(ir, "tl.vmi.vstore")
    assert not calls(ir, "tl.vmi.vgather") and not calls(ir, "tl.vmi.vscatter")
    if stages > 1:
        counters = []
        for ptr, slots in ((load.args[0], x_slots), (store.args[1], y_slots)):
            slot = ptr.args[0].indices[0]
            (counter,) = nodes(slot, tirx.BufferLoad)
            assert counter.buffer.scope() == "local.var"
            assert tvm.arith.Analyzer().can_prove_equal(slot, counter % slots)
            counters.append(counter.buffer.data)
        assert counters[0].same_as(counters[1])
