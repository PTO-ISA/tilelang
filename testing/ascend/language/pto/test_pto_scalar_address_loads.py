"""Scalar reads used in PTO Parallel addresses must observe each VF's entry value."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.target import normalize_pto_target
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq
from test_pto_parallel_gather_scatter import capture_passes

pytestmark = pytest.mark.pto


@pytest.mark.parametrize("read_slots,write_slots", [(1, 1), (2, 2), (3, 3), (7, 7), (2, 3)])
def test_dynamic_expand(read_slots, write_slots, monkeypatch):
    n = T.dynamic("n")

    @T.prim_func
    def expand(X: T.Tensor((n, 1280), "bfloat16"), Y: T.Tensor((n, 4, 1280), "bfloat16")):
        with T.Kernel(4) as core:
            x = T.alloc_shared((256,), "bfloat16")
            y = T.alloc_shared((4, 256), "bfloat16")
            T.annotate_buffer_versions({x: read_slots, y: write_slots})
            for row, col in T.Persistent([n, 5], 4, core, group_size=1, num_stages=write_slots):
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
    read, write = (nodes(ptr, tirx.BufferLoad)[0] for ptr in (load.args[0], store.args[1]))
    if write_slots == 1:
        assert (len(read.indices), len(write.indices)) == (1, 2)
    else:
        (snapshot,) = [b for b in nodes(ir, tirx.Bind) if isinstance(b.value, tirx.BufferLoad) and b.value.buffer.scope() == "local.var"]
        analyzer = tvm.arith.Analyzer()
        assert analyzer.can_prove_equal(read.indices[0], snapshot.var % read_slots)
        assert analyzer.can_prove_equal(write.indices[0], snapshot.var % write_slots)


def _module(body, buffers, params=()):
    func = kernel(body, buffers=buffers, params=params)["main"]
    return tvm.IRModule({"main": func.with_attr("target", normalize_pto_target("pto"))})


def _parallel(src, dst, index):
    lane = tirx.Var("lane", "int32")
    return tirx.For(lane, 0, 64, tirx.ForKind.PARALLEL, tirx.BufferStore(dst, src[index, lane], [index, lane]))


def _buffers():
    row = tirx.decl_buffer((1,), "int32", scope="local.var", name="row")
    src, dst = [tirx.decl_buffer((3, 64), "float32", scope="shared.dyn", name=name) for name in ("src", "dst")]
    return row, src, dst


def test_refresh_user_index_after_store_and_inside_guard():
    row, src, dst = _buffers()
    active = tirx.Var("active", "bool")

    def vf():
        block = tirx.SBlock([], [], [], "SIMD_VF", _parallel(src, dst, row[0]))
        return tirx.SBlockRealize([], active, block)

    mod = _module(seq(tirx.BufferStore(row, 0, [0]), vf(), tirx.BufferStore(row, 1, [0]), vf()), [row, src, dst], [active])
    result = transform.HoistPTOScalarAddressLoads()(mod)
    init, first, advance, second = result["main"].body.block.body.seq
    assert init.buffer.same_as(row) and advance.buffer.same_as(row) and int(advance.value) == 1
    snapshots = []
    for guard in (first, second):
        assert guard.condition.same_as(active)
        binding, vf = guard.then_case.seq
        assert binding.value.buffer.same_as(row) and int(vf.predicate) == 1
        (store,) = nodes(vf, tirx.BufferStore)
        assert store.indices[0].same_as(binding.var) and store.value.indices[0].same_as(binding.var)
        snapshots.append(binding.var)
    assert not snapshots[0].same_as(snapshots[1])


@pytest.mark.parametrize(
    "case",
    ["store", "view", "data", "address_of", "access_ptr", "match", "predicate_escape", "private", "vf_local", "unallocated",
     "volatile", "conditional_init", "predicated_load", "serial_vf"],
)
def test_unsafe_or_inapplicable_load_stays_in_place(case):
    row, src, dst = _buffers()
    active = tirx.Var("active", "bool")
    buffers = [src, dst]
    outer, inner = [], []
    init = tirx.BufferStore(row, 0, [0])
    if case == "volatile":
        outer.append(tirx.AllocBuffer(row, annotations={"tirx.volatile": True}))
    elif case not in ("vf_local", "unallocated"):
        buffers.append(row)
    if case == "vf_local":
        inner.append(init)
    else:
        outer.append(tirx.IfThenElse(active, init, None) if case == "conditional_init" else init)

    if case in ("store", "view", "data", "address_of", "access_ptr", "match"):
        alias = row
        if case == "view":
            alias = tirx.decl_buffer((1,), "int32", data=row.data)
            outer.append(tirx.DeclBuffer(alias))
        elif case != "store":
            alias = tirx.decl_buffer((1,), "int32", scope="local.var")
            if case != "match":
                addr = row.data
                if case == "address_of":
                    addr = T.address_of(row[0])
                elif case == "access_ptr":
                    addr = T.access_ptr(row[0], "rw", extent=1)
                outer.extend([tirx.Bind(alias.data, addr), tirx.DeclBuffer(alias)])
        inner.append(tirx.BufferStore(alias, 1, [0]))
    if case == "predicate_escape":
        inner.append(tirx.BufferStore(dst, 0.0, [0, 0], predicate=tirx.call_extern("bool", "advance", row.data)))
    index = tirx.BufferLoad(row, [0], predicate=active) if case == "predicated_load" else row[0]
    loop = _parallel(src, dst, index)
    if case in ("private", "serial_vf"):
        loop = tirx.For(loop.loop_var, 0, 64, tirx.ForKind.SERIAL if case == "serial_vf" else loop.kind, loop.body,
                        annotations={"tl.parallel_private_allocs": [row.data]} if case == "private" else {})
    block = tirx.SBlock([], [], [], "SIMD_VF", seq(*inner, loop), alloc_buffers=[row] if case == "vf_local" else [])
    if case == "match":
        match = tirx.MatchBufferRegion(alias, tirx.BufferRegion(row, [tvm.ir.Range(0, 1)]))
        block = tirx.SBlock([], [], [], "wrapper", block, match_buffers=[match])
    params = [active, row] if case == "unallocated" else [active]
    mod = _module(seq(*outer, block), buffers, params)
    tvm.ir.assert_structural_equal(mod, transform.HoistPTOScalarAddressLoads()(mod))
