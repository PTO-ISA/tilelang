"""The reuse contract survives scheduled lowering and is consumed by packing."""

import pytest
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.target import normalize_pto_target
from tvm import tirx
from testing.ascend._ir import nodes, schedule
from testing.ascend.auto_schedule._reuse_utils import _make_cyclic_scalar_program, _make_pto_reduce_reuse_program


@pytest.mark.pto
@pytest.mark.parametrize(
    "options, may_alias",
    [
        ({"extent": 128}, True),
        ({}, True),
        ({"extent": 3}, True),
        ({"extent": 63, "lanes": 64, "dtype": "float16", "out_dtype": "float16"}, True),
        ({"dim": 0}, True),
        ({"clear": False}, True),
        ({"versions": 2}, True),
        ({"versions": 3}, True),
        ({"source_offset": 3}, True),
        ({"interleave": True}, False),
        ({"dim": 0, "interleave": True}, False),
        ({"keep_padding": True}, False),
        ({"complete_read": False}, False),
        ({"complete_read": False, "versions": 2}, False),
    ],
    ids=[
        "full",
        "tail",
        "short",
        "physical-read-width",
        "independent-outputs",
        "old-output",
        "two-versions",
        "three-versions",
        "offset",
        "interleaved-source",
        "interleaved-outputs",
        "live-in-padding",
        "uncompleted-vf",
        "uncompleted-versioned-vf",
    ],
)
def test_direct_reduce_reuse_follows_storage_lifetimes(options, may_alias):
    before = schedule(_make_pto_reduce_reuse_program(**options), through="ResolveCore", auto_schedule=False)
    before = tvm.IRModule({gvar: func.with_attr("target", normalize_pto_target("pto")) for gvar, func in before.functions.items()})
    snapshot = transform.InsertSync()(before)
    root = next(block for block in nodes(snapshot, tirx.SBlock) if block.name_hint == "tilelang_root")
    aliases = root.annotations["tl.buffer_alias_map"]
    storage = {buffer.name: buffer.data for buffer in root.alloc_buffers}
    assert (storage["later"] in aliases[storage["a"]]) == may_alias


def test_alias_contract_lifetime():
    snapshots = {}
    wanted = {
        "tl.InsertSync",
        "tl.LowerScheduledTIR",
        "tl.MergeUBAllocations",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(_make_cyclic_scalar_program(), target="ascend")

    assert any("tl.buffer_alias_map" in block.annotations for block in nodes(snapshots["tl.InsertSync"], tirx.SBlock))
    assert any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.LowerScheduledTIR"], tirx.AttrStmt))
    assert not any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.MergeUBAllocations"], tirx.AttrStmt))


def test_fixed_versions_allow_ub_reuse_above_raw_capacity():
    @T.prim_func
    def main(x: T.Tensor((8,), "int32"), out: T.Tensor((8, 2), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((32768,), "int32")
            b = T.alloc_shared((32768,), "int32")
            T.annotate_buffer_versions({a: 1, b: 1})
            for i in T.Pipelined(8, num_stages=1):
                with T.Stage(0):
                    a[i * 4096] = x[i]
                with T.Stage(0):
                    out[i, 0] = a[i * 4096]
                with T.Stage(0):
                    b[i * 4096] = x[i] + 1
                with T.Stage(0):
                    out[i, 1] = b[i * 4096]

    allocations = []

    @tvm.ir.instrument.pass_instrument
    class CaptureAllocations:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                allocations.extend(
                    node.buffer
                    for func in mod.functions.values()
                    for node in nodes(func, tirx.AllocBuffer)
                    if node.buffer.scope() == "shared.dyn"
                )

    with tvm.transform.PassContext(instruments=[CaptureAllocations()]):
        tilelang.lower(main, target="ascend", enable_device_compile=False)

    # The raw sum is 256 KiB, but disjoint lifetimes need only 128 KiB of UB.
    (allocation,) = allocations
    assert allocation.dtype == "uint8"
    assert tuple(int(dim) for dim in allocation.shape) == (131072,)


@pytest.mark.pto
@pytest.mark.parametrize("disable_reuse", [False, True])
def test_direct_reduce_reuse_reaches_physical_allocation(disable_reuse):
    """The tail fits its owner and both lifetimes use the same physical arena."""
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                snapshots["packed"] = mod

    program = _make_pto_reduce_reuse_program()
    target = normalize_pto_target("pto")
    with target, tvm.transform.PassContext(instruments=[Capture()], config={"tl.disable_shared_memory_reuse": disable_reuse}):
        tilelang.lower(program, target=target, enable_device_compile=False)

    bodies = [func.body for func in snapshots["packed"].functions.values()]
    source_offsets, later_offsets = [], []
    analyzer = tvm.arith.Analyzer()
    for body in bodies:
        for call in nodes(body, tirx.Call):
            if isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.vmi.vload":
                source_offsets.append(int(analyzer.simplify(call.args[0].args[2])))
        for store in nodes(body, tirx.BufferStore):
            if store.buffer.name == "later":
                base = tirx.stmt_functor.substitute(
                    store.indices[0],
                    {var: tirx.const(0, var.dtype) for var in tirx.analysis.undefined_vars(store.indices[0])},
                )
                base = analyzer.simplify(base)
                if isinstance(base, tirx.Ramp):
                    base = base.base
                elif isinstance(base, tirx.Broadcast):
                    base = base.value
                later_offsets.append(int(analyzer.simplify(base)))

    assert len(source_offsets) == 2 and source_offsets[1] - source_offsets[0] == 128
    assert len(later_offsets) == 1
    assert (later_offsets[0] == source_offsets[0]) == (not disable_reuse)
