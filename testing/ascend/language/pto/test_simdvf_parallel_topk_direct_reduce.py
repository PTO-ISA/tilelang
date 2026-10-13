"""Stage-4b TopK + direct-Reduce contract tests (Gate E2, task 5B).

Structural checks run on the ported GPU-shaped top-k frontend
(`examples/ascend/example_simdvf_topk_gate_frontend.py`): the kernel lowers
through ``Verify -> Legalize -> Vectorize``, every direct reduce is consumed,
the int32 reduce result is written back with a single-element VMI store
(three-argument ``vstore`` over a one-element ``access_ptr``; PTOAS lowers it
to the dtype's ``1PT`` pattern), and ``idx_ub[k] = best[0]`` stays an
ordinary scalar BufferStore/BufferLoad pair.

The int64 GM ``IdxOut`` ABI is checked structurally on the Vectorize output
module: the int32 winner slots feed a bare widen SIMD_VF that
writes an int64 UB scratch (VMI ``vintlv`` pair for K in {2,4,8,64},
per-element int64 ``Cast`` otherwise), and a *same-dtype* int64
``tl.tileop.ascend_copy`` publishes that scratch into ``IdxOut`` over exactly
K elements. ``idx_cand``/``best``/``idx_ub`` stay int32.

One device smoke test compiles and runs the frontend end to end; the full
numeric matrix ((E, L) x K x input class) lives next to the example in
``examples/ascend/test_simdvf_parallel_topk_frontend.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

import tilelang.testing
import tvm

os.environ.setdefault("TILELANG_DISABLE_DATA_RACE_CHECK", "1")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
EXAMPLES_ASCEND = os.path.join(REPO_ROOT, "examples/ascend")
if EXAMPLES_ASCEND not in sys.path:
    sys.path.insert(0, EXAMPLES_ASCEND)

from example_simdvf_topk_gate_frontend import (  # noqa: E402
    INT_MAX,
    _stage_capture,
    compile_frontend,
    ref_topk_indices,
)


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _collect(mod, predicate):
    found = []

    def visit(node):
        if predicate(node):
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


def _binds(mod):
    binds = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Bind):
            binds[node.var] = node.value

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return binds


def _vmi_calls(mod):
    calls = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            name = getattr(getattr(node, "op", None), "name", None)
            if isinstance(name, str) and name.startswith("tl.vmi."):
                calls.setdefault(name, []).append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return calls


def _buffers(mod):
    bufs = {}

    def visit(node):
        buf = None
        if isinstance(node, (tvm.tirx.BufferStore, tvm.tirx.BufferLoad)):
            buf = node.buffer
        elif isinstance(node, tvm.tirx.Call):
            for arg in node.args:
                if isinstance(arg, tvm.tirx.BufferLoad):
                    buf = arg.buffer
                    break
        if buf is not None:
            bufs.setdefault(buf.name, buf)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return bufs


def _regions(call):
    return [
        arg
        for arg in call.args
        if isinstance(arg, tvm.tirx.Call) and getattr(getattr(arg, "op", None), "name", None) == "tl.region"
    ]


def _reduce_result_vars(mod):
    result = set()
    for var, value in _binds(mod).items():
        if isinstance(value, tvm.tirx.Call) and getattr(getattr(value, "op", None), "name", None) in (
            "tl.vmi.vcmax",
            "tl.vmi.vcmin",
            "tl.vmi.vcadd",
        ):
            result.add(var)
    return result


def _compile_with_stages(num_experts=256, num_topk=8, lanes=128):
    with _stage_capture() as captured:
        compiled = compile_frontend(num_experts, num_topk, lanes)
    return compiled, captured


@pytest.mark.pto
def test_topk_lowers_through_verify_legalize_vectorize():
    compiled, captured = _compile_with_stages()
    assert captured["order"] == ["verify", "legalize", "vectorize"], captured["order"]
    vectorized = captured["vectorize"]
    assert not _collect(
        vectorized,
        lambda n: isinstance(n, tvm.tirx.Call)
        and getattr(getattr(n, "op", None), "name", None) == "tl.tileop.reduce",
    ), "Vectorize must consume every tl.tileop.reduce"
    names = _vmi_calls(vectorized)
    assert names.get("tl.vmi.vcmax") and names.get("tl.vmi.vcmin"), sorted(names)
    assert compiled.get_kernel_source()


@pytest.mark.pto
def test_topk_int32_reduce_writeback_is_a_single_element_vstore():
    _, captured = _compile_with_stages()
    vectorized = captured["vectorize"]

    result_vars = _reduce_result_vars(vectorized)
    assert len(result_vars) == 2, f"expected the two tournament reduces, got {len(result_vars)}"
    stores = [
        call
        for call in _vmi_calls(vectorized).get("tl.vmi.vstore", [])
        if call.args and isinstance(call.args[0], tvm.tirx.Var) and call.args[0] in result_vars
    ]
    assert len(stores) == 2, f"each reduce writes back exactly once, got {len(stores)}"
    for call in stores:
        assert len(call.args) == 3, (
            f"the write-back must store the single-element result directly: {call}"
        )
        ptr = call.args[1]
        assert isinstance(ptr, tvm.tirx.Call) and getattr(ptr.op, "name", None) == "tl.access_ptr"
        assert int(ptr.args[1]) == 1, f"write-back access_ptr extent must be 1: {ptr}"
    # No VL-wide broadcast of any reduce result anywhere.
    for var, value in _binds(vectorized).items():
        if isinstance(value, tvm.tirx.Call) and getattr(getattr(value, "op", None), "name", None) == "tl.vmi.vbrc":
            if value.args and isinstance(value.args[0], tvm.tirx.Var) and value.args[0] in result_vars:
                raise AssertionError(f"reduce result broadcast into {var}")


@pytest.mark.pto
def test_topk_winner_store_is_a_plain_scalar_buffer_store():
    """``idx_ub[k] = best[0]`` keeps the ordinary scalar forms: a
    BufferStore whose value is a BufferLoad of the reduce destination.

    Matches the exact int32 winner buffer: with the int64 ABI the expansion
    also emits ``idx_ub_widen_pad[...] = idx_ub[...]`` into a differently
    named pad buffer, which a name-prefix match would misread as the winner
    store."""
    _, captured = _compile_with_stages()
    vectorized = captured["vectorize"]

    idx_ub_stores = [
        node
        for node in _collect(vectorized, lambda n: isinstance(n, tvm.tirx.BufferStore))
        if node.buffer.name == "idx_ub"
    ]
    assert idx_ub_stores, "the winner write idx_ub[k] = best[0] must survive"
    for store in idx_ub_stores:
        assert isinstance(store.value, tvm.tirx.BufferLoad), store.value
        assert store.value.buffer.name.startswith("best"), store.value


@pytest.mark.pto
def test_topk_index_copy_out_region_is_the_full_k_range():
    """The copy-out spans exactly the K winner slots at offset 0 on both
    sides (no offset drift). The int64 ABI binding of both endpoints is
    checked in ``test_topk_copy_out_widens_int32_winners_into_int64_scratch``."""
    _, captured = _compile_with_stages()
    vectorized = captured["vectorize"]

    copies = _collect(
        vectorized,
        lambda n: isinstance(n, tvm.tirx.Call)
        and getattr(getattr(n, "op", None), "name", None) == "tl.tileop.ascend_copy",
    )
    assert copies, "the idx_ub -> IdxOut copy must survive"
    regions = []
    for call in copies:
        for arg in call.args:
            if isinstance(arg, tvm.tirx.Call) and getattr(getattr(arg, "op", None), "name", None) == "tl.region":
                regions.append(arg)
    assert len(regions) >= 2, f"expected source and target regions, got {regions}"
    extents = [int(r.args[2]) for r in regions if isinstance(r.args[2], tvm.tirx.IntImm)]
    assert 8 in extents, f"the copy must span K=8 elements, got extents {extents}"


@pytest.mark.pto
@pytest.mark.parametrize("num_topk", [1, 6, 8], ids=["k1", "k6", "k8"])
def test_topk_copy_out_widens_int32_winners_into_int64_scratch(num_topk):
    """The int64 GM ABI is produced by widening inside UB, not by an int32
    copy: a bare widen SIMD_VF reads the int32 winner slots and writes an
    int64 shared.dyn scratch (VMI ``vintlv`` pair for K=8, per-element int64
    ``Cast`` stores for the K=1/K=6 serial fallback), then a *same-dtype*
    int64 ``tl.tileop.ascend_copy`` publishes that scratch into ``IdxOut``
    with both regions spanning exactly K elements."""
    _, captured = _compile_with_stages(num_experts=256, num_topk=num_topk, lanes=128)
    vectorized = captured["vectorize"]

    bufs = _buffers(vectorized)
    assert "idx_ub" in bufs, "the winner buffer must survive"
    for name in ("idx_ub", "idx_cand", "best"):
        assert name in bufs, f"{name} must survive"
        assert str(bufs[name].dtype) == "int32", f"{name} must stay int32, got {bufs[name].dtype}"

    copies = _collect(
        vectorized,
        lambda n: isinstance(n, tvm.tirx.Call)
        and getattr(getattr(n, "op", None), "name", None) == "tl.tileop.ascend_copy",
    )
    out_copies = []
    for call in copies:
        regions = _regions(call)
        if len(regions) == 2 and regions[1].args[0].buffer.scope() == "global":
            out_copies.append((call, regions[0], regions[1]))
    assert len(out_copies) == 1, f"expected exactly one copy into GM, got {len(out_copies)}"
    _, src_region, dst_region = out_copies[0]
    src_buf = src_region.args[0].buffer
    dst_buf = dst_region.args[0].buffer
    assert str(dst_buf.dtype) == "int64", f"IdxOut must be int64, got {dst_buf.dtype}"
    assert dst_buf.scope() == "global", dst_buf.scope()
    assert str(src_buf.dtype) == "int64", f"the copy source must be an int64 scratch, got {src_buf.dtype}"
    assert src_buf.scope() == "shared.dyn", src_buf.scope()
    for name, region, access in (("source", src_region, 1), ("target", dst_region, 2)):
        assert int(region.args[0].indices[0]) == 0, f"{name} region offset must stay 0: {region}"
        assert int(region.args[1]) == access, f"{name} region access mask: {region}"
        assert int(region.args[2]) == num_topk, f"{name} extent must be K={num_topk}: {region}"

    # The widen VF reads the int32 winner slots and writes the int64 scratch.
    widen_blocks = [
        node
        for node in _collect(vectorized, lambda n: isinstance(n, tvm.tirx.SBlock))
        if "tl.widen_vmi" in (node.annotations or {})
    ]
    assert len(widen_blocks) == 1, f"expected one widen VF, got {len(widen_blocks)}"
    block = widen_blocks[0]
    assert bufs["idx_ub"].data in {r.buffer.data for r in block.reads}, block.reads
    assert src_buf.data in {r.buffer.data for r in block.writes}, block.writes

    binds = _binds(vectorized)
    vmi = _vmi_calls(vectorized)
    if num_topk == 8:
        assert vmi.get("tl.vmi.vintlv"), "the VMI widening pair must exist for K=8"
        view_bufs = []
        for store in vmi.get("tl.vmi.vstore", []):
            ptr = store.args[1]
            ptr = binds.get(ptr, ptr)
            if not (isinstance(ptr, tvm.tirx.Call) and getattr(ptr.op, "name", None) == "tl.access_ptr"):
                continue
            load = ptr.args[0]
            if isinstance(load, tvm.tirx.BufferLoad) and load.buffer.data == src_buf.data:
                view_bufs.append(load.buffer)
        assert view_bufs, "a VMI vstore must write the scratch through its int32 view"
        assert all(str(buf.dtype) == "int32" for buf in view_bufs), view_bufs
    else:
        casts = [
            node
            for node in _collect(vectorized, lambda n: isinstance(n, tvm.tirx.BufferStore))
            if node.buffer.data == src_buf.data
            and str(node.buffer.dtype) == "int64"
            and isinstance(node.value, tvm.tirx.Cast)
            and str(node.value.dtype) == "int64"
            and isinstance(node.value.value, tvm.tirx.BufferLoad)
            and node.value.value.buffer.data == bufs["idx_ub"].data
        ]
        assert casts, "the serial fallback must store int64 Casts of the int32 winner slots"


@pytest.mark.pto
def test_topk_device_smoke_numeric():
    """End-to-end device smoke: the ported frontend runs and matches the
    reference tournament on a random input (full matrix lives in
    examples/ascend/test_simdvf_parallel_topk_frontend.py)."""
    import torch

    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu unavailable")
    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime unavailable: {err}")

    torch.manual_seed(7)
    scores = torch.randn(256, dtype=torch.float32)
    compiled = compile_frontend(256, 8, 128)
    out = compiled(scores.to(torch.device("npu")))
    torch.npu.synchronize()
    # int64 GM ABI: assert the dtype first, then compare full int64 values
    # (no narrowing cast back to int32).
    assert out.dtype == torch.int64, out.dtype
    got = out.cpu()
    exp = ref_topk_indices(scores, 8)
    assert exp.dtype == torch.int64, exp.dtype
    assert torch.equal(got, exp), f"got={got.tolist()} exp={exp.tolist()}"
    assert bool(((got >> 32) == 0).all()), got


if __name__ == "__main__":
    tilelang.testing.main()
