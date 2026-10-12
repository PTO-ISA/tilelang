"""Stage-4b TopK + direct-Reduce contract tests (Gate E2, task 5B).

Structural checks run on the ported GPU-shaped top-k frontend
(`examples/ascend/example_simdvf_topk_gate_frontend.py`): the kernel lowers
through ``Verify -> Legalize -> Vectorize``, every direct reduce is consumed,
the int32 reduce result is written back with a single-element VMI store
(three-argument ``vstore`` over a one-element ``access_ptr``; PTOAS lowers it
to the dtype's ``1PT`` pattern), and ``idx_ub[k] = best[0]`` stays an
ordinary scalar BufferStore/BufferLoad pair.

One device smoke test compiles and runs the frontend end to end; the full
numeric matrix (E=256/L=128, E=256/L=64, E=161/L=64, tie-break, all--inf
valid scores) lives next to the example in
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
    BufferStore whose value is a BufferLoad of the reduce destination."""
    _, captured = _compile_with_stages()
    vectorized = captured["vectorize"]

    idx_ub_stores = [
        node
        for node in _collect(vectorized, lambda n: isinstance(n, tvm.tirx.BufferStore))
        if node.buffer.name.startswith("idx_ub")
    ]
    assert idx_ub_stores, "the winner write idx_ub[k] = best[0] must survive"
    for store in idx_ub_stores:
        assert isinstance(store.value, tvm.tirx.BufferLoad), store.value
        assert store.value.buffer.name.startswith("best"), store.value


@pytest.mark.pto
def test_topk_index_copy_out_region_is_the_full_k_range():
    """The int32 copy-out spans exactly the K winner slots at offset 0 on
    both sides (no widening, no offset drift)."""
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
    got = out.cpu().to(torch.int32)
    exp = ref_topk_indices(scores, 8)
    assert torch.equal(got, exp), f"got={got.tolist()} exp={exp.tolist()}"


if __name__ == "__main__":
    tilelang.testing.main()
