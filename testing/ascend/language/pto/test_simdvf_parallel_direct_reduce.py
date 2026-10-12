"""Stage-4a PTO direct-Reduce tests (Gate E1, task 5A).

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module produced by
``VectorizeParallelToPTO`` (no PTOAS/Bisheng device compile, no device
execution, unless a case explicitly says otherwise).

Covered contract (design stage 4A / direct ``tl.tileop.reduce``):

* the reduce is a top-level statement of a converting ``SIMD_VF`` region,
  outside the ``T.Parallel`` units, and may be repeated by an enclosing serial
  loop;
* the source is a static 1-D buffer, the destination a one-element buffer,
  ``dim=0``, ``batch=1``, ``clear=True``, no ``nan_propagate``;
* lowering to VMI happens inside Vectorize: ceil-chunk full-width ``vload``,
  masked fold with keep-prior ``vsel``, one cross-lane ``vcmax``/``vcmin``/
  ``vcadd(reassoc=True)``, and a **single-element** VMI store of the result
  into ``destination[0]`` (a 3-argument ``vstore`` over a 1-element
  ``access_ptr``; no VL-wide broadcast, no VL-wide lane0 mask; PTOAS lowers
  it to the dtype's ``1PT`` store). The per-chunk input domain masks stay;
* reads of ``destination[0]`` elsewhere in the region stay ordinary
  ``BufferLoad`` accesses; the same-address store/load ordering is the
  toolchain's responsibility (the PTOAS vecscope mem_bar pass), not something
  Vectorize rewrites. Writes to the destination after the reduce are not
  rejected by any TileLang-side "producer" rule;
* the reduce may appear before the first ``T.Parallel`` unit: the lowering
  uses the region-level ``tl.simdvf_lanes`` annotation, never per-unit state;
* a lone reduce (no ``T.Parallel`` in the region), a reduce inside a
  ``T.Parallel`` unit, non-static/non-1-D sources and multi-element
  destinations are rejected with diagnostics.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
from tilelang.language.common import evaluate
from tilelang.language.utils import region
import tvm
from tvm import tirx


def _ceil_to_lanes(extent: int, lanes: int) -> int:
    return (extent + lanes - 1) // lanes * lanes


@contextmanager
def _stage_capture():
    """Capture the modules around Legalize and after Vectorize."""
    captured = {}
    original_legalize = ascend_transform.LegalizeParallelToPTO
    original_vectorize = ascend_transform.VectorizeParallelToPTO

    def capturing_legalize_factory():
        legalize_pass = original_legalize()

        def run(mod):
            result = legalize_pass(mod)
            captured["legalize"] = result
            return result

        return run

    def capturing_vectorize_factory():
        vectorize_pass = original_vectorize()

        def run(mod):
            result = vectorize_pass(mod)
            captured["vectorize"] = result
            return result

        return run

    ascend_transform.LegalizeParallelToPTO = capturing_legalize_factory
    ascend_transform.VectorizeParallelToPTO = capturing_vectorize_factory
    try:
        yield captured
    finally:
        ascend_transform.LegalizeParallelToPTO = original_legalize
        ascend_transform.VectorizeParallelToPTO = original_vectorize


def _lower_capture(func, *, expect_error=False):
    """Lower *func* for PTO and return (captured_stages, error)."""
    target = normalize_pto_target("pto")
    error = None
    with _stage_capture() as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured, error


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


def _vmi_calls(mod):
    calls = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op = getattr(node, "op", None)
            name = getattr(op, "name", None)
            if isinstance(name, str) and name.startswith("tl.vmi."):
                calls.setdefault(name, []).append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return calls


def _reduce_calls(mod):
    return _collect(
        mod,
        lambda n: isinstance(n, tvm.tirx.Call)
        and getattr(getattr(n, "op", None), "name", None) == "tl.tileop.reduce",
    )


def _binds(mod):
    binds = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Bind):
            binds[node.var] = node.value

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return binds


def _rewrite_reduce_eval(mod, mode):
    """Return a copy of *mod* with the direct-reduce ``Evaluate(Call)``
    rewritten into a non-standalone shape:

    ``mode="bind"``   -> ``Bind(var, Call(tl.tileop.reduce, ...))``
    ``mode="nested"`` -> ``Evaluate(call_extern(..., Call(tl.tileop.reduce)))``

    Used to prove that Verify rejects reduce calls that are bound or embedded
    instead of standing alone as an ``Evaluate`` statement.
    """

    def mutate(node):
        if isinstance(node, tvm.tirx.Evaluate) and isinstance(node.value, tvm.tirx.Call):
            if getattr(node.value.op, "name", None) == "tl.tileop.reduce":
                if mode == "bind":
                    var = tvm.tirx.Var("red_bind", node.value.dtype)
                    return tvm.tirx.Bind(var, node.value)
                if mode == "nested":
                    wrapped = tvm.tirx.call_extern("handle", "tl_nested_reduce_probe", node.value)
                    return tvm.tirx.Evaluate(wrapped)
        return None

    key, func = next(iter(mod.functions.items()))
    new_body = tvm.tirx.stmt_functor.ir_transform(func.body, None, mutate, ["tirx.Evaluate"])
    return tvm.IRModule({key: func.with_body(new_body)})


def _run_verify(mod):
    """Run VerifyParallelToPTO directly (no later pass can be the thrower)."""
    return ascend_transform.VerifyParallelToPTO()(mod)


def _scalar_load_binds(mod, buffer_name):
    """Binds whose value is a plain lane-uniform BufferLoad of *buffer_name*
    (the ordinary scalar-load path: no producer-value substitution)."""
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.Bind) and isinstance(node.value, tvm.tirx.BufferLoad):
            if node.value.buffer.name == buffer_name:
                found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


def _reduce_result_vars(mod):
    """Vars bound to a cross-lane direct-reduce call."""
    result = set()
    for var, value in _binds(mod).items():
        if not isinstance(value, tvm.tirx.Call):
            continue
        if getattr(getattr(value, "op", None), "name", None) in (
            "tl.vmi.vcmax",
            "tl.vmi.vcmin",
            "tl.vmi.vcadd",
        ):
            result.add(var)
    return result


def _writeback_stores(mod):
    """vstore calls whose value is the cross-lane reduce result."""
    result_vars = _reduce_result_vars(mod)
    stores = []
    for call in _vmi_calls(mod).get("tl.vmi.vstore", []):
        if call.args and isinstance(call.args[0], tvm.tirx.Var) and call.args[0] in result_vars:
            stores.append(call)
    return stores


def _single_element_writeback(mod):
    """Assert and return the single-element reduce write-back.

    The target form (design section 7.2) is one VMI store of the 1-element
    reduce result into a 1-element access_ptr: ``vstore(reduced, dst_ptr, 0)``
    with no VL-wide broadcast and no VL-wide lane0 mask. Returns
    ``(store_call, access_ptr_call)``.
    """
    stores = _writeback_stores(mod)
    assert len(stores) == 1, (
        f"expected exactly one reduce write-back store, got {len(stores)}"
    )
    call = stores[0]
    assert len(call.args) == 3, (
        "the write-back must store the single-element reduce result directly "
        f"(value, ptr, offset); got {len(call.args)} arguments"
    )
    ptr = call.args[1]
    assert isinstance(ptr, tvm.tirx.Call) and getattr(ptr.op, "name", None) == "tl.access_ptr", ptr
    extent = int(ptr.args[1])
    assert extent == 1, f"the write-back access_ptr must have extent 1, got {extent}"
    return call, ptr


def _reduce_result_broadcasts(mod):
    """Binds of a ``vbrc`` whose source is a cross-lane reduce result (the
    full-width broadcast the write-back must not use)."""
    result_vars = _reduce_result_vars(mod)
    found = []
    for var, value in _binds(mod).items():
        if not isinstance(value, tvm.tirx.Call):
            continue
        if getattr(getattr(value, "op", None), "name", None) != "tl.vmi.vbrc":
            continue
        if value.args and isinstance(value.args[0], tvm.tirx.Var) and value.args[0] in result_vars:
            found.append(var)
    return found


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k_reduce(E, lanes, rtype="max", dtype="float32"):
    """A converting region: one Parallel unit, then the direct reduce."""
    reduce_op = {"max": T.reduce_max, "min": T.reduce_min, "sum": T.reduce_sum}[rtype]

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            best = T.alloc_shared((1,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                reduce_op(a, best, dim=0, clear=True)
            T.copy(best, B)

    return main


def _k_reduce_consume(E, lanes, dtype="float32"):
    """A consumer unit reads the reduce destination after the producer."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((E,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            best = T.alloc_shared((1,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                T.reduce_max(a, best, dim=0, clear=True)
                for i in T.Parallel(E):
                    a[i] = a[i] * best[0]
            T.copy(a, B)

    return main


def _k_reduce_consume_then_write(E, lanes, dtype="float32"):
    """After the reduce: a scalar write to the destination, then a consumer
    unit reads it back (both as ordinary buffer accesses)."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((E,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            best = T.alloc_shared((1,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                T.reduce_max(a, best, dim=0, clear=True)
                best[0] = a[0]
                for i in T.Parallel(E):
                    a[i] = a[i] + best[0]
            T.copy(a, B)

    return main


def _k_reduce_before_parallel(E, lanes, dtype="float32"):
    """The direct reduce precedes the first T.Parallel unit in the region."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            best = T.alloc_shared((1,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                T.reduce_max(a, best, dim=0, clear=True)
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
            T.copy(best, B)

    return main


def _k_reduce_serial_k(E, lanes, dtype="float32"):
    """The whole unit + reduce sequence repeats under a serial k loop."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            best = T.alloc_shared((1,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for _k in T.serial(2):
                    for i in T.Parallel(E):
                        a[i] = a[i] + a[i]
                    T.reduce_max(a, best, dim=0, clear=True)
            T.copy(best, B)

    return main


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_reduce_max_emits_fold_cross_lane_and_single_element_store():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_reduce(E, lanes, "max"))

    # Legalize leaves the direct reduce untouched (it is not a Select form).
    assert _reduce_calls(stages["legalize"]), "Legalize must keep the direct reduce call"
    assert not _reduce_calls(stages["vectorize"]), "Vectorize must consume tl.tileop.reduce"

    vectorized = stages["vectorize"]
    names = _vmi_calls(vectorized)
    assert names.get("tl.vmi.vmax"), f"the fold must use vmax: {sorted(names)}"
    assert names.get("tl.vmi.vsel"), f"the masked fold must keep prior lanes via vsel: {sorted(names)}"
    assert len(names.get("tl.vmi.vcmax", [])) == 1, f"exactly one cross-lane vcmax expected: {sorted(names)}"
    assert names.get("tl.vmi.vload"), "the source must be read full-width in chunks"
    assert names.get("tl.vmi.create_mask"), "the per-chunk input domain masks must stay"

    _single_element_writeback(vectorized)


@pytest.mark.pto
def test_destination_writeback_is_a_single_element_vstore():
    """The write-back is one 3-argument ``vstore(reduced, dst_ptr, 0)`` of the
    cross-lane result over a 1-element ``access_ptr``: no VL broadcast of the
    result and no VL-wide lane0 mask are emitted for it (design section 7.2),
    while the per-chunk input domain masks stay."""

    for rtype in ("max", "min", "sum"):
        E, lanes = 150, 128
        stages, _ = _lower_capture(_k_reduce(E, lanes, rtype))
        vectorized = stages["vectorize"]

        call, ptr = _single_element_writeback(vectorized)
        assert isinstance(call.args[0], tvm.tirx.Var)
        assert call.args[0] in _reduce_result_vars(vectorized), (
            "the stored value must be the cross-lane reduce result itself"
        )
        assert not _reduce_result_broadcasts(vectorized), (
            f"{rtype}: the write-back must not broadcast the single-element result to VL"
        )
        names = _vmi_calls(vectorized)
        assert names.get("tl.vmi.create_mask"), (
            f"{rtype}: the per-chunk input domain masks must stay"
        )


@pytest.mark.pto
def test_reduce_min_emits_vmin_vcmin_and_single_element_store():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_reduce(E, lanes, "min"))

    vectorized = stages["vectorize"]
    names = _vmi_calls(vectorized)
    assert names.get("tl.vmi.vmin"), f"the fold must use vmin: {sorted(names)}"
    assert len(names.get("tl.vmi.vcmin", [])) == 1, f"exactly one cross-lane vcmin expected: {sorted(names)}"
    assert not names.get("tl.vmi.vmax"), "min must not use vmax"
    assert not names.get("tl.vmi.vcadd"), "min must not use vcadd"

    _single_element_writeback(vectorized)


@pytest.mark.pto
def test_reduce_sum_uses_vcadd_with_reassoc():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_reduce(E, lanes, "sum"))

    names = _vmi_calls(stages["vectorize"])
    assert len(names.get("tl.vmi.vcadd", [])) == 1, f"exactly one cross-lane vcadd expected: {sorted(names)}"
    assert not names.get("tl.vmi.vcmax"), "sum must not use vcmax"
    assert not names.get("tl.vmi.vcmin"), "sum must not use vcmin"

    vcadd = names["tl.vmi.vcadd"][0]
    assert dict(vcadd.annotations).get("reassoc") is not None or "reassoc" in {
        str(key) for key in dict(vcadd.annotations).keys()
    }, f"vcadd must carry an explicit reassoc annotation: {dict(vcadd.annotations)}"


@pytest.mark.pto
def test_destination_read_stays_a_plain_buffer_load():
    """A consumer unit reading ``best[0]`` after the reduce keeps an ordinary
    lane-uniform BufferLoad: no producer-value substitution happens in
    Vectorize, and same-address ordering belongs to the toolchain."""

    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_reduce_consume(E, lanes))

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume tl.tileop.reduce"

    # The consumer's read survives as a plain lane-uniform BufferLoad bound
    # to a scalar var (the existing sload path).
    assert _scalar_load_binds(vectorized, "best"), (
        "reads of the reduce destination must stay plain BufferLoads"
    )

    # The reduce's own single-element write-back is still present.
    _single_element_writeback(vectorized)


@pytest.mark.pto
def test_reduce_inside_serial_k_loop_compiles_repeatedly():
    E, lanes, k_iters = 384, 128, 2
    stages, _ = _lower_capture(_k_reduce_serial_k(E, lanes))

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume the repeated direct reduce"

    names = _vmi_calls(vectorized)
    assert len(names.get("tl.vmi.vcmax", [])) == 1, f"one static vcmax inside the k loop: {sorted(names)}"

    _single_element_writeback(vectorized)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_reduce_inside_parallel_unit_is_rejected():
    E, lanes = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                    T.reduce_max(a, best, dim=0, clear=True)
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a reduce inside a T.Parallel unit must be rejected"
    message = str(error)
    assert "T.Parallel" in message or "parallel" in message, f"expected the in-unit placement diagnostic, got: {message}"


@pytest.mark.pto
def test_reduce_batch_gt_1_is_rejected():
    E, lanes = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                T.reduce_max(a, best, dim=0, clear=True, batch=2)
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "batch > 1 must be rejected in the first version"
    assert "batch" in str(error), f"expected the batch diagnostic, got: {error}"


@pytest.mark.pto
def test_reduce_clear_false_is_rejected():
    E, lanes = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                T.reduce_max(a, best, dim=0, clear=False)
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "clear=False must be rejected in the first version"
    assert "clear" in str(error), f"expected the clear diagnostic, got: {error}"


@pytest.mark.pto
def test_reduce_nan_propagate_is_rejected():
    E, lanes = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                T.reduce_max(a, best, dim=0, clear=True, annotations={"nan_propagate": True})
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "nan_propagate=True must be rejected in the first version"
    assert "nan_propagate" in str(error), f"expected the nan_propagate diagnostic, got: {error}"


@pytest.mark.pto
def test_reduce_2d_source_is_rejected():
    E, lanes = 300, 128

    @T.prim_func
    def main(
        A: T.Tensor((E, 2), "float32"),
        B: T.Tensor((2,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E, 2), "float32")
            best = T.alloc_shared((2,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E * 2):
                    a[i // 2, i % 2] = a[i // 2, i % 2] * 2
                T.reduce_max(a, best, dim=0, clear=True)
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a non-1-D source must be rejected in the first version"
    assert "1-D" in str(error) or "1D" in str(error), f"expected the 1-D source diagnostic, got: {error}"


@pytest.mark.pto
def test_reduce_multi_element_destination_is_rejected():
    E, lanes = 300, 128

    # The front end's own shape validation rejects a 1-D source with a
    # multi-element destination before any IR exists.
    with pytest.raises(ValueError, match="Invalid reduce output shape"):

        @T.prim_func
        def main(
            A: T.Tensor((E,), "float32"),
            B: T.Tensor((2,), "float32"),
        ):
            with T.Kernel(1):
                a = T.alloc_shared((E,), "float32")
                best = T.alloc_shared((2,), "float32")
                T.copy(A, a)
                with T.SimdVF(lanes=lanes):
                    for i in T.Parallel(E):
                        a[i] = a[i] * 2
                    T.reduce_max(a, best, dim=0, clear=True)
                T.copy(best, B)


@pytest.mark.pto
def test_reduce_dynamic_source_extent_is_rejected():
    E, lanes = 300, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                # Hand-built call through the same tl.region bridge the
                # front end uses, with a non-constant source extent.
                n = tirx.Var("n", "int32")
                evaluate(
                    tirx.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.reduce"),
                        region(tirx.BufferLoad(a, [tirx.IntImm("int32", 0)]), "r", n),
                        region(tirx.BufferLoad(best, [tirx.IntImm("int32", 0)]), "w", 1),
                        "max",
                        0,
                        True,
                    )
                )
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a non-static source extent must be rejected in the first version"
    assert "static" in str(error) or "compile-time" in str(error), f"expected the static-extent diagnostic, got: {error}"


@pytest.mark.pto
def test_lone_reduce_without_parallel_is_rejected():
    E, lanes = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                T.reduce_max(a, best, dim=0, clear=True)
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a lone reduce without T.Parallel must be rejected"
    message = str(error)
    assert "T.Parallel" in message, f"expected the placement diagnostic naming T.Parallel, got: {message}"


@pytest.mark.pto
def test_bound_reduce_call_is_rejected_by_verify():
    """A direct reduce bound to a variable (``Bind(handle, Call)``) is not a
    standalone statement. Vectorize only lowers ``Evaluate(Call)``, so Verify
    must reject the bound form with a clear diagnostic instead of letting it
    reach the region walk."""

    E, lanes = 256, 128
    stages, _ = _lower_capture(_k_reduce(E, lanes, "max"))
    mutated = _rewrite_reduce_eval(stages["legalize"], "bind")

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - FFI surfaces a generic error type
        _run_verify(mutated)
    message = str(excinfo.value)
    assert "[VerifyParallelToPTO]" in message, f"expected a Verify diagnostic, got: {message}"
    assert "Evaluate" in message and "standalone" in message, (
        f"the diagnostic must require a standalone Evaluate(Call): {message}"
    )


@pytest.mark.pto
def test_nested_reduce_call_is_rejected_by_verify():
    """A direct reduce embedded in another expression
    (``Evaluate(call_extern(..., Call))``) is rejected as well: the rule is
    "the reduce call is the direct value of its Evaluate", not merely "some
    Evaluate contains a reduce call"."""

    E, lanes = 256, 128
    stages, _ = _lower_capture(_k_reduce(E, lanes, "max"))
    mutated = _rewrite_reduce_eval(stages["legalize"], "nested")

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - FFI surfaces a generic error type
        _run_verify(mutated)
    message = str(excinfo.value)
    assert "[VerifyParallelToPTO]" in message, f"expected a Verify diagnostic, got: {message}"
    assert "standalone" in message or "embedded" in message, (
        f"the diagnostic must reject embedded reduce calls: {message}"
    )


@pytest.mark.pto
def test_write_to_destination_after_reduce_is_not_rejected():
    """A later write to the reduce destination is not rejected by any
    TileLang-side producer rule: the destination is an ordinary buffer and
    same-address ordering is the toolchain's responsibility."""

    E, lanes = 256, 128
    stages, _ = _lower_capture(_k_reduce_consume_then_write(E, lanes))

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume tl.tileop.reduce"

    # The scalar write and the subsequent plain load both survive.
    stores = _collect(
        vectorized,
        lambda n: isinstance(n, tvm.tirx.BufferStore) and n.buffer.name == "best",
    )
    assert stores, "the scalar write to the destination must survive"

    assert _scalar_load_binds(vectorized, "best"), (
        "the later read of the destination must stay a plain BufferLoad"
    )


@pytest.mark.pto
def test_conditional_write_to_destination_then_read_is_not_rejected():
    """A conditional (IfThenElse) write to the destination between the reduce
    and a later read is not rejected: no producer record exists to bypass."""

    E, lanes = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            best = T.alloc_shared((1,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    a[i] = a[i] * 2
                T.reduce_max(a, best, dim=0, clear=True)
                if a[0] > 0:
                    best[0] = a[0]
                for i in T.Parallel(E):
                    a[i] = a[i] + best[0]
            T.copy(a, B)

    stages, _ = _lower_capture(main)

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume tl.tileop.reduce"

    # The conditional scalar store and the consumer's plain load both stay.
    assert _collect(
        vectorized,
        lambda n: isinstance(n, tvm.tirx.BufferStore) and n.buffer.name == "best",
    ), "the conditional write to the destination must survive"
    assert _scalar_load_binds(vectorized, "best"), (
        "the consumer read after the conditional write must stay a plain BufferLoad"
    )


@pytest.mark.pto
def test_reduce_before_first_parallel_lowers_with_region_lanes():
    """The direct reduce may appear before the first T.Parallel unit: the
    lowering takes its lane count from the SIMD_VF annotation, never from
    per-unit conversion state."""

    E, lanes = 256, 128
    stages, _ = _lower_capture(_k_reduce_before_parallel(E, lanes))

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume tl.tileop.reduce"

    names = _vmi_calls(vectorized)
    assert len(names.get("tl.vmi.vcmax", [])) == 1, f"exactly one cross-lane vcmax expected: {sorted(names)}"
    _single_element_writeback(vectorized)


# ---------------------------------------------------------------------------
# Prefix source region: logical extent E <= backing capacity
# (issue-5B-REDUCE-PREFIX-REGION, user-approved minimal fix)
# ---------------------------------------------------------------------------


def _k_reduce_prefix(E, cap, lanes, rtype="max", dtype="float32"):
    """Source backing capacity *cap* written in full by the unit; the reduce
    consumes the prefix region ``src[0:E]``. The padding lanes hold values
    that would corrupt the answer if the logical-extent tail mask were
    missing (the device numeric uses answer-corrupting padding)."""
    reduce_op = {"max": T.reduce_max, "min": T.reduce_min, "sum": T.reduce_sum}[rtype]

    @T.prim_func
    def main(A: T.Tensor((cap,), dtype), B: T.Tensor((2,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((cap,), dtype)
            dst = T.alloc_shared((2,), dtype)
            T.copy(A, src)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(cap):
                    src[i] = src[i] * 2
                dst[1] = 7.0
                reduce_op(src[0:E], dst[0:1], dim=0, clear=True)
            T.copy(dst, B)

    return main


def _create_mask_constants(mod):
    """The IntImm operands of every tl.vmi.create_mask call."""
    consts = []
    for call in _vmi_calls(mod).get("tl.vmi.create_mask", []):
        if call.args and isinstance(call.args[0], tvm.tirx.IntImm):
            consts.append(int(call.args[0]))
    return consts


@pytest.mark.pto
def test_prefix_source_region_uses_the_logical_extent_for_masks():
    """E=150 < capacity 256, L=128: the region extent is the logical E, so
    the tail chunk mask is create_mask(22) — proof that the mask follows the
    prefix extent, not the backing capacity (a whole-P reduce would use only
    create_mask(128) chunks)."""
    E, cap, lanes = 150, 256, 128
    stages, _ = _lower_capture(_k_reduce_prefix(E, cap, lanes, "max"))

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume tl.tileop.reduce"
    names = _vmi_calls(vectorized)
    assert len(names.get("tl.vmi.vcmax", [])) == 1, f"one cross-lane vcmax expected: {sorted(names)}"
    assert names.get("tl.vmi.vmax"), f"the fold must use vmax: {sorted(names)}"
    assert names.get("tl.vmi.vsel"), f"the tail chunk needs the keep-prior vsel: {sorted(names)}"
    _single_element_writeback(vectorized)

    consts = _create_mask_constants(vectorized)
    assert 22 in consts, f"the tail chunk mask must cover the logical extent (150-128=22), got {consts}"


@pytest.mark.pto
def test_prefix_source_region_below_one_lane_uses_partial_mask():
    """E=70 < L=128: a single chunk whose domain mask is create_mask(70)."""
    E, cap, lanes = 70, 128, 128
    stages, _ = _lower_capture(_k_reduce_prefix(E, cap, lanes, "max"))

    vectorized = stages["vectorize"]
    names = _vmi_calls(vectorized)
    assert len(names.get("tl.vmi.vcmax", [])) == 1, f"one cross-lane vcmax expected: {sorted(names)}"
    _single_element_writeback(vectorized)
    consts = _create_mask_constants(vectorized)
    assert 70 in consts, f"the domain mask must cover the logical extent, got {consts}"


@pytest.mark.pto
def test_prefix_extent_beyond_capacity_is_rejected():
    """The prefix must stay inside the backing buffer: extent > capacity is
    rejected by Verify (rejection is on the reduce contract, so it holds for
    every later pass as well)."""
    E, cap, lanes = 300, 256, 128

    @T.prim_func
    def main(A: T.Tensor((cap,), "float32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((cap,), "float32")
            dst = T.alloc_shared((2,), "float32")
            T.copy(A, src)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(cap):
                    src[i] = src[i] * 2
                T.reduce_max(src[0:E], dst[0:1], dim=0, clear=True)
            T.copy(dst, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a prefix beyond the backing capacity must be rejected"
    message = str(error)
    assert "capacity" in message or "backing" in message, (
        f"expected the capacity diagnostic, got: {message}"
    )


@pytest.mark.pto
def test_prefix_region_with_nonzero_min_is_rejected():
    """A region that does not start at 0 is not a prefix and stays rejected."""
    E, cap, lanes = 256, 256, 128

    @T.prim_func
    def main(A: T.Tensor((cap,), "float32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((cap,), "float32")
            dst = T.alloc_shared((2,), "float32")
            T.copy(A, src)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(cap):
                    src[i] = src[i] * 2
                evaluate(
                    tirx.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.reduce"),
                        region(tirx.BufferLoad(src, [tirx.IntImm("int32", 1)]), "r", E - 1),
                        region(tirx.BufferLoad(dst, [tirx.IntImm("int32", 0)]), "w", 1),
                        "max",
                        0,
                        True,
                    )
                )
            T.copy(dst, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a non-prefix region must be rejected"
    assert "starting at 0" in str(error), f"expected the prefix diagnostic, got: {error}"


def _k_reduce_strided_source(stride, extent=256, backing_capacity=512, lanes=128):
    """Build a direct Reduce over an explicit strided UB view.

    The backing allocation is deliberately large enough for the physical
    footprint of the view.  This isolates the direct-Reduce source-layout
    contract from addressability: the first version accepts only a compact
    (unit-stride) UB source and must reject a view that would need
    stride-aware/gather lowering.
    """

    @T.prim_func
    def main(A: T.Tensor((backing_capacity,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            backing = T.alloc_shared((backing_capacity,), "float32")
            best = T.alloc_shared((1,), "float32")
            src = T.StridedTensor(
                (extent,), (stride,), "float32", data=backing.data, scope="shared.dyn"
            )
            T.copy(A, backing)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(backing_capacity):
                    backing[i] = backing[i] * 2
                T.reduce_max(src[0:extent], best, dim=0, clear=True)
            T.copy(best, B)

    return main


@pytest.mark.pto
def test_reduce_explicit_unit_stride_source_is_accepted():
    """An explicit stride proven equal to one is still a compact source."""

    stages, error = _lower_capture(_k_reduce_strided_source(1), expect_error=True)
    assert error is None, f"unit-stride source must be accepted, got: {error}"
    assert not _reduce_calls(stages["vectorize"]), "Vectorize must consume tl.tileop.reduce"


@pytest.mark.pto
def test_reduce_non_unit_stride_source_is_rejected():
    """Direct Reduce does not silently reinterpret a strided UB view."""

    _, error = _lower_capture(_k_reduce_strided_source(2), expect_error=True)
    assert error is not None, "non-unit-stride source must be rejected"
    assert "stride" in str(error).lower(), f"expected the stride diagnostic, got: {error}"


@pytest.mark.pto
def test_reduce_provably_unit_stride_expression_is_accepted():
    """A non-literal stride expression proven equal to one stays accepted.

    The stride is a ``Cast`` node rather than an ``IntImm``: a literal-only
    check would reject it, so this pins the analyzer-based acceptance
    (``CanProveEqual``) instead of an equality against the integer constant.
    """

    stride = tirx.Cast("int32", tirx.const(1, "int64"))
    stages, error = _lower_capture(_k_reduce_strided_source(stride), expect_error=True)
    assert error is None, f"provably-unit stride must be accepted, got: {error}"
    # Keep the probe honest: if the frontend ever folds the expression to a
    # literal, swap in another provably-unit expression so this still
    # exercises the analyzer path and not the IntImm fast path.
    assert "strides=(T.Cast" in stages["legalize"].script(), (
        "the expression stride was folded to a literal before reaching the "
        "pipeline; replace the expression to keep the test discriminating"
    )
    assert not _reduce_calls(stages["vectorize"]), "Vectorize must consume tl.tileop.reduce"


@pytest.mark.pto
def test_reduce_unprovable_dynamic_stride_source_is_rejected():
    """An unknown (unprovable) stride must not be treated as unit.

    ``n`` is a bound kernel parameter, so the rejection must come from the
    direct-reduce stride contract and not from an unbound-variable frontend
    error masking the parser diagnostic.
    """

    @T.prim_func
    def main(
        A: T.Tensor((512,), "float32"),
        B: T.Tensor((1,), "float32"),
        n: T.int32,
    ):
        with T.Kernel(1):
            backing = T.alloc_shared((512,), "float32")
            best = T.alloc_shared((1,), "float32")
            src = T.StridedTensor(
                (256,), (n,), "float32", data=backing.data, scope="shared.dyn"
            )
            T.copy(A, backing)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(512):
                    backing[i] = backing[i] * 2
                T.reduce_max(src[0:256], best, dim=0, clear=True)
            T.copy(best, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a dynamic unprovable stride must be rejected"
    assert "contiguous unit stride" in str(error), (
        f"expected the stride diagnostic, got: {error}"
    )
