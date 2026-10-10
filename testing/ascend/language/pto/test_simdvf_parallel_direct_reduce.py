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
  ``vcadd(reassoc=True)``, and a ``create_mask(1, size=VL)`` predicated store
  of the result into ``destination[0]`` (never gated by the lane0 config);
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


def _is_create_mask_of(mask_expr, lanes, count, binds):
    """Return whether *mask_expr* is a Var bound to create_mask(count, size=lanes)."""
    if not isinstance(mask_expr, tvm.tirx.Var):
        return False
    value = binds.get(mask_expr)
    if value is None or not isinstance(value, tvm.tirx.Call):
        return False
    if getattr(getattr(value, "op", None), "name", None) != "tl.vmi.create_mask":
        return False
    if not value.args or not isinstance(value.args[0], tvm.tirx.IntImm):
        return False
    return int(value.args[0]) == count


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
def test_reduce_max_emits_fold_cross_lane_and_lane0_predicated_store():
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

    binds = _binds(vectorized)
    stores = names.get("tl.vmi.vstore", [])
    lane0_stores = [
        call
        for call in stores
        if len(call.args) == 4 and _is_create_mask_of(call.args[3], lanes, 1, binds)
    ]
    assert lane0_stores, "the reduce result must be written with a create_mask(1) predicated store"


@pytest.mark.pto
def test_reduce_min_emits_vmin_vcmin_and_lane0_store():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_reduce(E, lanes, "min"))

    names = _vmi_calls(stages["vectorize"])
    assert names.get("tl.vmi.vmin"), f"the fold must use vmin: {sorted(names)}"
    assert len(names.get("tl.vmi.vcmin", [])) == 1, f"exactly one cross-lane vcmin expected: {sorted(names)}"
    assert not names.get("tl.vmi.vmax"), "min must not use vmax"
    assert not names.get("tl.vmi.vcadd"), "min must not use vcadd"

    binds = _binds(stages["vectorize"])
    stores = names.get("tl.vmi.vstore", [])
    assert any(
        len(call.args) == 4 and _is_create_mask_of(call.args[3], lanes, 1, binds) for call in stores
    ), "the reduce result must be written with a create_mask(1) predicated store"


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

    # The reduce's own lane0 write-back is still present and predicated.
    names = _vmi_calls(vectorized)
    binds = _binds(vectorized)
    stores = names.get("tl.vmi.vstore", [])
    assert any(
        len(call.args) == 4 and _is_create_mask_of(call.args[3], lanes, 1, binds) for call in stores
    ), "the reduce's own destination write must stay lane0-predicated"


@pytest.mark.pto
def test_reduce_inside_serial_k_loop_compiles_repeatedly():
    E, lanes, k_iters = 384, 128, 2
    stages, _ = _lower_capture(_k_reduce_serial_k(E, lanes))

    vectorized = stages["vectorize"]
    assert not _reduce_calls(vectorized), "Vectorize must consume the repeated direct reduce"

    names = _vmi_calls(vectorized)
    assert len(names.get("tl.vmi.vcmax", [])) == 1, f"one static vcmax inside the k loop: {sorted(names)}"

    binds = _binds(vectorized)
    stores = names.get("tl.vmi.vstore", [])
    assert any(
        len(call.args) == 4 and _is_create_mask_of(call.args[3], lanes, 1, binds) for call in stores
    ), "the repeated reduce must still write lane0 predicated"


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
    binds = _binds(vectorized)
    assert any(
        len(call.args) == 4 and _is_create_mask_of(call.args[3], lanes, 1, binds)
        for call in names.get("tl.vmi.vstore", [])
    ), "the reduce before the first unit must still write lane0 predicated"
