"""Integer PTO atomics: signless pointers; signedness= only on max/min."""

import pytest

import tilelang
from tilelang.ascend import language as T


def _atomic_kernel(op, dtype):
    @T.prim_func
    def func(counter: T.Buffer((4,), dtype), acc: T.Buffer((4,), dtype)):
        with T.Kernel(1):
            acc_ub = T.alloc_shared((4,), dtype)
            T.copy(acc, acc_ub)
            with T.SimtVF(threads=1):
                for i in T.serial(4):
                    op(counter[i], acc_ub[i])

    return func


def _atomic_call(source, name):
    idx = source.find(f"pto.{name}(")
    assert idx >= 0, source
    return source[idx : idx + 240]


@pytest.mark.pto
def test_integer_atomic_add_is_signless():
    source = tilelang.lower(_atomic_kernel(T.atomic_add, "int32"), target="pto").kernel_source
    call = _atomic_call(source, "atomic_add")
    assert "signedness=" not in source
    assert "pto.castptr(" in call
    assert "pto.i32" in call
    assert "pto.si32" not in call


@pytest.mark.pto
def test_integer_atomic_add_i64_is_signless():
    @T.prim_func
    def clamp_atomic_i64(counter: T.Buffer((2,), "int64"), acc: T.Buffer((2,), "int64")):
        with T.Kernel(1):
            acc_ub = T.alloc_shared((2,), "int64")
            T.copy(acc, acc_ub)
            with T.SimtVF(threads=1):
                for i in T.serial(2):
                    T.atomic_add(counter[i], acc_ub[i])

    source = tilelang.lower(clamp_atomic_i64, target="pto").kernel_source
    call = _atomic_call(source, "atomic_add")
    assert "signedness=" not in source
    assert "pto.i64" in call
    assert "pto.si64" not in call


@pytest.mark.pto
@pytest.mark.parametrize("op,name", [(T.atomic_max, "atomic_max"), (T.atomic_min, "atomic_min")])
def test_integer_atomic_extremum_signed_i32(op, name):
    source = tilelang.lower(_atomic_kernel(op, "int32"), target="pto").kernel_source
    call = _atomic_call(source, name)
    assert 'signedness="signed"' in call
    assert 'signedness="unsigned"' not in call
    assert "pto.castptr(" in call
    assert "pto.i32" in call
    assert "pto.si32" not in call
    assert "pto.ui32" not in call


@pytest.mark.pto
@pytest.mark.parametrize("op,name", [(T.atomic_max, "atomic_max"), (T.atomic_min, "atomic_min")])
def test_integer_atomic_extremum_unsigned_i32(op, name):
    source = tilelang.lower(_atomic_kernel(op, "uint32"), target="pto").kernel_source
    call = _atomic_call(source, name)
    assert 'signedness="unsigned"' in call
    assert "pto.castptr(" in call
    assert "pto.i32" in call
    assert "pto.si32" not in call
    assert "pto.ui32" not in call


@pytest.mark.pto
def test_integer_atomic_max_signed_i64():
    @T.prim_func
    def clamp_atomic_i64(counter: T.Buffer((2,), "int64"), acc: T.Buffer((2,), "int64")):
        with T.Kernel(1):
            acc_ub = T.alloc_shared((2,), "int64")
            T.copy(acc, acc_ub)
            with T.SimtVF(threads=1):
                for i in T.serial(2):
                    T.atomic_max(counter[i], acc_ub[i])

    source = tilelang.lower(clamp_atomic_i64, target="pto").kernel_source
    call = _atomic_call(source, "atomic_max")
    assert 'signedness="signed"' in call
    assert "pto.i64" in call
    assert "pto.si64" not in call


@pytest.mark.pto
def test_float_atomic_max_omits_signedness():
    source = tilelang.lower(_atomic_kernel(T.atomic_max, "float32"), target="pto").kernel_source
    call = _atomic_call(source, "atomic_max")
    assert "signedness=" not in call
