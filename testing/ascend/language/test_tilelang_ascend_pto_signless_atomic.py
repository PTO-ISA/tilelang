"""Integer pto.atomic_add must be signless (no signedness=)."""

import pytest

import tilelang
from tilelang.ascend import language as T


@pytest.mark.pto
def test_integer_atomic_add_is_signless():
    @T.prim_func
    def clamp_atomic(counter: T.Buffer((4,), "int32"), acc: T.Buffer((4,), "int32")):
        with T.Kernel(1):
            acc_ub = T.alloc_shared((4,), "int32")
            T.copy(acc, acc_ub)
            with T.SimtVF(threads=1):
                for i in T.serial(4):
                    T.atomic_add(counter[i], acc_ub[i])

    source = tilelang.lower(clamp_atomic, target="pto").kernel_source
    assert "pto.atomic_add(" in source
    assert "signedness=" not in source
    assert "pto.castptr(" in source
    assert "pto.i32" in source
    assert "pto.si32" not in source.split("pto.atomic_add", 1)[1][:200]


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
    assert "pto.atomic_add(" in source
    assert "signedness=" not in source
    assert "pto.i64" in source
    assert "pto.si64" not in source.split("pto.atomic_add", 1)[1][:200]
