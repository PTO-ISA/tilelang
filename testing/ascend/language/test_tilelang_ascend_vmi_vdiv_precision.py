"""Frontend coverage for T.vmi.vdiv(precision='exact')."""

from types import SimpleNamespace

import pytest

from tilelang import lower
from tilelang.ascend import language as T


def _vdiv_kernel(precision=None, dtype="float32"):
    @T.prim_func
    def func(A: T.Buffer((64,), dtype), B: T.Buffer((64,), dtype)):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), dtype)
            b_ub = T.alloc_shared((64,), dtype)
            T.copy(A, a_ub)
            with T.SimdVF():
                mask = T.vmi.create_mask(64, size=64)
                x = T.vmi.vload(a_ub[0], size=64)
                y = T.vmi.vbrc(T.float32(1), size=64)
                kwargs = {} if precision is None else {"precision": precision}
                quot = T.vmi.vdiv(x, y, mask, **kwargs)
                T.vmi.vstore(quot, b_ub[0], mask)
            T.copy(b_ub, B)

    return func


@pytest.mark.parametrize("precision", [None, "ftz_true"])
@pytest.mark.pto
def test_pto_vdiv_hw_has_no_residual_chain(precision):
    source = lower(_vdiv_kernel(precision), target="pto").kernel_source
    assert "pto.vmi.vdiv(" in source
    assert "pto.vmi.vmula(" not in source
    assert "precision=" not in source


@pytest.mark.parametrize("precision", ["exact", "vdiv_0ulp_ftz_true"])
@pytest.mark.pto
def test_pto_vdiv_exact_expands_residual_search(precision):
    source = lower(_vdiv_kernel(precision), target="pto").kernel_source
    assert source.count("pto.vmi.vdiv(") == 1
    assert source.count("pto.vmi.vmula(") == 3
    assert "pto.vmi.vsel(" in source
    assert "precision=" not in source
    compile(source, "<pto-vdiv-exact>", "exec")


@pytest.mark.pto
def test_pto_vdiv_precision_rejects_invalid(monkeypatch):
    from tilelang.ascend.language import vmi as V

    monkeypatch.setattr(V, "require_vmi_scope", lambda *args, **kwargs: None)
    value = SimpleNamespace(dtype="int32x64")
    mask = SimpleNamespace(dtype="boolx64")
    with pytest.raises(ValueError, match="precision"):
        V.vdiv(value, value, mask, precision="not-a-mode")
    with pytest.raises(ValueError, match="requires float32"):
        V.vdiv(value, value, mask, precision="exact")
