"""PTO pointer normalization for Ascend MTE copies."""

import pytest

import tilelang
from tilelang.ascend import language as T


def _mte_call(source: str, op_name: str) -> str:
    start = source.index(op_name)
    return source[start:].splitlines()[0]


@pytest.mark.pto
def test_ub_to_gm_inserts_castptr_for_aliased_dtype_mismatch():
    @T.prim_func
    def kernel(out: T.Buffer((16,), "uint8")):
        with T.Kernel(1):
            ub = T.alloc_shared((8,), "uint16")
            out_u16 = T.Tensor((8,), "uint16", out.data)
            T.copy(ub, out_u16)

    source = tilelang.lower(kernel, target="pto").kernel_source
    call = _mte_call(source, "pto.mte_ub_gm")

    assert 'out_u16: pto.ptr(pto.ui8, "gm")' in source, source
    assert call.startswith("pto.mte_ub_gm(pto.castptr("), call
    assert 'pto.ptr(pto.ui8, "ub")' in call, call
    assert 'pto.ui16, "gm"' not in call, call


@pytest.mark.pto
def test_unpadded_gm_to_ub_inserts_castptr_for_dtype_mismatch():
    @T.prim_func
    def kernel(src: T.Buffer((16,), "uint8")):
        with T.Kernel(1):
            ub = T.alloc_shared((8,), "uint16")
            src_u16 = T.Tensor((8,), "uint16", src.data)
            T.copy(src_u16, ub)

    source = tilelang.lower(kernel, target="pto").kernel_source
    call = _mte_call(source, "pto.mte_gm_ub")

    assert 'src_u16: pto.ptr(pto.ui8, "gm")' in source, source
    assert 'pto.ptr(pto.ui8, "ub")' in call, call
    assert 'pto.ui16, "gm"' not in call, call


@pytest.mark.pto
def test_padded_gm_to_ub_keeps_element_width_without_redundant_castptr():
    @T.prim_func
    def kernel(src: T.Buffer((4, 30), "float32"), dst: T.Buffer((4, 32), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 32), "float32")
            T.copy(src, ub[:, :30], pad_value=-1.0)
            T.copy(ub, dst)

    source = tilelang.lower(kernel, target="pto").kernel_source
    call = _mte_call(source, "pto.mte_gm_ub")

    assert 'dst: pto.ptr(pto.f32, "gm")' in source, source
    assert 'src: pto.ptr(pto.f32, "gm")' in source, source
    assert 'pto.ptr(pto.f32, "ub")' in source, source
    assert "pto.castptr" not in call, call
    assert "pad=(tl_copy_pad_0, 0, 2)" in call, call
