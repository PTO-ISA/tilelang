"""Explicit fused FP32 arithmetic must survive Parallel vectorization."""
import pytest
import torch
import tilelang
from tilelang.ascend import language as T
from testing.ascend.language.pto.test_pto_parallel_vectorize_structure import _lower_capture, _vmi_calls


def make_kernel(extent=192, uniform=False):
    @T.prim_func
    def main(A: T.Tensor((extent,), 'float32'), B: T.Tensor((extent,), 'float32'),
             C: T.Tensor((extent,), 'float32'), D: T.Tensor((extent,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((extent,), 'float32')
            b = T.alloc_shared((extent,), 'float32')
            c = T.alloc_shared((extent,), 'float32')
            T.copy(A, a)
            T.copy(B, b)
            T.copy(C, c)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    c[i] = T.call_intrin('float32', 'tirx.fma', a[i], T.float32(2.0) if uniform else b[i], c[i])
            T.copy(c, D)
    return main


@pytest.mark.parametrize('extent', [64, 192, 256])
@pytest.mark.parametrize('uniform', [False, True])
def test_parallel_fma_structure(extent, uniform):
    mod, error = _lower_capture(make_kernel(extent, uniform))
    assert error is None, str(error)
    operations = _vmi_calls(mod)
    assert 'tl.vmi.vmula' in operations
    assert 'tl.vmi.vmul' not in operations
    assert 'tl.vmi.vadd' not in operations
    assert not operations.get('tirx.fma')


@pytest.mark.pto
@pytest.mark.parametrize('extent', [64, 192, 256])
def test_parallel_fma_rounds_once(extent):
    kernel = tilelang.compile(make_kernel(extent), target='pto')
    a = torch.full((extent,), 1 + 2 ** -12, device='npu')
    b = torch.full((extent,), 1 - 2 ** -12, device='npu')
    c = torch.full((extent,), -1.0, device='npu')
    out = torch.empty_like(c)
    kernel(a, b, c, out)
    torch.testing.assert_close(out, torch.full_like(out, -(2 ** -24)), rtol=0, atol=0)


def test_parallel_fma_rejects_non_fp32():
    @T.prim_func
    def main(A: T.Tensor((128,), 'float16'), B: T.Tensor((128,), 'float16')):
        with T.Kernel(1):
            a = T.alloc_shared((128,), 'float16')
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    a[i] = T.call_intrin('float16', 'tirx.fma', a[i], a[i], a[i])
            T.copy(a, B)
    with pytest.raises(Exception, match='three float32 operands'):
        _lower_capture(main)
