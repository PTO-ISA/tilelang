"""Full backend compilation and NPU numerics for ordinary private locals."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from testing.ascend.language.pto.test_pto_parallel_private_alloc import external_kernel, private_kernel, rejected_kernel

pytestmark = pytest.mark.pto


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_external_scalar_array_compiles(dtype, mode):
    tilelang.compile(external_kernel(dtype, mode), target="pto", out_idx=-1)


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_private_array_vector_copy_compiles(dtype):
    tilelang.compile(rejected_kernel("local", dtype), target="pto", out_idx=-1)


@pytest.mark.parametrize(
    "n,lanes,rows,column,dtype",
    [
        (1, 64, 1, False, "int32"),
        (63, 64, 1, False, "int32"),
        (64, 64, 1, False, "int32"),
        (65, 64, 1, False, "int32"),
        (131, 64, 3, False, "int32"),
        (131, 64, 3, True, "int32"),
        (259, 128, 1, False, "int32"),
        (515, 256, 1, False, "int32"),
        (131, 64, 1, False, "float32"),
    ],
)
def test_private_state_snapshot_and_tail(n, lanes, rows, column, dtype):
    # This suite is a device gate: an unavailable runtime is a failure, not a skip.
    assert hasattr(torch, "npu"), "NPU runtime is required for this gate"
    p = (n + lanes - 1) // lanes * lanes
    tdtype = getattr(torch, dtype)
    kernel = tilelang.compile(private_kernel(n, lanes, rows, column, dtype), target="pto", out_idx=-1)
    for shift in [0, 11]:
        cpu = ((torch.arange(rows * p, dtype=torch.int32) % 37) - 18 + shift).to(tdtype)
        seed = torch.full((rows * p,), -12345, dtype=tdtype)
        actual = kernel(cpu.npu(), seed.npu()).cpu().reshape(rows, p)
        expected = seed.reshape(rows, p).clone()
        expected[:, :n] = 3 * cpu.reshape(rows, p)[:, :n] + 7
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_external_scalar_rounding(dtype):
    assert hasattr(torch, "npu")
    tdtype = getattr(torch, dtype)
    kernel = tilelang.compile(external_kernel(dtype), target="pto", out_idx=-1)
    a = torch.arange(192, dtype=torch.float32).to(tdtype).npu()
    actual = kernel(a).cpu()[:131]
    expected = torch.full((131,), 3.1, dtype=tdtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def sequential_kernel():
    @T.prim_func
    def main(A: T.Tensor((192,), "float32"), Seed: T.Tensor((192,), "float32"), Out: T.Tensor((192,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((192,), "float32")
            x = T.alloc_shared((192,), "float32")
            b = T.alloc_shared((192,), "float32")
            T.copy(A, a)
            T.copy(Seed, b)
            with T.SimdVF(lanes=64):
                scale = T.alloc_var("float32", init=2.0)
                bias = T.alloc_local((1,), "float32")
                bias[0] = 3.0
                for i in T.Parallel(131):
                    acc = T.alloc_var("float32", init=7.0)
                    acc = acc + a[i]
                    x[i] = acc * scale + bias[0]
                scale = 5.0
                bias[0] = -2.0
                for j in T.Parallel(131):
                    tmp = T.alloc_local((1,), "float32")
                    tmp[0] = x[j] * scale + bias[0]
                    b[j] = tmp[0]
            T.copy(b, Out)

    return main


def test_sequential_units_observe_external_updates():
    assert hasattr(torch, "npu")
    kernel = tilelang.compile(sequential_kernel(), target="pto", out_idx=-1)
    for shift in [0, 9]:
        a = (torch.arange(192, dtype=torch.float32) % 31) - 15 + shift
        seed = torch.full((192,), -12345.0)
        expected = seed.clone()
        expected[:131] = 5 * (2 * (a[:131] + 7) + 3) - 2
        actual = kernel(a.npu(), seed.npu()).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_private_array_cast_roundtrip(dtype):
    @T.prim_func
    def main(A: T.Tensor((192,), "float32"), Seed: T.Tensor((192,), "float32"), Out: T.Tensor((192,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((192,), "float32")
            b = T.alloc_shared((192,), "float32")
            T.copy(A, a)
            T.copy(Seed, b)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(131):
                    tmp = T.alloc_local((1,), dtype)
                    tmp[0] = T.Cast(dtype, a[i])
                    b[i] = T.Cast("float32", tmp[0])
            T.copy(b, Out)

    assert hasattr(torch, "npu")
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    source = ((torch.arange(192, dtype=torch.float32) % 37) - 18) / 8
    seed = torch.full((192,), -12345.0)
    expected = seed.clone()
    expected[:131] = source[:131].to(getattr(torch, dtype)).float()
    actual = kernel(source.npu(), seed.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
