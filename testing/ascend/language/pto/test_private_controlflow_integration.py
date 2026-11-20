"""Integration probes for PR38 with PR11; mandatory device tests."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from testing.ascend.language.pto.test_simdvf_parallel_control_flow import _lower_capture, _vmi_call_names

pytestmark = pytest.mark.pto


def kernel(kind, mode, n=131):
    p = (n + 63) // 64 * 64

    @T.prim_func
    def main(A: T.Tensor((p,), "float32"), Seed: T.Tensor((p,), "float32"), Out: T.Tensor((p,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((p,), "float32")
            b = T.alloc_shared((p,), "float32")
            T.copy(A, a)
            T.copy(Seed, b)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(n):
                    if kind == "var":
                        acc = T.alloc_var("float32")
                        acc = a[i]
                        old = acc
                        if mode == "two":
                            if acc > 0:
                                acc = acc + 7.0
                            else:
                                acc = acc - 3.0
                        else:
                            if acc > 0:
                                acc = acc + 7.0
                        b[i] = old * 2.0 + acc
                    else:
                        tmp = T.alloc_local((1,), "float32")
                        tmp[0] = a[i]
                        old = tmp[0]
                        if mode == "two":
                            if tmp[0] > 0:
                                tmp[0] = tmp[0] + 7.0
                            else:
                                tmp[0] = tmp[0] - 3.0
                        else:
                            if tmp[0] > 0:
                                tmp[0] = tmp[0] + 7.0
                        b[i] = old * 2.0 + tmp[0]
            T.copy(b, Out)

    return main


@pytest.mark.parametrize("kind", ["var", "local"])
@pytest.mark.parametrize("mode", ["one", "two"])
def test_lower(kind, mode):
    stages, error = _lower_capture(kernel(kind, mode), expect_error=True)
    assert error is None, str(error)
    assert "tl.vmi.vsel" in _vmi_call_names(stages["vectorize"])


@pytest.mark.parametrize("kind", ["var", "local"])
@pytest.mark.parametrize("mode", ["one", "two"])
@pytest.mark.parametrize("n", [64, 131])
def test_device(kind, mode, n):
    p = (n + 63) // 64 * 64
    f = tilelang.compile(kernel(kind, mode, n), target="pto", out_idx=-1)
    x = (torch.arange(p, dtype=torch.float32) % 37) - 18
    seed = torch.full((p,), -12345.0)
    expected = seed.clone()
    expected[:n] = 3 * x[:n] + torch.where(x[:n] > 0, 7.0, -3.0 if mode == "two" else 0.0)
    torch.testing.assert_close(f(x.npu(), seed.npu()).cpu(), expected, rtol=0, atol=0)


def lifecycle_kernel(case, n=131):
    padded = (n + 63) // 64 * 64

    @T.prim_func
    def main(A: T.Tensor((padded,), "float32"), Seed: T.Tensor((padded,), "float32"), Out: T.Tensor((padded,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((padded,), "float32")
            b = T.alloc_shared((padded,), "float32")
            T.copy(A, a)
            T.copy(Seed, b)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(n):
                    if case == "serial_default":
                        for _ in T.serial(2):
                            acc = T.alloc_var("float32")
                            acc = acc + a[i]
                            b[i] = acc
                    elif case == "serial_explicit":
                        for _ in T.serial(2):
                            acc = T.alloc_var("float32", init=0.0)
                            acc = acc + a[i]
                            b[i] = acc
                    elif case == "serial_outer":
                        acc = T.alloc_var("float32")
                        for _ in T.serial(2):
                            acc = acc + a[i]
                        b[i] = acc
                    elif case == "serial_local":
                        for _ in T.serial(2):
                            tmp = T.alloc_local((1,), "float32")
                            tmp[0] = a[i]
                            b[i] = tmp[0]
                    elif case == "branch_var":
                        if a[i] > 0:
                            acc = T.alloc_var("float32")
                            acc = acc + a[i]
                            b[i] = acc
                    elif case == "branch_local":
                        if a[i] > 0:
                            tmp = T.alloc_local((1,), "float32")
                            tmp[0] = a[i]
                            b[i] = tmp[0]
                    elif case == "complementary":
                        tmp = T.alloc_local((1,), "float32")
                        if a[i] > 0:
                            tmp[0] = a[i] * 2.0
                        if a[i] <= 0:
                            tmp[0] = a[i] - 3.0
                        b[i] = tmp[0]
            T.copy(b, Out)

    return main


@pytest.mark.parametrize(
    "case", ["serial_default", "serial_explicit", "serial_outer", "serial_local", "branch_var", "branch_local", "complementary"]
)
def test_device_declaration_lifetime(case):
    n = 131
    f = tilelang.compile(lifecycle_kernel(case, n), target="pto", out_idx=-1)
    x = torch.arange(192, dtype=torch.float32) % 37 - 18
    seed = -1000.0 - torch.arange(192, dtype=torch.float32)
    expected = seed.clone()
    a = x[:n]
    if case == "serial_outer":
        expected[:n] = 2 * a
    elif case.startswith("branch"):
        expected[:n] = torch.where(a > 0, a, seed[:n])
    elif case == "complementary":
        expected[:n] = torch.where(a > 0, 2 * a, a - 3)
    else:
        expected[:n] = a
    torch.testing.assert_close(f(x.npu(), seed.npu()).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scope", ["branch", "serial"])
@pytest.mark.parametrize("kind", ["var", "local"])
def test_private_declaration_cannot_escape(scope, kind):
    with pytest.raises(Exception, match="used outside its defining region"):

        @T.prim_func
        def invalid():
            with T.Kernel(1):
                a = T.alloc_shared((64,), "float32")
                with T.SimdVF(lanes=64):
                    for i in T.Parallel(64):
                        if scope == "branch":
                            if a[i] > 0:
                                if kind == "var":
                                    tmp = T.alloc_var("float32", init=a[i])
                                else:
                                    tmp = T.alloc_local((1,), "float32")
                                    tmp[0] = a[i]
                        else:
                            for _ in T.serial(2):
                                if kind == "var":
                                    tmp = T.alloc_var("float32", init=a[i])
                                else:
                                    tmp = T.alloc_local((1,), "float32")
                                    tmp[0] = a[i]
                        if kind == "var":
                            a[i] = tmp
                        else:
                            a[i] = tmp[0]
