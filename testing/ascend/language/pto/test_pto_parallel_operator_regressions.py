"""CPU-only end-to-end regressions for Parallel operator scheduling and values."""

import pytest
from tvm.ir.transform import PassContext
from tilelang.ascend.target import normalize_pto_target

import tilelang.ascend.language as T
from tilelang.engine.lower import lower


def division(n=77):
    padded = (n + 63) // 64 * 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((padded,), "float32")
            b = T.alloc_shared((padded,), "float32")
            c = T.alloc_shared((padded,), "float32")
            T.copy(A, a[:n])
            T.copy(B, b[:n])
            with T.SimdVF(lanes=64):
                for j in T.Parallel(n):
                    c[j] = a[j] / b[j]
            T.copy(c[:n], C)

    return main


@pytest.mark.parametrize("fast_math", [False, True])
def test_parallel_division_precision(fast_math):
    with PassContext(config={"tl.enable_fast_math": fast_math}):
        source = lower(division(), target="pto").kernel_source
    assert "pto.vmi.vdiv(" in source
    assert ('precision="precision_0ulp_ftz_true"' in source) != fast_math


@pytest.mark.parametrize("stages", [1, 2, 3, 7])
def test_scheduled_accumulator(stages):
    @T.prim_func
    def main(A: T.Tensor((17, 4, 128), "float32"), B: T.Tensor((17, 128), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((4, 128), "float32")
            b = T.alloc_shared((128,), "float32")
            T.annotate_buffer_versions({a: stages, b: stages})
            for row in T.Pipelined(17, num_stages=stages):
                T.copy(A[row, :, :], a)
                with T.SimdVF(lanes=64):
                    for j in T.Parallel(128):
                        acc = T.alloc_var(T.float32, init=0.0)
                        for k in T.serial(4):
                            acc += a[k, j]
                        b[j] = acc / 4.0 + 1e-20
                T.copy(b, B[row, :])

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vadd(" in source
    assert "pto.vmi.vstore(" in source


def test_single_valid_lane():
    @T.prim_func
    def main(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            T.copy(A, a[:1])
            with T.SimdVF(lanes=64):
                for j in T.Parallel(1):
                    b[j] = a[j]
            T.copy(b[:1], B)

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vstore(" in source


def test_implicit_shape_bound():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n, 64), "float32"), B: T.Tensor((n, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((16, 64), "float32")
            b = T.alloc_shared((16, 64), "float32")
            T.assume(n >= 0)
            T.assume(n <= 16)
            T.copy(A[:n, :], a[:n, :])
            with T.SimdVF(lanes=64):
                for i in T.serial(n):
                    for j in T.Parallel(64):
                        b[i, j] = a[i, j] + 1.0
            T.copy(b[:n, :], B[:n, :])

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vadd(" in source


def test_enclosing_scalar_is_broadcast_not_lane_local():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32"), scale: T.float32):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "float32")
            b = T.alloc_shared((64,), "float32")
            weight = T.alloc_var(T.float32, init=scale)
            weight = weight * 0.5
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for j in T.Parallel(64):
                    b[j] = a[j] * weight
            T.copy(b, B)

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vbrc(" in source
    assert "weight_lane" not in source


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int16", "uint16"])
def test_integer_point_write(dtype):
    @T.prim_func
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((64,), dtype)
            b = T.alloc_shared((64,), dtype)
            T.copy(A, a[:8])
            with T.SimdVF(lanes=64):
                for row in T.serial(8):
                    for point in T.Parallel(1):
                        b[row + point] = a[row + point]
            T.copy(b[:8], B)

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vcmax(" in source
    assert "size=1)" in source


@pytest.mark.parametrize("kind", ["sum", "max", "min", "abssum"])
def test_strided_reduction(kind):
    @T.prim_func
    def main(A: T.Tensor((3, 77), "float32"), B: T.Tensor((77,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((3, 128), "float32")
            b = T.alloc_shared((128,), "float32")
            T.copy(A, a[:, :77])
            with T.SimdVF(lanes=64):
                T.reduce(a[:, :77], b[:77], kind, dim=0, clear=True)
            T.copy(b[:77], B)

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vgather(" in source
    assert "pto.vmi.vstore(" in source


def test_gather_version_base_does_not_enter_u16_offsets():
    @T.prim_func
    def main(A: T.Tensor((4, 128, 512), "bfloat16"), B: T.Tensor((4, 128, 64), "bfloat16")):
        with T.Kernel(1):
            a = T.alloc_shared((128, 512), "bfloat16")
            b = T.alloc_shared((128, 64), "bfloat16")
            T.annotate_buffer_versions({a: 2, b: 2})
            for stage in T.Pipelined(4, num_stages=2):
                T.copy(A[stage, :, :], a)
                with T.SimdVF(lanes=64):
                    for row, j in T.Parallel(128, 64):
                        b[row, j] = a[row, j * 8]
                T.copy(b, B[stage, :, :])

    source = lower(main, target="pto").kernel_source
    assert "pto.vmi.vgather(" in source
    # Index casts represent the per-row coordinate; the much larger row and
    # version offsets belong to the gather's pointer rather than u16 lanes.
    index_lines = [line for line in source.splitlines() if line.strip().startswith("gather_indices") and "=" in line]
    assert index_lines and all("65536" not in line for line in index_lines)


def test_nested_serial_index_in_value_predicate():
    @T.prim_func
    def main(A: T.Tensor((8, 64), "float32"), B: T.Tensor((64,), "float32"), n: T.int32):
        with T.Kernel(1):
            a = T.alloc_shared((8, 64), "float32")
            b = T.alloc_shared((64,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for j in T.Parallel(64):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for row in T.serial(8):
                        acc += T.if_then_else(row < n, a[row, j], 0.0)
                    b[j] = acc
            T.copy(b, B)

    assert "pto.vmi.vsel(" in lower(main, target="pto").kernel_source


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_integer_atomic_signless_boundary(dtype):
    @T.prim_func
    def main(C: T.Tensor((1,), dtype), Previous: T.Tensor((1,), dtype)):
        with T.Kernel(1), T.SimtVF(threads=1):
            for i in T.Parallel(1):
                Previous[i] = T.atomic_add(C[i], T.cast(3, dtype), return_prev=True)

    with normalize_pto_target("pto"):
        source = lower(main, target="pto").kernel_source
    width = 64 if "64" in dtype else 32
    assert f'pto.ptr(pto.i{width}, "gm")' in source
    assert "pto.cast(pto.atomic_add(" in source


def test_integer_compute_dag_growth_is_bounded():
    @T.prim_func
    def main(A: T.Tensor((64,), "int32"), B: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "int32")
            b = T.alloc_shared((64,), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for j in T.Parallel(64):
                    p0 = a[j] * a[j]
                    p1 = p0 * p0
                    p2 = p1 * p1
                    p3 = p2 * p2
                    p4 = p3 * p3
                    p5 = p4 * p4
                    p6 = p5 * p5
                    p7 = p6 * p6
                    b[j] = p7
            T.copy(b, B)

    source = lower(main, target="pto").kernel_source
    # The unbounded inliner expands eight squarings into 128 VMI calls.
    assert source.count("pto.vmi.vmul(") < 64
    assert len(source) < 12000
