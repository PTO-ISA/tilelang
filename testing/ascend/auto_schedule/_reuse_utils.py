"""Shared buffer-reuse fixtures and alias-map inspection."""

import tilelang.ascend.language as T
from tilelang.ascend import transform
from tilelang import tvm
from testing.ascend._ir import schedule, nodes


def _make_cyclic_scalar_program(interleave: bool = False, versions: int = 1):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            if versions > 1:
                T.annotate_buffer_versions({a: versions, b: versions})
            for k in T.Pipelined(8, num_stages=1):
                a[0] = x[k]
                if interleave:
                    b[0] = y[k]
                out_x[k] = a[0]
                if not interleave:
                    b[0] = y[k]
                out_y[k] = b[0]

    return main


def _insert_sync_alias_pairs(program, *, disable_reuse=False) -> set[tuple[str, str]]:
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}):
        before = schedule(program, through="ResolveCore", auto_schedule=False)
        mod = transform.InsertSync()(before)
    pairs = set()
    for block in nodes(mod, tvm.tirx.SBlock):
        for storage, compatible in block.annotations.get("tl.buffer_alias_map", {}).items():
            for other in compatible:
                pairs.add(tuple(sorted((storage.name, other.name))))
    return pairs


def _make_pto_reduce_reuse_program(
    extent=130,
    lanes=128,
    dim=1,
    dtype="float32",
    out_dtype="float32",
    clear=True,
    versions=1,
    interleave=False,
    keep_padding=False,
    source_offset=0,
    complete_read=True,
):
    """Allocate the complete physical read, then test storage lifetime reuse.

    The strided allocation itself owns its row padding. The owner alias is used
    only to initialize that capacity; reduce still receives a complete logical
    Buffer. An ordinary scalar result read completes the VF before later is
    overwritten, while its DMA consumer independently exercises ring flags.
    """
    from tilelang import tvm

    rows = 1 if dim == 1 else 3
    width = max(lanes, 2048 // tvm.DataType(dtype).bits)
    pitch = (extent + source_offset + width - 1) // width * width
    if source_offset:
        # Leave a complete extra physical vector for the last row's offset.
        pitch += width
    output_count = rows if dim == 1 else extent

    @T.prim_func
    def main(
        A: T.Tensor((4,), dtype),
        B: T.Tensor((4,), dtype),
        Initial: T.Tensor((4,), out_dtype),
        C: T.Tensor((4, output_count + 2), out_dtype),
    ):
        with T.Kernel(1):
            a = T.sblock_alloc_buffer((rows, extent), dtype, strides=(pitch, 1), elem_offset=source_offset, scope="shared.dyn")
            owner = T.StridedTensor((rows, pitch), (pitch, 1), dtype, data=a.data, scope="shared.dyn")
            result = T.alloc_shared((output_count,), out_dtype)
            later = T.alloc_shared((rows, pitch), dtype)
            if versions > 1:
                T.annotate_buffer_versions({a: versions, later: versions})
            if keep_padding:
                T.fill(owner, T.cast(-999, dtype))
            for t in T.Pipelined(4, num_stages=2):
                with T.Stage(0):
                    if not keep_padding:
                        T.fill(owner, T.cast(-999, dtype))
                with T.Stage(0):
                    T.fill(a, A[t])
                with T.Stage(0):
                    T.fill(result, Initial[t])
                with T.Stage(0):
                    if interleave:
                        T.fill(later, B[t])
                with T.Stage(0), T.SimdVF(lanes=lanes):
                    T.reduce_sum(a, result, dim=dim, clear=clear)
                with T.Stage(0):
                    T.copy(result, C[t, 0:output_count])
                with T.Stage(0):
                    if complete_read:
                        C[t, output_count] = result[0]
                with T.Stage(0):
                    if not interleave:
                        T.fill(later, B[t])
                with T.Stage(0):
                    C[t, output_count + 1] = later[0, 0]
                with T.Stage(0):
                    if not complete_read:
                        C[t, output_count] = result[0]

    return main
