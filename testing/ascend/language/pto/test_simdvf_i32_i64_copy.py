"""NPU numeric acceptance for the PTO int32 -> int64 widening copy (Task 2).

The Ascend frontend spells ``T.copy(ub_i32, gm_i64)`` as an
``tl.tileop.ascend_copy``; ``ExpandDtypeMismatchedCopy`` widens i32 UB
elements into an int64 scratch (VMI ``vcmps/vsel/vintlv`` for K in
``{2,4,8,64}``, signed serial cast otherwise) and emits a same-dtype copy to
the original int64 GM region.

All cases run on a real NPU (the runtime guard only skips when the device is
genuinely unavailable) and compare full ``torch.int64`` values -- the output
is never cast back to int32.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1
PATTERN = [INT_MIN, -1, 0, 1, INT_MAX]
_SENTINEL = 1234567


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")
    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


def _cycled(n):
    return [PATTERN[i % len(PATTERN)] for i in range(n)]


def _kernel(k, *, ub_size, out_size, ub_base, out_base):
    @T.prim_func
    def main(A: T.Tensor((ub_size,), "int32"), Out: T.Tensor((out_size,), "int64")):
        with T.Kernel(1):
            ub = T.alloc_shared((ub_size,), "int32")
            T.copy(A, ub)
            T.copy(ub[ub_base : ub_base + k], Out[out_base : out_base + k])

    return main


def _kernel_observed(
    k, *, ub_size, out_size, ub_base, out_base, assume_first_nonneg
):
    """Kernel that optionally carries a real frontend ``T.assume`` about the
    first source element and reads the source UB back to GM after the widening
    copy, so the device test can observe that the source was not modified."""

    @T.prim_func
    def main(
        A: T.Tensor((ub_size,), "int32"),
        Out: T.Tensor((out_size,), "int64"),
        SrcOut: T.Tensor((ub_size,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((ub_size,), "int32")
            T.copy(A, ub)
            if assume_first_nonneg:
                T.assume(0 <= ub[0])
            T.copy(ub[ub_base : ub_base + k], Out[out_base : out_base + k])
            T.copy(ub, SrcOut)

    return main


def _run_observed(k, data, *, ub_size, out_size, ub_base=0, out_base=0, assume_first_nonneg=True):
    _require_npu_runtime()
    kernel = tilelang.compile(
        _kernel_observed(
            k,
            ub_size=ub_size,
            out_size=out_size,
            ub_base=ub_base,
            out_base=out_base,
            assume_first_nonneg=assume_first_nonneg,
        ),
        target="pto",
    )
    a_list = [_SENTINEL] * ub_size
    a_list[ub_base : ub_base + k] = data
    a = torch.tensor(a_list, dtype=torch.int32).to(torch.device("npu"))
    out = torch.full((out_size,), _SENTINEL, dtype=torch.int64).to(torch.device("npu"))
    src_out = torch.full((ub_size,), -12345, dtype=torch.int32).to(torch.device("npu"))
    kernel(a, out, src_out)
    torch.npu.synchronize()
    return out.cpu(), src_out.cpu()


def _run(k, data, *, ub_size=None, out_size=None, ub_base=0, out_base=0, out_fill=_SENTINEL):
    _require_npu_runtime()
    ub_size = ub_size if ub_size is not None else k
    out_size = out_size if out_size is not None else k
    kernel = tilelang.compile(
        _kernel(k, ub_size=ub_size, out_size=out_size, ub_base=ub_base, out_base=out_base),
        target="pto",
    )
    a_list = [_SENTINEL] * ub_size
    a_list[ub_base : ub_base + k] = data
    a = torch.tensor(a_list, dtype=torch.int32).to(torch.device("npu"))
    out = torch.full((out_size,), out_fill, dtype=torch.int64).to(torch.device("npu"))
    kernel(a, out)
    torch.npu.synchronize()
    return out.cpu()


@pytest.mark.parametrize(
    "k,data",
    [
        pytest.param(8, _cycled(8), id="k8"),
        pytest.param(64, _cycled(64), id="k64"),
        pytest.param(1, [-1], id="k1"),
        pytest.param(6, [INT_MIN, -1, 0, 1, INT_MAX, -7], id="k6"),
    ],
)
def test_signed_copy_values(k, data):
    out = _run(k, data)
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor(data, dtype=torch.int64))


def test_element_assume_is_not_region_proof():
    """The kernel carries a real ``T.assume(0 <= ub[0])`` (frontend form,
    after source initialization and before the widening copy).  Only element 0
    is proven non-negative; element 1 must still widen to -1 (signed), not to
    4294967295.  The host test
    ``test_frontend_assume_reaches_the_pass_and_stays_signed`` additionally
    demonstrates the assumption reaches the pass input and the widen stays on
    vcmps/vsel.  The source readback proves the assumption-carrying kernel did
    not modify UB."""

    out, src_back = _run_observed(2, [0, -1], ub_size=2, out_size=2)
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([0, -1], dtype=torch.int64))
    assert torch.equal(src_back, torch.tensor([0, -1], dtype=torch.int32))


@pytest.mark.parametrize("k", [1, 2, 4, 6, 8, 16, 32, 64, 65], ids=lambda k: f"k{k}")
def test_copy_k_boundary(k):
    data = _cycled(k)
    out = _run(k, data)
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor(data, dtype=torch.int64))


def test_copy_tail_source_capacity():
    """Source backing is exactly K (no padding read may need more), plus a
    variant with adjacent source/destination sentinels and non-zero offsets.

    Sentinels prove the copy wrote exactly its region -- they cannot alone
    prove the source was not over-read; that claim rests on the IR evidence
    (``task2-ir-*`` artifacts).
    """

    k = 8
    data = _cycled(k)

    # Exact-capacity source backing (UB holds exactly K elements).
    out = _run(k, data)
    assert torch.equal(out, torch.tensor(data, dtype=torch.int64))

    # Adjacent sentinels + non-zero source/destination offsets, with the
    # source UB read back to GM after the widening copy so the device test
    # observes the source (and its sentinels) itself, not just the CPU input.
    out, src_back = _run_observed(
        k, data, ub_size=k + 2, out_size=k + 5, ub_base=1, out_base=2
    )
    assert out.dtype == torch.int64
    assert torch.equal(out[2 : 2 + k], torch.tensor(data, dtype=torch.int64))
    assert int(out[0]) == _SENTINEL and int(out[1]) == _SENTINEL
    assert torch.equal(
        out[2 + k :], torch.full((3,), _SENTINEL, dtype=torch.int64)
    ), "destination sentinels past the region must be untouched"
    expected_ub = torch.tensor(
        [_SENTINEL] + list(data) + [_SENTINEL], dtype=torch.int32
    )
    assert torch.equal(src_back, expected_ub), (
        "the UB source contents (including both neighboring sentinels) must be "
        "preserved by the widening copy"
    )


# ---------------------------------------------------------------------------
# Task 3: liveness across iterations with shared-memory reuse on/off
# ---------------------------------------------------------------------------


def _kernel_loop(k, *, ub_size, guard_size, out_size, ub_base, out_base):
    """Two consecutive widening copies in a serial loop, with an independent
    live guard buffer and a UB readback after the loop.

    Each iteration also bumps ``ub[ub_base]`` and ``guard[0]``, so both
    readbacks consume loop-carried values and cannot legally be hoisted above
    the loop: the post-loop observations really observe post-loop state.
    """

    @T.prim_func
    def main(
        A: T.Tensor((ub_size,), "int32"),
        G: T.Tensor((guard_size,), "int32"),
        Out: T.Tensor((out_size,), "int64"),
        GuardOut: T.Tensor((guard_size,), "int32"),
        SrcOut: T.Tensor((ub_size,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((ub_size,), "int32")
            guard = T.alloc_shared((guard_size,), "int32")
            T.copy(A, ub)
            T.copy(G, guard)
            for it in T.serial(2):
                T.copy(ub[ub_base : ub_base + k], Out[out_base + it * k : out_base + (it + 1) * k])
                ub[ub_base] = ub[ub_base] + 1
                guard[0] = guard[0] + 1
            T.copy(guard, GuardOut)
            T.copy(ub, SrcOut)

    return main


@pytest.mark.parametrize("k", [8, 64, 6], ids=["k8-pad", "k64", "k6-serial"])
@pytest.mark.parametrize("disable_reuse", [False, True], ids=["reuse", "no-reuse"])
def test_widen_loop_liveness(k, disable_reuse):
    """Across two loop iterations the source UB and an adjacent live buffer
    must survive while the per-iteration pad/scratch temporaries come and go,
    with shared-memory reuse both enabled and disabled."""

    from tilelang import PassConfigKey

    _require_npu_runtime()
    guard_size = 8
    ub_size = k + 2
    out_size = 2 * k + 3
    ub_base = 1
    out_base = 1
    compiled = tilelang.compile(
        _kernel_loop(
            k,
            ub_size=ub_size,
            guard_size=guard_size,
            out_size=out_size,
            ub_base=ub_base,
            out_base=out_base,
        ),
        target="pto",
        pass_configs={PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: disable_reuse},
    )

    data = _cycled(k)
    guard_data = [101, -202, 303, -404, 505, -606, 707, -808]
    a_list = [_SENTINEL] * ub_size
    a_list[ub_base : ub_base + k] = data
    device = torch.device("npu")
    a = torch.tensor(a_list, dtype=torch.int32).to(device)
    g = torch.tensor(guard_data, dtype=torch.int32).to(device)
    out = torch.full((out_size,), _SENTINEL, dtype=torch.int64).to(device)
    guard_out = torch.full((guard_size,), -7, dtype=torch.int32).to(device)
    src_out = torch.full((ub_size,), -9, dtype=torch.int32).to(device)
    compiled(a, g, out, guard_out, src_out)
    torch.npu.synchronize()

    out_cpu = out.cpu()
    # Documented loop math: iteration 0 widens the original data and then
    # bumps ub[ub_base]; iteration 1 widens the bumped data and bumps again.
    # Both readbacks consume these loop-carried values (hence post-loop).
    iter0 = list(data)
    iter1 = [data[0] + 1] + list(data[1:])
    expected = torch.tensor(iter0 + iter1, dtype=torch.int64)
    assert out_cpu.dtype == torch.int64
    assert torch.equal(out_cpu[out_base : out_base + 2 * k], expected), (
        f"both loop iterations must widen correctly (k={k}, reuse={not disable_reuse})"
    )
    assert int(out_cpu[0]) == _SENTINEL
    assert torch.equal(
        out_cpu[out_base + 2 * k :], torch.full((2,), _SENTINEL, dtype=torch.int64)
    )
    expected_guard = list(guard_data)
    expected_guard[0] += 2
    assert torch.equal(guard_out.cpu(), torch.tensor(expected_guard, dtype=torch.int32)), (
        "the independent live guard buffer must survive both iterations "
        "(with its own loop-carried increment)"
    )
    expected_ub = list(a_list)
    expected_ub[ub_base] = data[0] + 2
    assert torch.equal(src_out.cpu(), torch.tensor(expected_ub, dtype=torch.int32)), (
        "the source UB (including both neighboring sentinels and the "
        "loop-carried bump) must stay intact across both iterations"
    )


def _ptodsl_loop_bounds(lines):
    """(header, end) line indices of the first emitted ``for it in range(...)``."""
    header = next(i for i, line in enumerate(lines) if "for it in range(" in line)
    indent = len(lines[header]) - len(lines[header].lstrip())
    for j in range(header + 1, len(lines)):
        stripped = lines[j]
        if stripped.strip() and (len(stripped) - len(stripped.lstrip())) <= indent:
            return header, j
    return header, len(lines)


def _const_offset(text):
    """Trailing integer offset in an address expression like ``..., 80)``."""
    import re

    tail = re.findall(r",\s*(\d+)\)\s*$", text.strip())
    return int(tail[-1]) if tail else None


@pytest.mark.parametrize("k", [8, 64, 6], ids=["k8-pad", "k64", "k6-serial"])
def test_widen_loop_observation_is_post_loop(k):
    """The source/guard readback copies must run AFTER the widening loop in the
    final PTODSL and read the widened UB region itself.  Otherwise the sentinel
    numerics only observe pre-loop state and prove nothing about lifetime."""

    _require_npu_runtime()
    compiled = tilelang.compile(
        _kernel_loop(
            k, ub_size=k + 2, guard_size=8, out_size=2 * k + 3, ub_base=1, out_base=1
        ),
        target="pto",
    )
    lines = compiled.get_kernel_source().splitlines()
    _header, loop_end = _ptodsl_loop_bounds(lines)

    def after(predicate):
        return [i for i, line in enumerate(lines) if predicate(line) and i > loop_end]

    src_copies = after(lambda line: "SrcOut" in line and "mte_ub_gm" in line)
    guard_copies = after(lambda line: "GuardOut" in line and "mte_ub_gm" in line)
    assert src_copies, (
        "the SrcOut readback must run after the widening loop in the emitted PTODSL"
    )
    assert guard_copies, (
        "the GuardOut readback must run after the widening loop in the emitted PTODSL"
    )

    # The readbacks must read the widened source UB and the guard buffer
    # themselves (their merged byte bases), not some unrelated buffer.  The
    # UB/guard element bases are the ones the A->ub / G->guard copies used.
    import re

    def element_base(param):
        line = next(line for line in lines if f"mte_gm_ub({param}," in line)
        match = re.search(
            r'mte_gm_ub\(\w+, pto\.addptr\(pto\.castptr\(buf_dyn_shmem, pto\.ptr\(pto\.si32, "ub"\)\), (\d+)\)',
            line,
        )
        return int(match.group(1)) if match else None

    ub_base_elem = element_base("A")
    guard_base_elem = element_base("G")
    assert ub_base_elem is not None and guard_base_elem is not None

    def byte_base(line):
        match = re.search(
            r'pto\.castptr\(pto\.const\((\d+), dtype=pto\.int64\), pto\.ptr\(pto\.si32, "ub"\)\)',
            line,
        )
        if match:
            return int(match.group(1))
        if 'castptr(buf_dyn_shmem, pto.ptr(pto.si32, "ub"))' in line:
            return 4 * ub_base_elem
        return None

    src_line = lines[src_copies[0]].strip()
    guard_line = lines[guard_copies[0]].strip()
    src_byte = byte_base(src_line)
    guard_byte = byte_base(guard_line)
    assert src_byte == 4 * ub_base_elem, (
        f"the SrcOut readback must read the widened UB (base {4 * ub_base_elem} "
        f"bytes); got {src_byte} on: {src_line[:150]}"
    )
    assert guard_byte == 4 * guard_base_elem and guard_byte != src_byte, (
        f"the GuardOut readback must read the guard buffer (base "
        f"{4 * guard_base_elem} bytes); got {guard_byte} on: {guard_line[:150]}"
    )
