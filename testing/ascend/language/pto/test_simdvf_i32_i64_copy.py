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
