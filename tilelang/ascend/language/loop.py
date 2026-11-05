"""Ascend dialect of the loop constructs: the common ops plus Ascend knobs."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import frame

from tilelang.language.loop import unroll as _common_unroll

__all__ = ["unroll"]


def unroll(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    step: tirx.PrimExpr | None = None,
    *,
    explicit: bool = False,
    unroll_factor: int | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """The unrolled For statement, with the Ascend unroll-factor pragma.

    Same semantics as the common :func:`tilelang.language.loop.unroll`.
    ``unroll_factor`` is lowered as the ``"pragma_unroll_factor"`` annotation,
    which the Ascend codegen emits as ``#pragma unroll N``.

    Parameters
    ----------
    start, stop, step : PrimExpr
        Iteration range.
    explicit : bool
        Whether to explicitly unroll the loop at compile time.
    unroll_factor : Optional[int]
        Partial unroll factor, lowered as the ``"pragma_unroll_factor"``
        annotation.
    annotations : Optional[Dict[str, Any]]
        Additional loop annotations; values in it take precedence.
    """
    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if unroll_factor is not None:
        ann.setdefault("pragma_unroll_factor", unroll_factor)
    return _common_unroll(start, stop, step, explicit=explicit, annotations=ann or None)


def Parallel(
    *extents: int | tirx.PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Construct a Parallel loop and record its frame in the current SIMD VF."""
    from tilelang.language.loop import Parallel as common_parallel
    from tilelang import _ffi_api

    if annotations is not None and "tl.parallel_private_allocs" in annotations:
        raise ValueError("tl.parallel_private_allocs is reserved for the Ascend frontend")
    result = common_parallel(*extents, coalesced_width=coalesced_width, loop_layout=loop_layout, annotations=annotations)
    _ffi_api.RegisterParallelFrame(result)
    return result


__all__ += ["Parallel"]
