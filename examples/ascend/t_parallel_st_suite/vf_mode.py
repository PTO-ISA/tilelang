#!/usr/bin/env python3
"""Select the VF frame and compile target for the ST suite's ``T.Parallel`` bodies.

``ST_VF_MODE`` (default ``simt-asc``):

| value | aliases | frame | compile target |
|-------|---------|-------|----------------|
| ``simt-asc`` | ``simt``, ``simtvf``, ``simt-ascend`` | ``T.SimtVF`` | ``ascend`` |
| ``simt-pto`` | ``simtvf-pto`` | ``T.SimtVF`` | ``pto`` |
| ``simd-asc`` | ``simd``, ``simdvf``, ``simd-ascend`` | ``T.SimdVF`` | ``ascend`` |
| ``simd-pto`` | ``simdvf-pto`` | ``T.SimdVF`` | ``pto`` |

``vf_frame`` / ``vf_region`` return only the frame. ``compile_target`` is separate
so a PTO mode is not compiled as ``ascend``. ``threads`` stays on the call so
kernel sites can be ``with vf_region(threads):``. ``T.SimdVF`` has no thread
dimensions, so that argument is not forwarded in simd modes.
"""

from __future__ import annotations

import os
from typing import Literal, NoReturn

VFMode = Literal["simt-asc", "simt-pto", "simd-asc", "simd-pto"]
CompileTarget = Literal["ascend", "pto"]
VFFrameName = Literal["SimtVF", "SimdVF"]

CANONICAL_VF_MODES: tuple[VFMode, ...] = ("simt-asc", "simt-pto", "simd-asc", "simd-pto")

_ALIASES: dict[str, VFMode] = {
    "simt-asc": "simt-asc",
    "simt": "simt-asc",
    "simtvf": "simt-asc",
    "simt-ascend": "simt-asc",
    "simt-pto": "simt-pto",
    "simtvf-pto": "simt-pto",
    "simd-asc": "simd-asc",
    "simd": "simd-asc",
    "simdvf": "simd-asc",
    "simd-ascend": "simd-asc",
    "simd-pto": "simd-pto",
    "simdvf-pto": "simd-pto",
}

_announced = False


def _never(value: object) -> NoReturn:
    raise AssertionError(f"unhandled VF mode: {value!r}")


def resolve_vf_mode(raw: str | None = None) -> VFMode:
    """Return one of ``simt-asc``, ``simt-pto``, ``simd-asc``, ``simd-pto``.

    ``raw is None`` reads ``ST_VF_MODE``. Empty means the default ``simt-asc``.
    """
    if raw is None:
        raw = os.environ.get("ST_VF_MODE", "")
    text = raw.strip().lower()
    if not text:
        text = "simt-asc"
    mode = _ALIASES.get(text)
    if mode is None:
        allowed = (
            "simt-asc, simt-pto, simd-asc, simd-pto (aliases: simt, simtvf, simt-ascend, simtvf-pto, simd, simdvf, simd-ascend, simdvf-pto)"
        )
        raise ValueError(f"ST_VF_MODE={raw!r} is not one of {allowed}")
    match mode:
        case "simt-asc" | "simt-pto" | "simd-asc" | "simd-pto":
            return mode
        case _ as unreachable:
            _never(unreachable)


def _frame_of(mode: VFMode) -> VFFrameName:
    match mode:
        case "simt-asc" | "simt-pto":
            return "SimtVF"
        case "simd-asc" | "simd-pto":
            return "SimdVF"
        case _ as unreachable:
            _never(unreachable)


def _target_of(mode: VFMode) -> CompileTarget:
    match mode:
        case "simt-asc" | "simd-asc":
            return "ascend"
        case "simt-pto" | "simd-pto":
            return "pto"
        case _ as unreachable:
            _never(unreachable)


def compile_target(raw: str | None = None) -> CompileTarget:
    """``ascend`` or ``pto`` for the active (or given) ``ST_VF_MODE``."""
    return _target_of(resolve_vf_mode(raw))


def announce_vf_mode() -> VFMode:
    """Print ``ST_VF_MODE=<mode> frame=<frame> target=<target>`` once and return the mode."""
    global _announced
    mode = resolve_vf_mode()
    if not _announced:
        print(
            f"ST_VF_MODE={mode} frame={_frame_of(mode)} target={_target_of(mode)}",
            flush=True,
        )
        _announced = True
    return mode


def vf_frame(threads: int | list[int] | tuple = 128, latency: int = 0):
    """Frame for the active ``ST_VF_MODE`` (SimtVF or SimdVF only).

    The compile target is :func:`compile_target`, not this frame.
    The Ascend dialect import stays inside this function so importing
    ``vf_mode`` does not load TileLang before ``boot()`` stubs ``torch_npu``.
    The returned object is the ``SimtVF`` / ``SimdVF`` frame; the eager
    builder enters that frame directly.
    """
    import tilelang.ascend.language as T

    mode = resolve_vf_mode()
    lat = int(latency)
    frame = _frame_of(mode)
    match frame:
        case "SimtVF":
            return T.SimtVF(threads=threads, latency=lat)
        case "SimdVF":
            return T.SimdVF(latency=lat)
        case _ as unreachable:
            _never(unreachable)


def vf_region(threads: int | list[int] | tuple = 128, latency: int = 0):
    """Alias of :func:`vf_frame`. Kernel sites use ``with vf_region(threads):``."""
    return vf_frame(threads, latency)
