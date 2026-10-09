#!/usr/bin/env python3
"""Select ``T.SimtVF`` or ``T.SimdVF`` for the ST suite's shared ``T.Parallel`` bodies.

``ST_VF_MODE`` (default ``simt``):

- ``simt`` / ``simtvf`` → ``T.SimtVF(threads=threads, latency=latency)``
- ``simd`` / ``simdvf`` → ``T.SimdVF(latency=latency)``

``threads`` stays on the call so kernel sites can be ``with vf_region(threads):``.
``T.SimdVF`` has no thread dimensions, so that argument is not forwarded in simd mode.
"""

from __future__ import annotations

import os
from typing import Literal, NoReturn

VFMode = Literal["simt", "simd"]

_ALIASES: dict[str, VFMode] = {
    "simt": "simt",
    "simtvf": "simt",
    "simd": "simd",
    "simdvf": "simd",
}

_announced = False


def _never(value: object) -> NoReturn:
    raise AssertionError(f"unhandled VF mode: {value!r}")


def resolve_vf_mode(raw: str | None = None) -> VFMode:
    """Return ``simt`` or ``simd``.

    ``raw is None`` reads ``ST_VF_MODE``. Empty means the default ``simt``.
    """
    if raw is None:
        raw = os.environ.get("ST_VF_MODE", "")
    text = raw.strip().lower()
    if not text:
        text = "simt"
    mode = _ALIASES.get(text)
    if mode is None:
        allowed = "simt, simd (aliases simtvf, simdvf)"
        raise ValueError(f"ST_VF_MODE={raw!r} is not one of {allowed}")
    match mode:
        case "simt":
            return "simt"
        case "simd":
            return "simd"
        case _ as unreachable:
            _never(unreachable)


def announce_vf_mode() -> VFMode:
    """Print ``ST_VF_MODE=<mode>`` once per process and return the mode."""
    global _announced
    mode = resolve_vf_mode()
    if not _announced:
        print(f"ST_VF_MODE={mode}", flush=True)
        _announced = True
    return mode


def vf_region(threads: int | list[int] | tuple = 128, latency: int = 0):
    """Frame for the active ``ST_VF_MODE``.

    The Ascend dialect import stays inside this function so importing
    ``vf_mode`` does not load TileLang before ``boot()`` stubs ``torch_npu``.
    The returned object is the ``SimtVF`` / ``SimdVF`` frame; the eager
    builder enters that frame directly.
    """
    import tilelang.ascend.language as T

    mode = resolve_vf_mode()
    lat = int(latency)
    match mode:
        case "simt":
            return T.SimtVF(threads=threads, latency=lat)
        case "simd":
            return T.SimdVF(latency=lat)
        case _ as unreachable:
            _never(unreachable)
