from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def _make_ascend_target(target_dict: dict | None = None) -> Target:
    target_dict = dict(target_dict or {})
    target_dict["kind"] = "ascend"
    return Target(target_dict)


def target_is_ascend(target: Target) -> bool:
    """Return whether *target* uses the Ascend architecture."""
    return _target_ffi_api().TargetIsAscend(target)


def target_is_pto(target: Target) -> bool:
    """Return whether *target* selects the PTO backend."""
    return target_is_ascend(target) and "pto" in target.keys


def target_is_plain_ascend(target: Target) -> bool:
    """Return whether *target* selects the AscendC backend."""
    return target_is_ascend(target) and "pto" not in target.keys


def _make_pto_target(target_dict: dict | None = None) -> Target:
    target_dict = dict(target_dict or {})
    keys = target_dict.get("keys", ())
    if isinstance(keys, str):
        keys = (keys,)
    target_dict["kind"] = "ascend"
    # PTO reuses Ascend lowering while the "pto" key selects its backend.
    target_dict["keys"] = list(dict.fromkeys([*keys, "pto", "ascend"]))
    return Target(target_dict)


def _with_pto_key(target: Target) -> Target:
    target_dict = dict(target.export())
    target_dict["kind"] = "ascend"
    target_dict["keys"] = list(dict.fromkeys([*target_dict.get("keys", ()), "pto", "ascend"]))
    return Target(target_dict)


def check_ascend_availability() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _detect_ascend_target() -> Target | None:
    if check_ascend_availability():
        return _make_ascend_target()
    return None


def normalize_ascend_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "ascend":
        return None

    try:
        return _make_ascend_target()
    except Exception:
        return None


def normalize_pto_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if "pto" not in target.keys:
            return None
        return _with_pto_key(target)

    if isinstance(target, dict):
        target_dict = dict(target)
        keys = target_dict.get("keys", ())
        if isinstance(keys, str):
            keys = (keys,)
        if target_dict.get("kind") == "pto" or "pto" in keys:
            try:
                return _make_pto_target(target_dict)
            except Exception:
                return None

        try:
            parsed_target = Target(target_dict)
        except Exception:
            return None
        if "pto" in parsed_target.keys:
            return _with_pto_key(parsed_target)
        return None

    if not isinstance(target, str) or target.strip() != "pto":
        return None

    try:
        return _make_pto_target()
    except Exception:
        return None


def normalize_asc_target(target: TargetLike) -> Target | None:
    """Accept ``asc`` as the concise name for the AscendC backend."""
    if isinstance(target, str) and target.strip() == "asc":
        return normalize_ascend_target("ascend")
    return None


register_target_detector("ascend", _detect_ascend_target, override=True)
register_target_normalizer("ascend", normalize_ascend_target, override=True)
register_target_normalizer("asc", normalize_asc_target, override=True)
register_target_normalizer("pto", normalize_pto_target, override=True)
