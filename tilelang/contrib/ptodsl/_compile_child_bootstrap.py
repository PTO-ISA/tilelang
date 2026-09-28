"""Load PTODSL contrib helpers without importing the TileLang package.

The PTO compile child runs `kernel.ptodsl.py`, which does
``from tilelang.contrib.ptodsl.* import ...``. A normal package import
executes ``tilelang/__init__.py``, which preloads torch and then
torch_npu. That second process fights the parent for the NPU and spins
at 100% CPU. Stub the parent packages and load only the helper files.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_HELPERS = ("dcache_bypass", "simt", "gemm", "rng")


def preload() -> None:
    contrib = Path(__file__).resolve().parent
    for pkg in ("tilelang", "tilelang.contrib"):
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__package__ = pkg
            module.__path__ = []
            sys.modules[pkg] = module

    package = "tilelang.contrib.ptodsl"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__package__ = package
        module.__path__ = [str(contrib)]
        sys.modules[package] = module

    for name in _HELPERS:
        full_name = f"{package}.{name}"
        source = contrib / f"{name}.py"
        if full_name in sys.modules:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"PTODSL compile-child helper missing: {source}")
        spec = importlib.util.spec_from_file_location(full_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
