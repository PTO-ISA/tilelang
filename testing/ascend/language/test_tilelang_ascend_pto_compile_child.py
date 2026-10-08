"""PTO compile child must not import TileLang / torch / torch_npu."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BOOTSTRAP = (
    Path(__file__).resolve().parents[3]
    / "tilelang"
    / "contrib"
    / "ptodsl"
    / "_compile_child_bootstrap.py"
)


@pytest.mark.pto
def test_ptodsl_compile_child_bootstrap_skips_torch():
    assert BOOTSTRAP.is_file()
    script = """
import importlib.util
import pathlib
import sys

bootstrap = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_tl_pto_bootstrap", bootstrap)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.preload()

from tilelang.contrib.ptodsl.dcache_bypass import pto_read_gm_bypass_dcache
from tilelang.contrib.ptodsl.simt import scalar_div

assert "torch" not in sys.modules
assert "torch_npu" not in sys.modules
assert "tilelang.jit" not in sys.modules
assert "tilelang.language" not in sys.modules
assert "tilelang.ascend" not in sys.modules
assert getattr(sys.modules["tilelang"], "__path__", None) == []
assert callable(pto_read_gm_bypass_dcache)
assert callable(scalar_div)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(BOOTSTRAP)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
