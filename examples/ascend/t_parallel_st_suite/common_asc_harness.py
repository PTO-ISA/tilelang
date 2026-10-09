#!/usr/bin/env python3
"""Shared ASC SimtVF harness: npu stub + int2→make_int2 patch + cython compile/export.

Extracted from simd_vs_simt_ab_oneshot/run_simd_vs_simt_perf.py (prior green AB).
Prefer deps-stack libtilelang.so; target=ascend + execution_backend=cython (NOT pto).
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import traceback
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ.setdefault("TILELANG_DISABLE_DATA_RACE_CHECK", "1")

NEG = np.float32(-3.402823e38).item()
INT_MAX = np.iinfo(np.int32).max


def bisheng_libstdcxx_fix() -> str:
    """Header passed to bisheng as ``-include``.

    ``BISHENG_LIBSTDCXX_FIX`` wins. Otherwise the header is taken from this
    directory or from ``$TILELANG_DEPS/examples/ascend/msprof_res/orig_pto_vmi_simd/``
    when that file exists.
    """
    explicit = os.environ.get("BISHENG_LIBSTDCXX_FIX", "").strip()
    if explicit:
        return explicit
    here = Path(__file__).resolve().parent
    candidates = [
        here / "bisheng_libstdcxx_clang_fix.h",
        here.parent / "msprof_res" / "orig_pto_vmi_simd" / "bisheng_libstdcxx_clang_fix.h",
    ]
    deps = os.environ.get("TILELANG_DEPS", "").strip()
    if deps:
        candidates.append(
            Path(deps) / "examples" / "ascend" / "msprof_res" / "orig_pto_vmi_simd" / "bisheng_libstdcxx_clang_fix.h"
        )
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise RuntimeError(
        "Set BISHENG_LIBSTDCXX_FIX to bisheng_libstdcxx_clang_fix.h (passed to bisheng as -include)."
    )


# Default remote OUT; callers may override via set_out()
OUT = Path(os.environ.get("ST_SIMTVF_OUT", "/tmp/t_parallel_st_suite"))


def set_out(path: str | Path) -> Path:
    global OUT
    OUT = Path(path)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "so").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    return OUT


def stub_torch_npu() -> None:
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        try:
            torch.utils.rename_privateuse1_backend("npu")
        except Exception:
            pass
        sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))
    try:
        ctypes.CDLL("libruntime.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    try:
        from tilelang.ascend import torch_exchange as te

        te.install_torch_npu_stream_exchange = lambda: False
    except Exception:
        pass


def _fix_uint2_half_pack(code: str) -> tuple[str, int]:
    """Rewrite illegal `uint2{half,half,half,half}` packs from ASC codegen.

    Emitted form packs four ``half`` into ``uint2`` via a narrowing ctor and
    fails bisheng (-Wc++11-narrowing). Rebuild as two half2 bitcasts.
    """
    import re

    pat = re.compile(
        r"uint2\s+(\w+)\s*=\s*uint2\{"
        r"\s*([^,{}]+)\s*,\s*([^,{}]+)\s*,\s*([^,{}]+)\s*,\s*([^,{}]+)\s*\}\s*;"
    )
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        name, a, b, c, d = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()
        return (
            f"half2 {name}_h0 = half2{{{a}, {b}}}; "
            f"half2 {name}_h1 = half2{{{c}, {d}}}; "
            f"uint2 {name} = uint2{{*reinterpret_cast<uint32_t*>(&{name}_h0), "
            f"*reinterpret_cast<uint32_t*>(&{name}_h1)}};"
        )

    return pat.sub(repl, code), n


def _fix_uint1_half_pack(code: str) -> tuple[str, int]:
    """Rewrite illegal `uint1{half,half}` packs from ASC codegen.

    Emitted form packs two ``half`` into ``uint1`` (uint32) via a narrowing
    ctor and fails bisheng (-Wc++11-narrowing / excess elements). Rebuild as
    a half2 bitcast into uint1.
    """
    pat = re.compile(
        r"uint1\s+(\w+)\s*=\s*uint1\{"
        r"\s*([^,{}]+)\s*,\s*([^,{}]+)\s*\}\s*;"
    )
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        name, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
        return (
            f"half2 {name}_h = half2{{{a}, {b}}}; "
            f"uint1 {name} = uint1{{*reinterpret_cast<uint32_t*>(&{name}_h)}};"
        )

    return pat.sub(repl, code), n


def _fix_int2_code(code: str) -> tuple[str, int]:
    """Rewrite illegal ASC `int2(...)` ctor to `make_int2(...)`."""
    if not code:
        return code, 0
    n = 0

    def repl(_m):
        nonlocal n
        n += 1
        return "make_int2("

    fixed = re.sub(r"(?<!make_)int2\(", repl, code)
    return fixed, n


_PATCHED = False


def patch_load_lib() -> None:
    """Hook bisheng.compile_ascend + ascend_compile callback + LibraryGenerator."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from tilelang.contrib import bisheng
    from tilelang.jit.adapter import libgen

    _ca = bisheng.compile_ascend

    def _ca_fix(code, *args, **kwargs):
        fixed, n = _fix_int2_code(code if isinstance(code, str) else str(code))
        if n:
            print(f"PATCHED bisheng.compile_ascend int2 -> make_int2 count={n}", flush=True)
            dump = OUT / "last_asc_int2_patch_snippet.txt"
            hits = [ln for ln in fixed.splitlines() if "make_int2" in ln or "int2" in ln]
            dump.write_text("\n".join(hits[:80]))
        fixed2, n2 = _fix_uint2_half_pack(fixed)
        if n2:
            print(f"PATCHED bisheng.compile_ascend uint2{{half..}} count={n2}", flush=True)
            fixed = fixed2
        fixed3, n3 = _fix_uint1_half_pack(fixed)
        if n3:
            print(f"PATCHED bisheng.compile_ascend uint1{{half,half}} count={n3}", flush=True)
            fixed = fixed3
        return _ca(fixed, *args, **kwargs)

    bisheng.compile_ascend = _ca_fix
    print("HOOKED bisheng.compile_ascend for int2", flush=True)

    try:
        import tvm_ffi

        _asc = tvm_ffi.get_global_func("tilelang_callback_ascend_compile", allow_missing=True)
        if _asc is not None:

            def _asc_fix_int2(code, target, pass_config=None):
                fixed, n = _fix_int2_code(code if isinstance(code, str) else str(code))
                if n:
                    print(f"PATCHED ascend_callback int2 -> make_int2 count={n}", flush=True)
                fixed2, n2 = _fix_uint2_half_pack(fixed)
                if n2:
                    print(f"PATCHED ascend_callback uint2{{half..}} count={n2}", flush=True)
                    fixed = fixed2
                fixed3, n3 = _fix_uint1_half_pack(fixed)
                if n3:
                    print(f"PATCHED ascend_callback uint1{{half,half}} count={n3}", flush=True)
                    fixed = fixed3
                return _asc(fixed, target, pass_config)

            tvm_ffi.register_global_func(
                "tilelang_callback_ascend_compile", _asc_fix_int2, override=True
            )
            print("HOOKED tilelang_callback_ascend_compile for int2", flush=True)
    except Exception as e:
        print(f"ascend_callback hook skipped: {type(e).__name__}: {e}", flush=True)

    _load = libgen.LibraryGenerator.load_lib

    def _load_optional(self, *args, **kwargs):
        try:
            return _load(self, *args, **kwargs)
        except OSError as e:
            if "rtFunctionRegister" not in str(e):
                raise
            return None

    libgen.LibraryGenerator.load_lib = _load_optional

    _compile = libgen.LibraryGenerator.compile_lib

    def _compile_fix_int2(self, *args, **kwargs):
        code = getattr(self, "lib_code", "") or ""
        fixed, n = _fix_int2_code(code)
        if n:
            self.lib_code = fixed
            code = fixed
            print(f"PATCHED lib_code int2 -> make_int2 count={n}", flush=True)
            dump = OUT / "last_int2_patch_snippet.txt"
            hits = [ln for ln in fixed.splitlines() if "make_int2" in ln or "int2" in ln]
            dump.write_text("\n".join(hits[:80]))
        fixed2, n2 = _fix_uint2_half_pack(code if isinstance(code, str) else str(code))
        if n2:
            self.lib_code = fixed2
            code = fixed2
            print(f"PATCHED lib_code uint2{{half..}} pack count={n2}", flush=True)
            dump = OUT / "last_uint2_half_patch_snippet.txt"
            dump.write_text("\n".join([ln for ln in fixed2.splitlines() if "uint2" in ln or "_h0" in ln][:80]))
        fixed3, n3 = _fix_uint1_half_pack(code if isinstance(code, str) else str(code))
        if n3:
            self.lib_code = fixed3
            print(f"PATCHED lib_code uint1{{half,half}} pack count={n3}", flush=True)
            dump = OUT / "last_uint1_half_patch_snippet.txt"
            dump.write_text("\n".join([ln for ln in fixed3.splitlines() if "uint1" in ln or "_h" in ln][:80]))
        return _compile(self, *args, **kwargs)

    libgen.LibraryGenerator.compile_lib = _compile_fix_int2

    try:
        from tilelang.jit.adapter.cython import adapter as cy_adapter

        _orig_init = cy_adapter.CythonKernelAdapter.__init__

        def _tolerant_init(self, *args, **kwargs):
            try:
                return _orig_init(self, *args, **kwargs)
            except (OSError, AttributeError) as e:
                print(f"adapter init tolerate: {type(e).__name__}: {e}", flush=True)
                self.lib = None
                return None

        cy_adapter.CythonKernelAdapter.__init__ = _tolerant_init
    except Exception as e:
        print(f"cython adapter hook skipped: {type(e).__name__}: {e}", flush=True)


def ensure_asc_compile_env() -> None:
    os.environ.setdefault(
        "CPLUS_INCLUDE_PATH",
        "/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12",
    )


def compile_prim(prim, tag: str, target: str = "ascend"):
    """Compile prim_func with ASC cython backend; export .so under OUT/so/{tag}.so."""
    import tilelang

    if target == "pto":
        raise RuntimeError(
            "REFUSING target=pto for SimtVF (deps may call SimdVFLowerControlFlow missing). Use ascend."
        )

    ensure_asc_compile_env()
    set_out(OUT)

    compile_kwargs = {
        "target": target,
        "out_idx": -1,
        "execution_backend": "cython",
        "compile_flags": [
            "-include",
            bisheng_libstdcxx_fix(),
            "-isystem",
            "/usr/include/c++/12",
        ],
    }
    try:
        from tilelang import PassConfigKey

        compile_kwargs["pass_configs"] = {PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}
    except Exception:
        pass

    print(
        f"== compile {tag} target={target} backend=cython flags={compile_kwargs['compile_flags']} ==",
        flush=True,
    )

    # TIR / TVMScript dump (pre-compile) — best-effort, never fail compile.
    tir_path = OUT / "sources" / f"{tag}_tir.txt"
    try:
        tir_txt = None
        for getter in (
            lambda: prim.script(),
            lambda: prim.show(style="plain"),
            lambda: str(prim),
        ):
            try:
                tir_txt = getter()
                if tir_txt:
                    break
            except Exception:
                continue
        if not tir_txt:
            try:
                import tvm

                mod = tvm.IRModule.from_expr(prim)
                tir_txt = mod.script() if hasattr(mod, "script") else str(mod)
            except Exception as e:
                tir_txt = f"tir_dump_err: {type(e).__name__}: {e}\nprim={prim!r}"
        tir_path.write_text(tir_txt if isinstance(tir_txt, str) else str(tir_txt))
        print(f"TIR_DUMP {tir_path}", flush=True)
    except Exception as e:
        try:
            tir_path.write_text(f"tir_dump_failed: {type(e).__name__}: {e}")
            print(f"TIR_DUMP_FAIL {tir_path}: {e}", flush=True)
        except Exception:
            print(f"TIR_DUMP_FAIL {tag}: {e}", flush=True)

    try:
        compiled = tilelang.compile(prim, **compile_kwargs)
    except Exception:
        tb = traceback.format_exc()
        (OUT / "logs" / f"compile_fail_{tag}.txt").write_text(tb)
        print(tb)
        return None

    # Lowered IR dump (post-compile) — best-effort.
    lowered_path = OUT / "sources" / f"{tag}_lowered.txt"
    try:
        lowered = None
        for attr in ("ir_module", "mod", "module", "prim_func"):
            obj = getattr(compiled, attr, None)
            if obj is None:
                continue
            try:
                lowered = obj.script() if hasattr(obj, "script") else str(obj)
                if lowered:
                    break
            except Exception:
                continue
        if lowered is None:
            ad = getattr(compiled, "adapter", None)
            if ad is not None:
                for attr in ("ir_module", "mod", "module", "scheduled_mod", "prim_func"):
                    obj = getattr(ad, attr, None)
                    if obj is None:
                        continue
                    try:
                        lowered = obj.script() if hasattr(obj, "script") else str(obj)
                        if lowered:
                            break
                    except Exception:
                        continue
        if lowered:
            lowered_path.write_text(lowered if isinstance(lowered, str) else str(lowered))
            print(f"LOWERED_DUMP {lowered_path}", flush=True)
        else:
            lowered_path.write_text("lowered_ir_unavailable (no ir_module/mod on compiled/adapter)")
            print(f"LOWERED_DUMP_SKIP {lowered_path}", flush=True)
    except Exception as e:
        try:
            lowered_path.write_text(f"lowered_dump_failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        print(f"LOWERED_DUMP_FAIL {tag}: {e}", flush=True)

    src = ""
    try:
        src = compiled.get_kernel_source() or ""
    except Exception as e:
        src = f"src_err: {e}"
    src_s = src if isinstance(src, str) else str(src)
    src_fixed, n = _fix_int2_code(src_s)
    src_fixed, n2 = _fix_uint2_half_pack(src_fixed)
    src_fixed, n3 = _fix_uint1_half_pack(src_fixed)
    src_path = OUT / "sources" / f"{tag}_source.txt"
    src_path.write_text(src_fixed)
    print(f"SOURCE_DUMP {src_path}", flush=True)
    if n:
        print(f"SOURCE_INT2_PATCH count={n}", flush=True)
    if n2:
        print(f"SOURCE_UINT2_PATCH count={n2}", flush=True)
    if n3:
        print(f"SOURCE_UINT1_PATCH count={n3}", flush=True)

    so = OUT / "so" / f"{tag}.so"
    lib_src = None
    ad = getattr(compiled, "adapter", None)
    if ad is not None and getattr(ad, "libpath", None):
        lib_src = ad.libpath
    lg = getattr(ad, "lib_generator", None) if ad is not None else None
    if lib_src is None and lg is not None and getattr(lg, "libpath", None):
        lib_src = lg.libpath
    if lib_src and Path(lib_src).is_file():
        shutil.copy2(lib_src, so)
        print(f"EXPORT_FROM libpath={lib_src}", flush=True)
    elif hasattr(compiled, "export_library"):
        compiled.export_library(str(so))
        print("EXPORT_FROM export_library", flush=True)
    else:
        raise RuntimeError("no libpath to export")

    nm = subprocess.run(["nm", "-u", str(so)], capture_output=True, text=True, check=False)
    if "TVMFFI" in (nm.stdout or ""):
        print("WARN so still has TVMFFI undef — opsim may fail under /usr/bin/python3", flush=True)
    print(f"COMPILE_OK {tag} {so} {so.stat().st_size}", flush=True)
    return so


def patch_simtvf_ffi() -> None:
    """Ensure SimtVF is available on the Ascend dialect (PTO-ISA pto-dev).

    On PTO-ISA/tilelang ``pto-dev``, SimtVF lives in
    ``tilelang.ascend.language.frame`` and already calls the 3-arg FFI
    ``SimtVF(Array, latency, source_index)``. Older Wenbo/deps layouts used
    ``tilelang.language.vf`` (2-arg or patched 3-arg). Prefer the stock Ascend
    wrapper; only fall back to a local 3-arg shim if needed.
    """
    import tilelang.ascend.language as T
    from tilelang import _ffi_api

    # Stock Ascend dialect already wires SimtVF — verify FFI arity.
    try:
        _ffi_api.SimtVF([32, 1, 1], 0, 0)
        print("SimtVF FFI 3-arg OK — using tilelang.ascend.language.SimtVF", flush=True)
        return
    except TypeError:
        pass
    try:
        _ffi_api.SimtVF([32, 1, 1], 0)
        print("SimtVF FFI 2-arg OK — leave stock wrapper", flush=True)
        return
    except TypeError:
        pass

    def SimtVF(threads: int | list | tuple = 128, latency: int = 0, source_index: int | None = None):
        if isinstance(threads, int):
            normalized = [threads, 1, 1]
        elif isinstance(threads, (list, tuple)):
            if len(threads) > 3:
                raise ValueError("SimtVF supports at most 3 thread dimensions")
            normalized = list(threads) + [1] * (3 - len(threads))
        else:
            raise TypeError(f"threads must be int, list, or tuple, got {type(threads)}")
        if any(t <= 0 for t in normalized):
            raise ValueError("SimtVF requires all thread dimensions > 0")
        if source_index is None:
            source_index = 0
        try:
            return _ffi_api.SimtVF(normalized, int(latency), int(source_index))
        except TypeError:
            return _ffi_api.SimtVF(normalized, int(latency))

    T.SimtVF = SimtVF
    print("PATCHED SimtVF FFI shim on tilelang.ascend.language", flush=True)



def boot() -> None:
    set_out(OUT)
    stub_torch_npu()
    patch_simtvf_ffi()
    patch_load_lib()
