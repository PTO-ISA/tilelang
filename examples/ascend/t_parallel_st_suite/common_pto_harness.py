#!/usr/bin/env python3
"""Shared PTO SimdVF / VMI harness for Case-3 twins (sv5v / sv6v / sv8v).

Green SimdVF path (simd_vs_simt AB on pto-b10):
- ``target=pto`` (NOT ascend / NOT cython)
- overlay ``libtilelang.so`` with SimdVFLower* + GetPredicate(PrimExpr)
- ``T.SimdVF(lanes=…)``; prefer ``alloc_shared`` for VF working set

Do **not** mix with the Simt-family kernels:
- Those use ``common_asc_harness``. ``ST_VF_MODE`` selects their frame and target.
- Overlay GetPredicate(PrimExpr) vs deps GetPredicate(Var) — oneshot must restore BK.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import traceback
import types
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ.setdefault("TILELANG_DISABLE_DATA_RACE_CHECK", "1")


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


def patch_simdvf_ffi() -> None:
    """Best-effort SimdVF(lanes=) arity shim — leave stock if already OK."""
    import tilelang.ascend.language as T
    from tilelang import _ffi_api

    vf_mod = None
    try:
        import tilelang.ascend.language.frame as vf_mod  # type: ignore
    except Exception:
        try:
            import tilelang.language.simdvf as vf_mod  # type: ignore
        except Exception:
            vf_mod = None

    try:
        import inspect

        sig = inspect.signature(T.SimdVF)
        if "lanes" in sig.parameters:
            print(f"SimdVF FFI OK signature={sig}", flush=True)
            return
    except Exception as e:
        print(f"SimdVF signature probe: {type(e).__name__}: {e}", flush=True)

    try:
        cm = T.SimdVF(lanes=32)
        del cm
        print("SimdVF(lanes=32) call OK — leave stock wrapper", flush=True)
        return
    except TypeError:
        pass
    except Exception as e:
        print(f"SimdVF(lanes=) probe other: {type(e).__name__}: {e}", flush=True)

    def SimdVF(lanes: int = 128, latency: int = 0, source_index: int | None = None):
        if source_index is None:
            source_index = 0
            if vf_mod is not None:
                for name in ("_next_vf_source_index", "_alloc_vf_source_index"):
                    fn = getattr(vf_mod, name, None)
                    if callable(fn):
                        try:
                            source_index = fn("simdvf")
                            break
                        except Exception:
                            pass
        for args in (
            (int(lanes), int(latency), int(source_index)),
            (int(lanes), int(latency)),
            (int(lanes),),
        ):
            try:
                return _ffi_api.SimdVF(*args)
            except TypeError:
                continue
        raise TypeError("SimdVF FFI arity unknown — overlay lib missing?")

    T.SimdVF = SimdVF
    if vf_mod is not None:
        vf_mod.SimdVF = SimdVF
    print("PATCHED SimdVF FFI lanes/latency/source_index fallback", flush=True)


def ensure_pto_compile_env() -> None:
    os.environ.setdefault(
        "CPLUS_INCLUDE_PATH",
        "/usr/include/c++/12:/usr/include/aarch64-linux-gnu/c++/12",
    )


def compile_prim(prim, tag: str, target: str = "pto"):
    """Compile prim_func with PTO / SimdVF backend; export OUT/so/{tag}.so."""
    import tilelang

    if target == "ascend":
        raise RuntimeError(
            "REFUSING target=ascend for SimdVF/VMI twin — use common_asc_harness "
            "for SimtVF. VMI path is target=pto + overlay lib."
        )

    ensure_pto_compile_env()
    set_out(OUT)

    compile_kwargs = {
        "target": target,
        "out_idx": -1,
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
        f"== compile {tag} target={target} (SimdVF/VMI) "
        f"flags={compile_kwargs['compile_flags']} ==",
        flush=True,
    )

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
        except Exception:
            pass
        print(f"TIR_DUMP_FAIL {tag}: {e}", flush=True)

    try:
        compiled = tilelang.compile(prim, **compile_kwargs)
    except Exception:
        tb = traceback.format_exc()
        (OUT / "logs" / f"compile_fail_{tag}.txt").write_text(tb)
        print(tb)
        return None

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
            lowered_path.write_text("lowered_ir_unavailable")
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
    src_path = OUT / "sources" / f"{tag}_source.txt"
    src_path.write_text(src if isinstance(src, str) else str(src))
    print(f"SOURCE_DUMP {src_path}", flush=True)

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


def boot() -> None:
    set_out(OUT)
    stub_torch_npu()
    patch_simdvf_ffi()
