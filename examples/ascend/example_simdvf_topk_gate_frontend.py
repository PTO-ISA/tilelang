"""Map TileKernels GPU topk_gate Parallel body onto Ascend PTO SimdVF.

Ported from PR #262 head (`pr262-tree-verify-9008cc1b`,
commit 9008cc1b285d80fa4275b18c623b9e15a4575813,
``examples/ascend/example_simdvf_topk_gate_frontend.py``,
sha256 492a5b0a91c8f3b8...).  Adaptations for the current pto-dev pipeline
(tracked as the 5B diff, see runs/n22-task5b-topk-20261012):

* pipeline contract is now ``Verify -> Legalize -> Vectorize`` (PR262 used
  ``SimdVFLowerControlFlow -> SimdVFLowerReduce -> Verify -> Vectorize``);
  ''assert_lowered_contract'' captures and checks the three PTO passes;
* the whole index chain is **int32** (PR #2 / 5B): tournament ``idx_cand``,
  ``best``, ``idx_ub`` and the GM ``IdxOut`` output are int32; the int64
  widening copy-out and its ``T.assume(0 <= idx_ub[0])`` are gone;
* the staging buffer ``s`` is allocated at ``P = align(E, lanes)`` and only
  the first ``E`` elements are copied; ``s[E:P]`` stays uninitialized but
  readable (design TopK contract);
* the candidate condition includes ``i < E``: the padded ``scores[E:P]`` hold
  ``-inf``, so without the guard a tail lane could tie with an all--inf valid
  score and leak a padding index into the result (tail lanes keep
  ``INT_MAX``);
* the TileLang dialect import is ``tilelang.ascend.language`` (the dialect
  that owns ``SimdVF`` / ``reduce_max`` / ``reduce_min`` here).

Upstream (deepseek-ai/TileKernels topk_gate_kernel.py) Parallel body::

    aligned = align(E, threads)  # Ascend: align(E, lanes)
    Parallel(aligned): if i < E: scores[i]=A[i] else -inf
    Parallel(aligned): idxs[i] = i
    for k in unroll/serial(K):
        reduce_max(scores, amax)
        fill(idx_reducer, INT_MAX)
        Parallel(aligned): if scores[i]==amax: idx_reducer = min(..., idxs[i])
        finalize_reducer; topk[k] = winner
        Parallel(aligned): if idxs[i]==winner: scores[i] = -inf

PTO adaptations (keep Parallel shape / tournament semantics)::

- ``T.Kernel(1)`` + ``T.SimdVF(lanes=L)`` instead of CUDA ``threads=32``
- Shared UB instead of fragment (Ascend converting path)
- ``T.reduce_min`` on a VL candidate buffer instead of ``alloc_reducer``+
  ``finalize_reducer`` (reducer is SIMT/cross-warp; the direct reduce path
  covers cross-lane min/max under SimdVF)
- ``T.serial(K)`` tournament loop (same as GPU unroll for small K)
- Keep ``if i < E else -inf`` in source; when ``E % lanes == 0`` at compile
  time, expect DCE of the else arm (do not delete the if/else)

Tie-break: smaller index wins (same as upstream).
As in PR262, replacing a selected score with -inf does not exclude it once
all remaining scores are -inf. The smallest valid index can repeat at that
boundary; distinct output indices are not guaranteed for those inputs.
"""

from __future__ import annotations

import ctypes
import os
import sys
import types
from contextlib import contextmanager

import numpy as np

os.environ.setdefault("TILELANG_DISABLE_DATA_RACE_CHECK", "1")
os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

E = 256
L = 128
K_DEFAULT = 8
INT_MAX = np.iinfo(np.int32).max


def align(x: int, y: int) -> int:
    """Ceil-align x up to multiple of y (TileKernels utils.align)."""
    return ((x + y - 1) // y) * y


def _prepare_torch_npu_stub():
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


def topk_gate_frontend(
    num_experts: int = E,
    num_topk: int = K_DEFAULT,
    lanes: int = L,
):
    """GPU-mapped Parallel body: pad+align, split init, reduce_max → min-index → kill."""
    import tilelang.ascend.language as T

    num_aligned = align(num_experts, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((num_experts,), "float32"),
        IdxOut: T.Tensor((num_topk,), "int32"),
    ):
        with T.Kernel(1):
            # s32 compute is valid: expert ids / Parallel indices fit int32.
            T.assume(0 <= num_experts)
            T.assume(num_experts <= 2147483647)
            T.assume(num_aligned <= 2147483647)
            # Staging buffer sized P: only the first E elements are copied;
            # s[E:P] is never read (the init guard keeps i < E).
            s = T.alloc_shared((num_aligned,), "float32")
            scores = T.alloc_shared((num_aligned,), "float32")
            idxs = T.alloc_shared((num_aligned,), "int32")
            idx_cand = T.alloc_shared((num_aligned,), "int32")
            idx_ub = T.alloc_shared((num_topk,), "int32")
            amax = T.alloc_shared((1,), "float32")
            best = T.alloc_shared((1,), "int32")
            T.copy(A, s[0:num_experts])
            # Init in its own SimdVF so PIPE_V can fence init vstores before
            # tournament vloads (board V->V RAW; camodel only warns).
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(num_aligned):
                    if i < num_experts:
                        scores[i] = s[i]
                    else:
                        scores[i] = -T.infinity(T.float32)
                for i in T.Parallel(num_aligned):
                    idxs[i] = i
            T.ascend_pipe_barrier("PIPE_V")
            # Tournament: one SimdVF per phase so PIPE_V sits outside vecscope.
            # The compare/index work runs on V (vcmp/vsel + plain UB access);
            # same-address store/load ordering inside one vector scope is the
            # toolchain's responsibility (PTOAS vecscope memory barriers).
            for k in T.serial(num_topk):
                with T.SimdVF(lanes=lanes):
                    T.reduce_max(scores, amax)
                    # Candidate condition includes i < E: the padded lanes
                    # hold -inf like a valid all--inf score would, so without
                    # the guard a tail lane could win the tie-break. Tail
                    # lanes keep INT_MAX in idx_cand.
                    for i in T.Parallel(num_aligned):
                        if (i < num_experts) & (scores[i] == amax[0]):
                            idx_cand[i] = idxs[i]
                        else:
                            idx_cand[i] = INT_MAX
                T.ascend_pipe_barrier("PIPE_V")
                with T.SimdVF(lanes=lanes):
                    T.reduce_min(idx_cand, best)
                    idx_ub[k] = best[0]
                    for i in T.Parallel(num_aligned):
                        if idxs[i] == best[0]:
                            scores[i] = -T.infinity(T.float32)
                T.ascend_pipe_barrier("PIPE_V")
            # Same-dtype int32 copy-out (GPU TileKernels returns int64; the
            # Ascend frontend keeps the whole index chain int32).
            T.copy(idx_ub, IdxOut)

    return main


@contextmanager
def _stage_capture():
    """Capture the three PTO Parallel passes around a full compile."""
    from tilelang.ascend import transform as ascend_transform

    captured = {"order": []}
    originals = {
        "verify": ascend_transform.VerifyParallelToPTO,
        "legalize": ascend_transform.LegalizeParallelToPTO,
        "vectorize": ascend_transform.VectorizeParallelToPTO,
    }

    def wrap(name):
        def factory():
            pass_fn = originals[name]()

            def run(mod):
                captured["order"].append(name)
                result = pass_fn(mod)
                captured[name] = result
                return result

            return run

        return factory

    try:
        for name in originals:
            setattr(ascend_transform, originals[name].__name__, wrap(name))
        yield captured
    finally:
        for name in originals:
            setattr(ascend_transform, originals[name].__name__, originals[name])


def compile_frontend(num_experts: int = E, num_topk: int = K_DEFAULT, lanes: int = L):
    _prepare_torch_npu_stub()
    import tilelang
    from tilelang.jit.adapter import libgen

    _load = libgen.LibraryGenerator.load_lib

    def _load_optional(self, *args, **kwargs):
        try:
            return _load(self, *args, **kwargs)
        except OSError as e:
            if "rtFunctionRegister" not in str(e):
                raise
            return None

    libgen.LibraryGenerator.load_lib = _load_optional
    return tilelang.compile(
        topk_gate_frontend(num_experts, num_topk, lanes),
        target="pto",
        out_idx=-1,
    )


def source_markers(compiled) -> dict:
    src = ""
    try:
        src = compiled.get_kernel_source() or ""
    except Exception:
        src = ""
    low = src.lower()
    return {
        "vcmax": "vcmax" in low,
        "vcmin": "vcmin" in low,
        "vmax": "vmax" in low,
        "vmin": "vmin" in low,
        "vsel": "vsel" in low,
        "vcmp": "vcmp" in low,
        "ifthenelse": "ifthenelse" in low,
        "tileop.reduce": "tileop.reduce" in low,
        "src_len": len(src),
    }


def assert_lowered_contract(compiled, captured: dict | None = None) -> dict:
    """Compile + source contract under the current pipeline.

    With *captured* (from :func:`_stage_capture`) the three PTO passes must
    have run in the order Verify -> Legalize -> Vectorize, Legalize must have
    removed the residual control flow, and Vectorize must have consumed the
    direct reduces.
    """
    markers = source_markers(compiled)
    assert markers["src_len"] > 0, "empty kernel source"
    assert markers["vcmax"] or markers["vmax"], markers
    assert markers["vcmin"] or markers["vmin"], markers
    assert markers["vsel"] or markers["vcmp"], markers
    assert not markers["ifthenelse"], markers
    assert not markers["tileop.reduce"], markers
    if captured is not None:
        assert captured["order"] == ["verify", "legalize", "vectorize"], captured["order"]
        assert "tileop.reduce" not in str(captured["vectorize"])
    return markers


def ref_topk_indices(scores_1d, k: int):
    """PR262 tournament reference, including repeated winners at -inf."""
    import torch

    scores = torch.as_tensor(scores_1d, dtype=torch.float32).clone()
    out = []
    neg = float("-inf")
    for _ in range(k):
        m = scores.max()
        idxs = (scores == m).nonzero(as_tuple=False).flatten()
        i = int(idxs.min())
        out.append(i)
        scores[i] = neg
    return torch.tensor(out, dtype=torch.int32)


if __name__ == "__main__":
    for e, k, lanes in ((E, K_DEFAULT, L), (256, 8, 64), (161, 8, 64)):
        with _stage_capture() as captured:
            compiled = compile_frontend(e, k, lanes)
        markers = assert_lowered_contract(compiled, captured)
        print("COMPILE_OK", f"E={e} K={k} lanes={lanes} aligned={align(e, lanes)}", markers)
    print("source_contract_ok")
