"""Manual alias inference uses completion edges on the accessing hardware pipe."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


def _call(name, *args):
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl." + name), list(args)))


def _load(gm, ub):
    return _call("ascend_copy_gm_to_ubuf", ub.access_ptr("w"), gm.access_ptr("r"), 0, 1, 256, 0, 0, 0, 0, 0, 0)


def _store(ub, gm):
    return _call("ascend_copy_ubuf_to_gm", gm.access_ptr("w"), ub.access_ptr("r"), 0, 1, 256, 0, 0, 0)


def _flags(pipe):
    return [_call("ascend_set_flag", pipe, 0), _call("ascend_wait_flag", pipe, 0)]


def _module(buffers, body, gm):
    return tvm.IRModule({"main": tirx.PrimFunc([gm.data], tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in buffers], *body]))})


def _may_alias(after, a, b):
    (contract,) = [node.node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.buffer_alias_map"]
    return b.data in contract[a.data]


@pytest.mark.parametrize("release", ["none", "acquire-only", "MTE3_MTE2", "PIPE_ALL"])
def test_dma_read_requires_a_completion_edge_before_reuse(release):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    between = []
    if release == "acquire-only":
        between = [_call("ascend_wait_flag", "MTE3_MTE2", 0)]
    elif release == "MTE3_MTE2":
        between = _flags(release)
    elif release == "PIPE_ALL":
        between = [_call("ascend_pipe_barrier", "PIPE_ALL")]
    before = _module(
        [a, b], [_load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm), *between, _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)], gm
    )
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release in ("MTE3_MTE2", "PIPE_ALL"))


@pytest.mark.parametrize("release_pipe", ["S_MTE2", "V_MTE2"])
def test_pad_value_operand_is_scalar_even_inside_vector_block(release_pipe):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    body = tirx.SeqStmt(
        [_load(gm, a), *_flags("MTE2_S"), tirx.Evaluate(T.ascend_set_copy_pad_value(a[0])), *_flags(release_pipe), _load(gm, b)]
    )
    before = _module([a, b], [tirx.SBlock([], [], [], "VECTOR", body)], gm)
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release_pipe == "S_MTE2")


@pytest.mark.parametrize("release_pipe", ["PIPE_MTE1", "PIPE_FIX"])
def test_cross_core_handshake_only_releases_its_local_pipe(release_pipe):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.l1")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.l1")
    l0a = tirx.decl_buffer((64,), "float32", name="l0a", scope="shared.l0a")
    # The Cube reader is MTE1. A FIX release cannot complete that read, even
    # if Vector returns a token that Cube acquires on MTE2 before filling b.
    cube_read = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _call("ascend_copy_gm_to_cbuf", a.access_ptr("w"), gm.access_ptr("r"), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
                *_flags("MTE2_MTE1"),
                _call("ascend_load_cbuf_to_ca", l0a.access_ptr("w"), a.access_ptr("r")),
                _call("ascend_cross_core_set_flag", 4, release_pipe, 0),
            ]
        ),
    )
    vector = tirx.SBlock(
        [],
        [],
        [],
        "VECTOR",
        tirx.SeqStmt(
            [
                _call("ascend_cross_core_wait_flag", 4, "PIPE_MTE3", 0),
                _call("ascend_cross_core_set_flag", 4, "PIPE_MTE3", 1),
            ]
        ),
    )
    cube_write = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _call("ascend_cross_core_wait_flag", 4, "PIPE_MTE2", 1),
                _call("ascend_copy_gm_to_cbuf", b.access_ptr("w"), gm.access_ptr("r"), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
            ]
        ),
    )
    before = _module([a, b, l0a], [cube_read, vector, cube_write], gm)
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release_pipe == "PIPE_MTE1")


# ---------------------------------------------------------------------------
# Task 3: widen scratch view/backing coherence and V -> MTE3 dependency
# ---------------------------------------------------------------------------


def _buffers(ir):
    found = []
    for node in nodes(ir, tirx.BufferLoad):
        found.append(node.buffer)
    for node in nodes(ir, tirx.BufferStore):
        found.append(node.buffer)
    for node in nodes(ir, tirx.BufferRegion):
        found.append(node.buffer)
    return found


def test_widen_scratch_alias_views():
    """The int32 widening view and the int64 scratch are one storage: same
    data Var, identical byte ranges (2K x int32 == K x int64), and the view is
    not a second allocation."""

    from testing.ascend.transform.test_expand_dtype_mismatched_copy import (
        _copy_module,
        _expand,
    )

    k = 8
    after = _expand(_copy_module(k=k, dst_base=0, dst_shape=16)[0])
    vf = next(b for b in nodes(after, tirx.SBlock) if b.name_hint == "SIMD_VF")
    src_buf = vf.reads[0].buffer
    mte = [
        call
        for call in nodes(after, tirx.Call)
        if getattr(call.op, "name", None) == "tl.tileop.ascend_copy"
    ][0]
    backing = mte.args[0].args[0].buffer
    assert str(backing.dtype) == "int64"

    # Exactly one int32 buffer shares the scratch storage: the widen view.
    views = [
        buf
        for buf in _buffers(after)
        if str(buf.dtype) == "int32" and buf.data.same_as(backing.data) and not buf.same_as(src_buf)
    ]
    assert views, "the widen must write through an int32 view of the scratch"
    view = views[0]
    assert int(view.shape[0]) == 2 * k and int(backing.shape[0]) == k
    assert int(view.shape[0]) * 4 == int(backing.shape[0]) * 8, (
        "view and backing must address the same byte range"
    )

    # The view shares the storage, it is not allocated again.
    allocs = [buf for block in nodes(after, tirx.SBlock) for buf in (block.alloc_buffers or [])]
    assert any(buf.same_as(backing) for buf in allocs), "the int64 scratch is the allocation"
    assert not any(buf.same_as(view) for buf in allocs), "the view must not be a second allocation"

    # The same-dtype MTE copy reads the whole scratch [0, K).
    assert int(mte.args[0].args[0].indices[0]) == 0
    assert int(mte.args[0].args[2]) == k


def _widen_frontend_kernel(k):
    @T.prim_func
    def main(A: T.Tensor((k,), "int32"), Out: T.Tensor((k,), "int64")):
        with T.Kernel(1):
            ub = T.alloc_shared((k,), "int32")
            T.copy(A, ub)
            T.copy(ub, Out)

    return main


def _assert_widen_dependency_chain(source, k):
    """Assert the full source -> widen -> MTE3 chain in emitted PTODSL.

    AST based: simple pointer assignments and constant address arithmetic
    (``pto.castptr(pto.const(N, ...))`` bases, ``pto.addptr`` element offsets,
    load/store element indices with loop variables at their lower bound) are
    resolved to byte addresses, so the checks bind to the *actual* buffers:

      (1) the UB target of the source GM copy is bound; the first real read of
          that window is found independently of any wait position, and the
          order ``source_copy < MTE2 set < MTE2 wait < first source read`` is
          asserted with a legal, matching event id pair;
      (2) the widen's scratch *output interval* is derived from its actual
          output stores -- the VMI vstores through the int32 view (K8/K64 pair
          halves) and the scalar int64-cast stores of the K<64 serial fallback
          (loop extents enumerated); pad copies are scalar int32 stores and are
          not output stores. The ``V -> MTE3`` pair must be a legal matched id
          pair ordered after the last output store, and the MTE3 copy after the
          wait must be a real int64 view into ``Out`` whose byte interval is
          *exactly* that scratch interval (one contiguous range, 8*K bytes).

    Extracted so the one-off mutation probes call the exact same checker."""

    import ast
    import itertools

    env = {}
    calls = []

    def attr_chain(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    def const_int(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    call_loops = {}

    def loop_domain(iter_node):
        """``(low, high, step)`` for ``range(...)`` with literal bounds, else None."""
        if (
            isinstance(iter_node, ast.Call)
            and attr_chain(iter_node.func) == "range"
            and 1 <= len(iter_node.args) <= 3
        ):
            bounds = [const_int(arg) for arg in iter_node.args]
            if len(bounds) == 1:
                return 0, bounds[0], 1
            if len(bounds) == 2:
                return bounds[0], bounds[1], 1
            return tuple(bounds)
        return None

    def visit(node, loops=()):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            env[node.targets[0].id] = node.value
        if isinstance(node, ast.Call):
            calls.append((node.lineno, node))
            call_loops[id(node)] = loops
        if isinstance(node, ast.For):
            domain = loop_domain(node.iter)
            if domain is not None and isinstance(node.target, ast.Name):
                loops = loops + ((node.target.id,) + domain,)
            else:
                loops = loops + (None,)  # unknown loop: output stores fail closed
        for child in ast.iter_child_nodes(node):
            visit(child, loops)

    visit(ast.parse(source))

    def dtype_name(node):
        return attr_chain(node).rsplit(".", 1)[-1]

    def pointee_bytes(castptr_call):
        if not (isinstance(castptr_call, ast.Call) and attr_chain(castptr_call.func) == "pto.castptr"):
            return None
        ptr = castptr_call.args[1]
        if isinstance(ptr, ast.Call) and attr_chain(ptr.func) == "pto.ptr":
            return {"si32": 4, "si64": 8, "ui8": 1}.get(dtype_name(ptr.args[0]))
        return None

    def ptr_elem_bytes(expr, depth=0):
        if depth > 6:
            return None
        if isinstance(expr, ast.Name):
            return ptr_elem_bytes(env[expr.id], depth + 1) if expr.id in env else None
        if isinstance(expr, ast.Call):
            if attr_chain(expr.func) == "pto.addptr":
                return ptr_elem_bytes(expr.args[0], depth + 1)
            return pointee_bytes(expr)
        return None

    def byte_addr(expr, depth=0):
        """Constant byte address of a pointer expression, or None."""
        if depth > 6:
            return None
        if isinstance(expr, ast.Name):
            return byte_addr(env[expr.id], depth + 1) if expr.id in env else None
        if isinstance(expr, ast.Call):
            name = attr_chain(expr.func)
            if name == "pto.castptr":
                raw = expr.args[0]
                if isinstance(raw, ast.Call) and attr_chain(raw.func) == "pto.const":
                    return const_int(raw.args[0])
                return byte_addr(raw, depth + 1)
            if name == "pto.addptr":
                base = byte_addr(expr.args[0], depth + 1)
                off = const_int(expr.args[1]) if len(expr.args) > 1 else None
                elem = ptr_elem_bytes(expr.args[0], depth + 1)
                if base is None or off is None or elem is None:
                    return None
                return base + off * elem
        return None

    def eval_int(node, values=None):
        """Evaluate a constant address/loop-index expression; loop variables not
        bound in ``values`` are taken at their lower bound (0)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if values and node.id in values:
                return values[node.id]
            return 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = eval_int(node.operand, values)
            return -inner if inner is not None else None
        if isinstance(node, ast.BinOp):
            left, right = eval_int(node.left, values), eval_int(node.right, values)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.FloorDiv, ast.Div)) and right != 0:
                return left // right
            return None
        if isinstance(node, ast.Call) and attr_chain(node.func) in ("T.FloorDiv", "T.floordiv"):
            left, right = eval_int(node.args[0], values), eval_int(node.args[1], values)
            if left is not None and right not in (None, 0):
                return left // right
        return None

    def event_id(call):
        for kw in call.keywords:
            if kw.arg == "event_id":
                return const_int(kw.value)
        return None

    def str_arg(call, index):
        node = call.args[index]
        return node.value if isinstance(node, ast.Constant) else None

    def flag_pairs(set_pipe, wait_pipe):
        sets = [
            (ln, c)
            for ln, c in calls
            if attr_chain(c.func) == "pto.set_flag"
            and str_arg(c, 0) == set_pipe
            and str_arg(c, 1) == wait_pipe
        ]
        waits = [
            (ln, c)
            for ln, c in calls
            if attr_chain(c.func) == "pto.wait_flag"
            and str_arg(c, 0) == set_pipe
            and str_arg(c, 1) == wait_pipe
        ]
        return sets, waits

    def assert_matched_pair(tag, sets, waits):
        assert sets and waits, f"the {tag} set/wait pair must exist"
        set_ln, set_call = sets[0]
        wait_ln, wait_call = waits[0]
        set_id, wait_id = event_id(set_call), event_id(wait_call)
        assert set_id is not None and wait_id is not None, (
            f"the {tag} set/wait must carry explicit event ids"
        )
        assert set_id == wait_id, (
            f"the {tag} set/wait must share one event id; got {set_id} and {wait_id}"
        )
        assert set_ln < wait_ln, f"the {tag} set must precede its wait"
        return set_ln, wait_ln

    # (0) bind the source GM copy and its UB target window.
    src_copies = [
        (ln, c)
        for ln, c in calls
        if attr_chain(c.func) == "pto.mte_gm_ub"
        and isinstance(c.args[0], ast.Name)
        and c.args[0].id == "A"
    ]
    assert src_copies, "the source GM->UB copy must exist"
    src_ln, src_call = src_copies[0]
    ub_target = byte_addr(src_call.args[1])
    ub_bytes = const_int(src_call.args[3]) if len(src_call.args) > 3 else None
    assert ub_target is not None and ub_bytes, (
        "the source copy's UB target and length must resolve"
    )
    window = range(ub_target, ub_target + ub_bytes)

    # (1) first real read of the source window, independent of wait position.
    ub_reads = []
    for ln, call in calls:
        name = attr_chain(call.func)
        if name not in ("pto.load", "pto.vmi.vload"):
            continue
        base = byte_addr(call.args[0])
        elem = ptr_elem_bytes(call.args[0])
        index = eval_int(call.args[1]) if len(call.args) > 1 else None
        if base is None or elem is None or index is None:
            continue
        addr = base + elem * index
        if addr in window:
            ub_reads.append((ln, addr))
    assert ub_reads, "no read of the source UB window was found"
    first_read_ln, first_read_addr = ub_reads[0]

    m2v_sets, m2v_waits = flag_pairs("MTE2", "V")
    m2v_set_ln, m2v_wait_ln = assert_matched_pair("MTE2->V", m2v_sets, m2v_waits)
    assert src_ln < m2v_set_ln < m2v_wait_ln < first_read_ln, (
        f"order must be source_copy < MTE2 set < MTE2 wait < first source read; "
        f"got {src_ln}, {m2v_set_ln}, {m2v_wait_ln}, {first_read_ln}"
    )

    # (2) the widen's scratch output interval, bound to its actual output
    # stores. Pad copies are scalar si32 stores and are not output stores; the
    # K64 pair writes two adjacent halves through the int32 view and must merge
    # into one contiguous interval; the K<64 serial fallback stores int64 casts
    # one element per loop iteration (extents enumerated below).
    def mask_elements(node):
        if isinstance(node, ast.Name):
            node = env.get(node.id)
        if isinstance(node, ast.Call) and attr_chain(node.func) == "pto.vmi.create_mask":
            return const_int(node.args[0])
        return const_int(node)

    def output_store_interval(call):
        """Byte interval ``[start, end)`` written by one widen output store."""
        name = attr_chain(call.func)
        dst = call.args[1]
        elem = ptr_elem_bytes(dst)
        base = byte_addr(dst)
        if elem is None or base is None:
            return None
        if name == "pto.vmi.vstore":
            if elem != 4 or len(call.args) <= 3:  # the int32 view of the scratch
                return None
            off = eval_int(call.args[2])
            count = mask_elements(call.args[3])
            if off is None or count is None:
                return None
            start = base + off * elem
            return (start, start + count * elem)
        # serial fallback: pto.store(pto.cast(..., si64), <si64 view>, index)
        value = call.args[0]
        if not (
            isinstance(value, ast.Call)
            and attr_chain(value.func) == "pto.cast"
            and value.args
            and dtype_name(value.args[-1]) == "si64"
        ):
            return None
        stack = call_loops.get(id(call), ())
        if any(item is None for item in stack):
            return None
        domains = []
        for var, low, high, step in stack:
            if None in (low, high, step) or step <= 0 or high - low > 4096:
                return None
            domains.append((var, range(low, high, step)))
        index = call.args[2] if len(call.args) > 2 else None
        addrs = set()
        for combo in itertools.product(*[vals for _, vals in domains]):
            values = dict(zip([var for var, _ in domains], combo))
            off = eval_int(index, values)
            if off is None:
                return None
            addrs.add(base + off * elem)
        if not addrs:
            return None
        if max(addrs) - min(addrs) != (len(addrs) - 1) * elem:
            return None  # not a contiguous element range
        return (min(addrs), max(addrs) + elem)

    def merge_intervals(intervals):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                if start < merged[-1][1]:
                    return None  # overlapping output stores
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    output_stores = []
    for ln, call in calls:
        if ln < first_read_ln or attr_chain(call.func) not in ("pto.store", "pto.vmi.vstore"):
            continue
        interval = output_store_interval(call)
        if interval is not None:
            output_stores.append((ln, interval))
    assert output_stores, (
        "the widen's output stores (VMI vstore / int64-cast store) must exist"
    )
    merged = merge_intervals([interval for _, interval in output_stores])
    assert merged is not None and len(merged) == 1, (
        f"the widen output stores must form one contiguous scratch interval; got "
        f"{sorted(interval for _, interval in output_stores)}"
    )
    scratch_start, scratch_end = merged[0]
    last_store_ln = max(ln for ln, _ in output_stores)

    v2m3_sets, v2m3_waits = flag_pairs("V", "MTE3")
    v2m3_set_ln, v2m3_wait_ln = assert_matched_pair("V->MTE3", v2m3_sets, v2m3_waits)
    assert last_store_ln < v2m3_set_ln < v2m3_wait_ln, (
        f"the V->MTE3 pair must follow the last scratch store; got "
        f"{last_store_ln}, {v2m3_set_ln}, {v2m3_wait_ln}"
    )

    mte3 = [
        (ln, c)
        for ln, c in calls
        if attr_chain(c.func) == "pto.mte_ub_gm"
        and isinstance(c.args[1], ast.Name)
        and c.args[1].id == "Out"
        and ln > v2m3_wait_ln
    ]
    assert mte3, "the scratch MTE3 copy must follow the V -> MTE3 wait"
    mte3_ln, mte3_call = mte3[0]
    assert ptr_elem_bytes(mte3_call.args[0]) == 8, (
        "the MTE3 copy source must be a real int64 view"
    )
    src_byte = byte_addr(mte3_call.args[0])
    length = const_int(mte3_call.args[2]) if len(mte3_call.args) > 2 else None
    assert src_byte is not None and length is not None, (
        "the MTE3 copy source byte base and length must resolve"
    )
    assert length == 8 * k, f"the MTE3 copy must transfer exactly 8*K bytes; got {length}"
    assert (src_byte, src_byte + length) == (scratch_start, scratch_end), (
        f"the MTE3 copy must publish exactly the widen scratch interval "
        f"[{scratch_start}, {scratch_end}); got [{src_byte}, {src_byte + length})"
    )


@pytest.mark.parametrize("k", [8, 64, 6], ids=["k8-pad", "k64", "k6-serial"])
def test_widen_cast_mte_dependency(k):
    """The full source -> widen -> MTE3 chain in the emitted PTODSL:

      mte_gm_ub(source) -> (MTE2,V) set/wait with matching event ids ->
      first widen source read -> widen scratch writes ->
      (V,MTE3) set/wait with matching event ids ->
      the MTE3 copy that publishes exactly that scratch region (8*K bytes).

    Event ids are compared, not just string presence."""

    import tilelang

    compiled = tilelang.compile(_widen_frontend_kernel(k), target="pto")
    _assert_widen_dependency_chain(compiled.get_kernel_source(), k)
