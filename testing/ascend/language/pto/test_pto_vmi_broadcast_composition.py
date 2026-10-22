"""Source-level coverage for explicit VMI broadcast-load compositions."""

import ast

import pytest

import tilelang.ascend.language as T
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower


@pytest.mark.pto
@pytest.mark.parametrize("group,stride", [(None, None), (2, 1), (4, 3)])
def test_pto_codegen_preserves_vload_vbrc_composition(group, stride):
    load_size = 1 if group is None else group

    @T.prim_func
    def func(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                loaded = T.vmi.vload(a_ub[4], size=load_size, group=group, stride=stride)
                value = T.vmi.vbrc(loaded, size=64, group=group)
                mask = T.vmi.create_mask(64, size=64)
                T.vmi.vstore(value, b_ub[0], mask)
            T.copy(b_ub, B)

    target = normalize_pto_target("pto")
    with target:
        source = _lower(func, target=target).kernel_source
    compile(source, "<pto-broadcast-load>", "exec")
    load_line = next(line for line in source.splitlines() if "pto.vmi.vload(" in line)
    broadcast_line = next(line for line in source.splitlines() if "pto.vmi.vbrc(" in line)
    load = ast.parse(load_line.strip()).body[0]
    broadcast = ast.parse(broadcast_line.strip()).body[0]
    assert isinstance(load, ast.Assign)
    assert isinstance(broadcast, ast.Assign)
    assert broadcast.value.args[0].id == load.targets[0].id
    assert ast.literal_eval(load.value.args[1]) == 0
    load_options = {kw.arg: ast.literal_eval(kw.value) for kw in load.value.keywords}
    broadcast_options = {kw.arg: ast.literal_eval(kw.value) for kw in broadcast.value.keywords}
    assert "dist_mode" not in source
    if group is None:
        assert load_options == {"size": 1}
        assert broadcast_options == {"size": 64}
    else:
        assert load_options == {"size": group, "group": group, "stride": stride}
        assert broadcast_options == {"size": 64, "group": group}

