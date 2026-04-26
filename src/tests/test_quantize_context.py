"""Tests for QuantizeContext — written task-by-task, each task adds cases."""
import torch
import torch.nn as nn
import pytest

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.formats.int_formats import IntFormat


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _int8_scheme():
    return QuantScheme(
        format=IntFormat(bits=8),
        granularity=GranularitySpec(GranularityMode.PER_TENSOR),
    )


def _make_cfg(**roles):
    """Build OpQuantConfig; default all forward roles to int8 if not given."""
    s = _int8_scheme()
    defaults = dict(input=(s,), weight=(s,), output=(s,))
    defaults.update(roles)
    return OpQuantConfig(**defaults)


# ---------------------------------------------------------------------------
# Task 2 — _CtxState + _ctx_state
# ---------------------------------------------------------------------------

from src.context._state import _ctx_state, _CtxState, _EMPTY_CFG


def test_ctx_state_not_active_by_default():
    assert _ctx_state.get(None) is None


def test_ctx_state_set_and_reset():
    cfg = OpQuantConfig()
    state = _CtxState(cfg=cfg)
    tok = _ctx_state.set(state)
    assert _ctx_state.get().cfg is cfg
    _ctx_state.reset(tok)
    assert _ctx_state.get(None) is None


def test_ctx_state_resolve_default():
    cfg = _make_cfg()
    state = _CtxState(cfg=cfg)
    assert state.resolve("matmul") is cfg
    assert state.resolve("add") is cfg


def test_ctx_state_resolve_per_op_override():
    default_cfg = OpQuantConfig()
    matmul_cfg = _make_cfg()
    state = _CtxState(cfg=default_cfg, op_cfgs={"matmul": matmul_cfg})
    assert state.resolve("matmul") is matmul_cfg
    assert state.resolve("add") is default_cfg


# ---------------------------------------------------------------------------
# Task 2 — module stack
# ---------------------------------------------------------------------------

from src.context._stack import install_stack_hooks, remove_stack_hooks, get_layer_name


def test_get_layer_name_empty():
    assert get_layer_name() == ""


def test_stack_records_module_name_during_forward():
    captured = []

    class Probe(nn.Module):
        def forward(self, x):
            captured.append(get_layer_name())
            return x

    model = nn.Sequential(Probe())
    hooks = install_stack_hooks(model)
    model(torch.zeros(1))
    remove_stack_hooks(hooks)
    assert captured[0] == "0"


def test_stack_cleans_up_after_forward():
    model = nn.Linear(4, 4)
    hooks = install_stack_hooks(model)
    model(torch.zeros(2, 4))
    remove_stack_hooks(hooks)
    assert get_layer_name() == ""


# ---------------------------------------------------------------------------
# Task 3 — MatMulFunction.symbolic + BMMFunction.symbolic
# ---------------------------------------------------------------------------

def test_matmul_symbolic_method_exists():
    from src.ops.matmul import MatMulFunction
    assert hasattr(MatMulFunction, "symbolic")


def test_bmm_symbolic_method_exists():
    from src.ops.bmm import BMMFunction
    assert hasattr(BMMFunction, "symbolic")


# ---------------------------------------------------------------------------
# Task 4 — SIMD symbolic() methods
# ---------------------------------------------------------------------------

def test_simd_symbolic_methods_exist():
    from src.ops.elemwise import SIMDAdd, SIMDSub, SIMDMul, SIMDDiv, SIMDExp, SIMDLog
    for cls in (SIMDAdd, SIMDSub, SIMDMul, SIMDDiv, SIMDExp, SIMDLog):
        assert hasattr(cls, "symbolic"), f"{cls.__name__} missing symbolic()"


# ---------------------------------------------------------------------------
# Task 5 — _patches: each patched fn passes through without context
# ---------------------------------------------------------------------------

def test_patch_table_has_all_ops():
    from src.context._patches import _PATCH_TABLE
    expected = {
        ("torch", "matmul"), ("torch", "mm"), ("torch", "bmm"),
        ("torch", "add"), ("torch", "sub"), ("torch", "mul"),
        ("torch", "div"), ("torch", "exp"), ("torch", "log"),
        ("F", "linear"),
    }
    assert expected.issubset(set(_PATCH_TABLE.keys()))


def test_patched_matmul_passthrough_without_context():
    from src.context._patches import _patched_matmul
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    assert torch.equal(_patched_matmul(a, b), torch.matmul(a, b))


def test_patched_add_passthrough_without_context():
    from src.context._patches import _patched_add
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    assert torch.equal(_patched_add(a, b), torch.add(a, b))


def test_patched_matmul_quantizes_with_active_context():
    from src.context._patches import _patched_matmul
    cfg = _make_cfg()
    state = _CtxState(cfg=cfg)
    tok = _ctx_state.set(state)
    try:
        a, b = torch.randn(3, 4), torch.randn(4, 5)
        result = _patched_matmul(a, b)
        plain = torch.matmul(a, b)
        assert not torch.equal(result, plain)
    finally:
        _ctx_state.reset(tok)


def test_patched_F_linear_quantizes_with_active_context():
    from src.context._patches import _patched_F_linear
    import torch.nn.functional as F_orig
    cfg = _make_cfg()
    state = _CtxState(cfg=cfg)
    tok = _ctx_state.set(state)
    try:
        x = torch.randn(2, 8)
        w = torch.randn(4, 8)
        result = _patched_F_linear(x, w)
        plain = F_orig.linear(x, w)
        assert not torch.equal(result, plain)
    finally:
        _ctx_state.reset(tok)


def test_patched_add_with_scalar_passthrough():
    """Scalar second argument must not be routed through SIMDAdd."""
    from src.context._patches import _patched_add
    a = torch.randn(3, 4)
    state = _CtxState(cfg=_make_cfg())
    tok = _ctx_state.set(state)
    try:
        result = _patched_add(a, 1.0)  # scalar → passthrough
        assert torch.equal(result, a + 1.0)
    finally:
        _ctx_state.reset(tok)


# ---------------------------------------------------------------------------
# Task 6 — QuantizeContext class
# ---------------------------------------------------------------------------

def test_context_quantizes_torch_matmul():
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)  # just needs a module for hook installation
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    plain = torch.matmul(a, b)

    with QuantizeContext(model, cfg):
        result = torch.matmul(a, b)

    assert not torch.equal(result, plain)


def test_context_restores_ops_on_exit():
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(4, 5)

    with QuantizeContext(model, cfg):
        pass

    # After exit, unpatched torch.matmul must return float result again
    result = torch.matmul(a, b)
    expected = torch.matmul(a, b)
    assert torch.equal(result, expected)


def test_context_intercepts_nn_linear_forward():
    """nn.Linear.forward calls F.linear, which the context intercepts."""
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)

    plain = model(x)
    with QuantizeContext(model, cfg):
        quant = model(x)

    assert not torch.equal(plain, quant)


def test_context_intercepts_torch_add():
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    plain = torch.add(a, b)

    with QuantizeContext(model, cfg):
        result = torch.add(a, b)

    assert not torch.equal(result, plain)


def test_per_op_override_only_quantizes_specified_op():
    """Default cfg = no-quant; only matmul is overridden to int8."""
    from src.context.quantize_context import QuantizeContext
    default_cfg = OpQuantConfig()   # no quantization
    matmul_cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(4, 5)

    with QuantizeContext(model, default_cfg, op_cfgs={"matmul": matmul_cfg}):
        matmul_result = torch.matmul(a, b)
        # torch.add with default (empty) cfg → passthrough
        c, d = torch.randn(3, 5), torch.randn(3, 5)
        add_result = torch.add(c, d)

    assert not torch.equal(matmul_result, torch.matmul(a, b))
    assert torch.equal(add_result, torch.add(c, d))


def test_context_no_double_quantization():
    """F.linear inside LinearFunction.forward uses _F_linear, not the patch."""
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 8)

    with QuantizeContext(model, cfg):
        r1 = model(x)
        r2 = model(x)

    assert torch.equal(r1, r2)  # deterministic = no unintended extra quantization


def test_context_restores_ops_on_exception():
    """Patches must be removed even when the with-block raises."""
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    try:
        with QuantizeContext(model, cfg):
            raise RuntimeError("forced")
    except RuntimeError:
        pass
    result = torch.matmul(torch.randn(2, 3), torch.randn(3, 4))
    assert result.requires_grad is False  # plain float output, not a Function result


def test_nested_contexts_isolate_cfg():
    """Inner context cfg does not bleed into outer after inner exits."""
    from src.context.quantize_context import QuantizeContext
    outer_cfg = OpQuantConfig()        # no quantization
    inner_cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(4, 5)

    with QuantizeContext(model, outer_cfg):
        plain_inside_outer = torch.matmul(a, b)   # no quant (outer_cfg is empty)
        with QuantizeContext(model, inner_cfg):
            quant_inside_inner = torch.matmul(a, b)
        plain_after_inner = torch.matmul(a, b)    # back to outer_cfg (no quant)

    assert torch.equal(plain_inside_outer, torch.matmul(a, b))
    assert not torch.equal(quant_inside_inner, torch.matmul(a, b))
    assert torch.equal(plain_after_inner, torch.matmul(a, b))


def test_patched_mm_quantizes_with_active_context():
    """torch.mm should be intercepted and quantized like torch.matmul."""
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    plain = torch.mm(a, b)

    with QuantizeContext(model, cfg):
        result = torch.mm(a, b)

    assert not torch.equal(result, plain)


def test_patched_mul_scalar_first_passthrough():
    """torch.mul(scalar, tensor) must not crash when context is active."""
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)
    t = torch.randn(3, 4)

    with QuantizeContext(model, cfg):
        result = torch.mul(2.0, t)   # scalar-first

    assert torch.equal(result, 2.0 * t)


def test_context_is_not_active_outside_with_block():
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Linear(1, 1)

    with QuantizeContext(model, cfg):
        inside = torch.matmul(torch.randn(2, 3), torch.randn(3, 4))

    outside = torch.matmul(torch.randn(2, 3), torch.randn(3, 4))
    # outside should be a plain float result (not wrapped by Function)
    assert outside.requires_grad is False


# ---------------------------------------------------------------------------
# Task 7 — ONNX export via ctx.export_onnx
# ---------------------------------------------------------------------------

def test_export_onnx_nn_linear(tmp_path):
    """Linear model exports valid ONNX with QDQ nodes."""
    import onnx
    from src.context.quantize_context import QuantizeContext
    cfg = _make_cfg()
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    x = torch.randn(2, 8)

    with QuantizeContext(model, cfg) as ctx:
        ctx.export_onnx(x, str(tmp_path / "linear.onnx"))

    m = onnx.load(str(tmp_path / "linear.onnx"))
    onnx.checker.check_model(m)
    node_types = {n.op_type for n in m.graph.node}
    assert "QuantizeLinear" in node_types or "MxQuantize" in node_types


def test_export_onnx_inline_matmul(tmp_path):
    """Model with inline torch.matmul exports valid ONNX."""
    import onnx
    from src.context.quantize_context import QuantizeContext

    class SelfAttnShape(nn.Module):
        def forward(self, x):
            # x: (B, S, D) — self-attention score matrix
            return torch.matmul(x, x.transpose(-2, -1))

    cfg = _make_cfg()
    model = SelfAttnShape()
    x = torch.randn(1, 4, 8)

    with QuantizeContext(model, cfg) as ctx:
        ctx.export_onnx(x, str(tmp_path / "attn.onnx"))

    m = onnx.load(str(tmp_path / "attn.onnx"))
    onnx.checker.check_model(m)
    node_types = {n.op_type for n in m.graph.node}
    assert "MatMul" in node_types


def test_export_onnx_with_add(tmp_path):
    """Model with torch.add exports valid ONNX."""
    import onnx
    from src.context.quantize_context import QuantizeContext

    class Residual(nn.Module):
        def forward(self, x):
            return torch.add(x, x)

    cfg = _make_cfg()
    model = Residual()
    x = torch.randn(2, 8)

    with QuantizeContext(model, cfg) as ctx:
        ctx.export_onnx(x, str(tmp_path / "residual.onnx"))

    m = onnx.load(str(tmp_path / "residual.onnx"))
    onnx.checker.check_model(m)
    assert any(n.op_type == "Add" for n in m.graph.node)
