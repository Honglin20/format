"""
Tests for ObservableMixin + QuantEvent — P3.1-b.

Key invariants:
- No observers → _emit returns immediately (zero overhead)
- With observers → QuantEvent is dispatched with correct fields
- QuantEvent is frozen (immutable) and validates all fields
- ObservableMixin._observers is instance-level (not shared across instances)
"""
import pytest
import torch

from src.analysis.mixin import ObservableMixin
from src.analysis.events import QuantEvent
from src.analysis.observer import ObserverBase, SliceAwareObserver
from src.analysis.slicing import SliceKey
from src.scheme.quant_scheme import QuantScheme


def _s(fmt="fp8_e4m3"):
    return QuantScheme(format=fmt)


def _valid_event_kwargs(**overrides):
    """Minimal valid QuantEvent kwargs for testing."""
    kw = dict(
        layer_name="test", role="input", pipeline_index=0, stage="input_pre_quant",
        fp32_tensor=torch.randn(2), quant_tensor=torch.randn(2), scheme=_s(),
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# ObservableMixin — no-op path
# ---------------------------------------------------------------------------

class _DummyOperator(ObservableMixin):
    """Minimal class mixing in ObservableMixin for testing."""
    pass


def test_mixin_default_observers_empty():
    op = _DummyOperator()
    assert op._observers == []


def test_mixin_observers_not_shared_across_instances():
    """C1 fix: _observers must be per-instance, not class-level."""
    op1 = _DummyOperator()
    op2 = _DummyOperator()
    assert op1._observers is not op2._observers
    op1._observers = ["x"]
    assert op2._observers == []


def test_mixin_emit_no_op_without_observers():
    """_emit must return immediately when no observers attached."""
    op = _DummyOperator()
    x = torch.randn(3, 4)
    s = _s()
    op._emit("input", 0, "input_pre_quant", fp32=x, quant=x, scheme=s)


def test_mixin_emit_dispatches_to_observers():
    """When observers are present, _emit creates QuantEvent and dispatches."""
    received = []

    class _Collector(ObserverBase):
        def on_event(self, event):
            received.append(event)

    op = _DummyOperator()
    op._analysis_name = "test.linear"
    op._observers = [_Collector()]

    x = torch.randn(3, 4)
    s = _s()
    op._emit("input", 0, "input_pre_quant", fp32=x, quant=x, scheme=s)

    assert len(received) == 1
    evt = received[0]
    assert evt.layer_name == "test.linear"
    assert evt.role == "input"
    assert evt.pipeline_index == 0
    assert evt.stage == "input_pre_quant"
    assert evt.scheme == s
    assert evt.group_map is None


def test_mixin_emit_detaches_tensors():
    """QuantEvent tensors must be detached (no grad graph)."""
    x = torch.randn(3, 4, requires_grad=True)
    q = x + 1

    received = []
    class _Collector(ObserverBase):
        def on_event(self, event):
            received.append(event)

    op = _DummyOperator()
    op._observers = [_Collector()]
    op._emit("input", 0, "input_pre_quant", fp32=x, quant=q, scheme=_s())

    evt = received[0]
    assert not evt.fp32_tensor.requires_grad
    assert not evt.quant_tensor.requires_grad


def test_mixin_emit_with_group_map():
    """group_map is detached and passed through."""
    received = []
    class _Collector(ObserverBase):
        def on_event(self, event):
            received.append(event)

    op = _DummyOperator()
    op._observers = [_Collector()]

    x = torch.randn(4)
    gm = torch.tensor([0, 0, 1, 1])
    op._emit("input", 0, "input_pre_quant", fp32=x, quant=x, scheme=_s(), group_map=gm)

    assert received[0].group_map is not None
    assert torch.equal(received[0].group_map, gm)


def test_mixin_default_layer_name_is_class_name():
    """Without _analysis_name, layer_name falls back to class name."""
    received = []
    class _Collector(ObserverBase):
        def on_event(self, event):
            received.append(event)

    op = _DummyOperator()
    op._observers = [_Collector()]
    op._emit("weight", 0, "weight_pre_quant", fp32=torch.randn(2,3), quant=torch.randn(2,3), scheme=_s())

    assert received[0].layer_name == "_DummyOperator"


def test_mixin_dispatches_to_multiple_observers():
    """All observers in the list receive the event."""
    count = [0, 0]
    class _Obs1(ObserverBase):
        def on_event(self, event):
            count[0] += 1
    class _Obs2(ObserverBase):
        def on_event(self, event):
            count[1] += 1

    op = _DummyOperator()
    op._observers = [_Obs1(), _Obs2()]
    op._emit("output", 0, "output_post_quant", fp32=torch.randn(2), quant=torch.randn(2), scheme=_s())

    assert count == [1, 1]


# ---------------------------------------------------------------------------
# QuantEvent — frozen dataclass + validation (C2)
# ---------------------------------------------------------------------------

def test_quant_event_is_frozen():
    evt = QuantEvent(**_valid_event_kwargs())
    with pytest.raises(AttributeError):
        evt.role = "output"


def test_quant_event_construction():
    s = _s()
    f = torch.randn(2)
    q = torch.randn(2)
    evt = QuantEvent(
        layer_name="l", role="weight", pipeline_index=1, stage="weight_pre_quant",
        fp32_tensor=f, quant_tensor=q, scheme=s,
    )
    assert evt.layer_name == "l"
    assert evt.role == "weight"
    assert evt.pipeline_index == 1
    assert evt.stage == "weight_pre_quant"
    assert evt.scheme == s
    assert evt.group_map is None


# --- QuantEvent validation: layer_name ---
def test_quant_event_empty_layer_name_raises():
    with pytest.raises(TypeError, match="layer_name must be a non-empty str"):
        QuantEvent(**_valid_event_kwargs(layer_name=""))


def test_quant_event_non_str_layer_name_raises():
    with pytest.raises(TypeError, match="layer_name must be a non-empty str"):
        QuantEvent(**_valid_event_kwargs(layer_name=123))


# --- QuantEvent validation: role ---
def test_quant_event_empty_role_raises():
    with pytest.raises(TypeError, match="role must be a non-empty str"):
        QuantEvent(**_valid_event_kwargs(role=""))


def test_quant_event_non_str_role_raises():
    with pytest.raises(TypeError, match="role must be a non-empty str"):
        QuantEvent(**_valid_event_kwargs(role=42))


# --- QuantEvent validation: pipeline_index ---
def test_quant_event_negative_pipeline_index_raises():
    with pytest.raises(ValueError, match="pipeline_index must be a non-negative int"):
        QuantEvent(**_valid_event_kwargs(pipeline_index=-1))


# --- QuantEvent validation: stage ---
def test_quant_event_empty_stage_raises():
    with pytest.raises(TypeError, match="stage must be a non-empty str"):
        QuantEvent(**_valid_event_kwargs(stage=""))


# --- QuantEvent validation: fp32_tensor ---
def test_quant_event_non_tensor_fp32_raises():
    with pytest.raises(TypeError, match="fp32_tensor must be a Tensor"):
        QuantEvent(**_valid_event_kwargs(fp32_tensor=[1, 2, 3]))


# --- QuantEvent validation: quant_tensor ---
def test_quant_event_non_tensor_quant_raises():
    with pytest.raises(TypeError, match="quant_tensor must be a Tensor"):
        QuantEvent(**_valid_event_kwargs(quant_tensor=[1, 2, 3]))


# --- QuantEvent validation: scheme ---
def test_quant_event_non_scheme_raises():
    with pytest.raises(TypeError, match="scheme must be a QuantScheme"):
        QuantEvent(**_valid_event_kwargs(scheme="fp8_e4m3"))


# --- QuantEvent validation: group_map ---
def test_quant_event_invalid_group_map_raises():
    with pytest.raises(TypeError, match="group_map must be a Tensor or None"):
        QuantEvent(**_valid_event_kwargs(group_map="bad"))


# ---------------------------------------------------------------------------
# emit_fn integration tests — C1: wired into operator forward/backward
# ---------------------------------------------------------------------------

from src.scheme.op_config import OpQuantConfig


def _qcfg():
    """One-scheme-per-role OpQuantConfig for emit_fn integration tests."""
    s = _s("fp8_e4m3")
    return OpQuantConfig(
        input=s, weight=s, bias=s, output=s,
        grad_output=s, grad_weight=s, grad_input=s,
    )


class _SpyObserver(ObserverBase):
    def __init__(self):
        self.events = []
    def on_event(self, event):
        self.events.append(event)


def test_emit_not_called_without_observers():
    """No observers attached → emit_fn is None → operator runs normally."""
    from src.ops.linear import QuantizedLinear
    mod = QuantizedLinear(4, 8, cfg=_qcfg())
    x = torch.randn(2, 4)
    out = mod(x)
    assert out.shape == (2, 8)


def test_emit_called_with_observer():
    """Attaching a spy observer → events are emitted during forward."""
    from src.ops.linear import QuantizedLinear
    mod = QuantizedLinear(4, 8, cfg=_qcfg())
    spy = _SpyObserver()
    mod._observers = [spy]
    mod(torch.randn(2, 4))
    assert len(spy.events) > 0


def test_emit_forward_roles_present():
    """Forward pass emits 'input', 'weight', 'output' roles."""
    from src.ops.linear import QuantizedLinear
    mod = QuantizedLinear(4, 8, cfg=_qcfg())
    spy = _SpyObserver()
    mod._observers = [spy]
    mod(torch.randn(2, 4))
    roles = {e.role for e in spy.events}
    assert "input" in roles
    assert "weight" in roles
    assert "output" in roles


def test_emit_backward_roles_present():
    """Backward pass emits 'grad_output', 'grad_weight', 'grad_input' roles."""
    from src.ops.linear import QuantizedLinear
    mod = QuantizedLinear(4, 8, cfg=_qcfg())
    spy = _SpyObserver()
    mod._observers = [spy]
    x = torch.randn(2, 4, requires_grad=True)
    out = mod(x)
    spy.events.clear()
    out.sum().backward()
    roles = {e.role for e in spy.events}
    assert "grad_output" in roles
    assert "grad_weight" in roles
    assert "grad_input" in roles
