"""
Tests for ObservableMixin + QuantEvent — P3.1-b.

Key invariants:
- No observers → _emit returns immediately (zero overhead)
- With observers → QuantEvent is dispatched with correct fields
- QuantEvent is frozen (immutable)
- ObservableMixin._observers defaults to empty list
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


# ---------------------------------------------------------------------------
# ObservableMixin — no-op path
# ---------------------------------------------------------------------------

class _DummyOperator(ObservableMixin):
    """Minimal class mixing in ObservableMixin for testing."""
    pass


def test_mixin_default_observers_empty():
    op = _DummyOperator()
    assert op._observers == []


def test_mixin_emit_no_op_without_observers():
    """_emit must return immediately when no observers attached."""
    op = _DummyOperator()
    x = torch.randn(3, 4)
    s = _s()
    # Should not raise or do anything
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
    q = x + 1  # still in grad graph

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
# QuantEvent — frozen dataclass
# ---------------------------------------------------------------------------

def test_quant_event_is_frozen():
    evt = QuantEvent(
        layer_name="test", role="input", pipeline_index=0, stage="input_pre_quant",
        fp32_tensor=torch.randn(2), quant_tensor=torch.randn(2), scheme=_s(),
    )
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
