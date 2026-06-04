# Agent API: Intervention Evaluation (Agent 7)

> Available to: `intervention_evaluator` agent

## Single-Layer FP32 Restore

Skip quantization for specific layers to measure their individual contribution to error.

```python
import copy
from src.session import Session, QuantConfig

model_copy = copy.deepcopy(model)
config = QuantConfig(name="W4A4", w_format="int4", a_format="int4", ...)

# Override: skip quantization for a layer (use FP32)
overrides = {"target_layer": None}
sess = Session(model_copy, config)
result = sess.run(calib_data, eval_data=eval_data, eval_fn=eval_fn, overrides=overrides)
```

## Bit-Width Boost

Increase bit-width for specific layers using `OpQuantConfig`.

```python
from src.scheme.op_config import OpQuantConfig

# Use int8 weight for a specific layer in a W4A4 config
override_cfg = OpQuantConfig(weight_scheme=None)  # None = FP32 for that role
overrides = {"target_layer": override_cfg}
sess = Session(copy.deepcopy(model), config)
result = sess.run(calib_data, eval_data=eval_data, eval_fn=eval_fn, overrides=overrides)
```

## Transform Testing

Run configs with SmoothQuant or Hadamard to test transform effect.

```python
config_sq = QuantConfig(
    name="W4A4+SQ", w_format="int4", a_format="int4",
    w_granularity="per_block", a_granularity="per_block",
    w_block_size=16, a_block_size=16,
    transform="smoothquant",
)
sess = Session(copy.deepcopy(model), config_sq)
result = sess.run(calib_data, eval_data=eval_data, eval_fn=eval_fn)
```

## Combined Strategy Pattern

Test multiple interventions simultaneously:

```python
# Top-3 layers int8 + smoothquant
overrides = {layer: int8_cfg for layer in top_3_layers}
config_sq = QuantConfig(..., transform="smoothquant")
sess = Session(copy.deepcopy(model), config_sq)
result = sess.run(calib_data, eval_data=eval_data, eval_fn=eval_fn, overrides=overrides)
```
