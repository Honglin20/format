# quantize_nonlinear=True 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `quantize_nonlinear=True` 时，非线性算子（norm/activation/softmax/pooling）的 operand 入口施加 storage → per_block 两级量化，与 matmul 对齐。中间 vec_ops 和 backward 保持 storage-only。

**Architecture:** 在 `_model.py` 新增 `_nonlinear_true_cfg()` 辅助函数，保留原始 cfg 的 per_block compute 字段同时填充 storage 级 backward 字段。Norm 架构已就绪（cfg 与 inner_scheme 分离），只需改 cfg 构造。Activation/Softmax/Pooling 目前 `cfg.input` 同时充当 entry quant 和 inner_scheme，需要分离——在 `forward()` 中增加显式 entry quant 步骤，inner_scheme 固定为 storage。

**Tech Stack:** Python, PyTorch, pytest

---

## Task 1: 新增 `_nonlinear_true_cfg` 辅助函数 + 单元测试

**Files:**
- Modify: `src/session/_model.py`
- Modify: `src/tests/test_session_unit.py`

### Step 1: 写 `_nonlinear_true_cfg` 的单元测试

在 `TestNonMatmulCfg` 类中新增测试（test_session_unit.py:1480 附近），覆盖三种场景：

```python
# ---- _nonlinear_true_cfg ----

def test_nonlinear_true_cfg_with_storage_and_per_block(self):
    """storage + per_block compute: keeps compute fields, populates backward from storage."""
    from src.session._model import _nonlinear_true_cfg

    storage = self._make_bf16_storage()
    per_block = self._make_fp8_per_block()
    cfg = OpQuantConfig(input=per_block, weight=per_block, bias=per_block, storage=storage)

    result = _nonlinear_true_cfg(cfg)

    # Forward compute fields preserved
    assert result.storage is storage
    assert result.input is per_block       # NOT stripped
    assert result.weight is per_block      # NOT stripped
    assert result.bias is per_block        # NOT stripped
    # Backward fields populated from storage
    assert result.grad_output is storage
    assert result.grad_input is storage
    assert result.grad_weight is storage
    assert result.grad_bias is storage
    assert result.is_training is True

def test_nonlinear_true_cfg_mx_per_block_no_storage(self):
    """MX per_block compute, no storage → keeps compute, backward stays None."""
    from src.session._model import _nonlinear_true_cfg

    per_block = self._make_fp8_per_block()
    cfg = OpQuantConfig(input=per_block, weight=per_block, storage=None)

    result = _nonlinear_true_cfg(cfg)

    assert result.storage is None
    assert result.input is per_block        # kept
    assert result.weight is per_block       # kept
    assert result.grad_output is None       # no storage → no backward
    assert result.is_training is False

def test_nonlinear_true_cfg_compat_per_tensor(self):
    """Compat-style per_tensor → pass through unchanged (same as _non_matmul_cfg)."""
    from src.session._model import _nonlinear_true_cfg

    per_tensor = self._make_per_tensor_elemwise()
    cfg = OpQuantConfig(input=per_tensor, storage=None)

    result = _nonlinear_true_cfg(cfg)

    assert result is cfg  # pass through unchanged

def test_nonlinear_true_cfg_preserves_explicit_backward(self):
    """When cfg has explicit backward fields, they take precedence over storage fallback."""
    from src.session._model import _nonlinear_true_cfg
    from src.scheme.quant_scheme import QuantScheme
    from src.scheme.granularity import GranularitySpec
    from src.formats.bf16_fp16 import BFloat16Format

    storage = self._make_bf16_storage()
    per_block = self._make_fp8_per_block()
    # explicit backward scheme different from storage
    explicit_bw = QuantScheme(
        format=BFloat16Format(), granularity=GranularitySpec.per_tensor()
    )
    cfg = OpQuantConfig(
        input=per_block, weight=per_block,
        storage=storage,
        grad_input=explicit_bw,   # explicit takes precedence
    )

    result = _nonlinear_true_cfg(cfg)

    assert result.grad_input is explicit_bw  # explicit preserved, not storage
    assert result.grad_output is storage     # fallback to storage

def test_nonlinear_true_cfg_empty_cfg(self):
    """Empty cfg → returns empty cfg."""
    from src.session._model import _nonlinear_true_cfg

    cfg = OpQuantConfig()
    result = _nonlinear_true_cfg(cfg)

    assert result.storage is None
    assert result.input is None
    assert result.weight is None
    assert result.is_training is False
```

### Step 2: 运行测试确认 FAIL

```bash
pytest src/tests/test_session_unit.py::TestNonMatmulCfg::test_nonlinear_true_cfg_with_storage_and_per_block -v
# Expected: ImportError or NameError (function not defined)
```

### Step 3: 实现 `_nonlinear_true_cfg`

在 `src/session/_model.py` 的 `_activation_cfg` 函数之后（约 line 99）添加：

```python
def _nonlinear_true_cfg(cfg: OpQuantConfig) -> OpQuantConfig:
    """Derive an OpQuantConfig that keeps per_block compute for operand entry.

    Used when ``quantize_nonlinear=True`` — norm/activation/pool operands
    receive the same storage → per_block two-level quantization as matmul ops,
    while backward fields stay storage-only.

    Three cases:
    - Two-level model with storage: keep storage + input/weight/bias per_block,
      populate backward from storage.
    - MX with bfloat=0 (storage=None, input is per_block MX compute):
      keep per_block compute, backward stays None.
    - Compat-style config where input carries per_tensor elemwise scheme:
      pass through unchanged (no separate compute quant to add).
    """
    if cfg.storage is not None:
        return OpQuantConfig(
            storage=cfg.storage,
            input=cfg.input,              # per_block compute kept
            weight=cfg.weight,            # per_block compute kept
            bias=cfg.bias,                # per_block compute kept
            grad_output=cfg.grad_output or cfg.storage,
            grad_input=cfg.grad_input or cfg.storage,
            grad_weight=cfg.grad_weight or cfg.storage,
            grad_bias=cfg.grad_bias or cfg.storage,
        )
    # No storage: either compat-style (input is per_tensor elemwise) or MX bfloat=0
    if _is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight):
        # MX bfloat=0: keep compute, backward stays None
        return OpQuantConfig(
            input=cfg.input,
            weight=cfg.weight,
            bias=cfg.bias,
        )
    return cfg  # compat-style: input/weight carry per_tensor elemwise schemes
```

### Step 4: 运行测试确认 PASS

```bash
pytest src/tests/test_session_unit.py::TestNonMatmulCfg -v
# Expected: all 14 tests PASS (9 existing + 5 new)
```

### Step 5: Commit

```bash
git add src/session/_model.py src/tests/test_session_unit.py
git commit -m "feat(session): add _nonlinear_true_cfg helper for quantize_nonlinear=True"
```

---

## Task 2: `_replace_module` 传递 `quantize_nonlinear` + norm `_make_*` 使用新 cfg

**Files:**
- Modify: `src/session/_model.py`

### Step 1: 修改 `_replace_module` 签名和调用

将 `quantize_nonlinear` 参数传递给所有 `_make_*` 函数：

```python
def _replace_module(
    module: nn.Module,
    cfg: Union[OpQuantConfig, Dict[str, OpQuantConfig]],
    name: str,
    *,
    quantize_nonlinear: bool = True,
):
    # ... existing code ...
    mod = make_fn(module, resolved_cfg, name, quantize_nonlinear=quantize_nonlinear)
    # ... rest unchanged ...
```

### Step 2: 修改 norm `_make_*` 函数

`_make_bn`, `_make_ln`, `_make_gn`, `_make_rms_norm` — 每个接受 `quantize_nonlinear` 参数，在 `True` 时使用 `_nonlinear_true_cfg(cfg)` 代替 `_non_matmul_cfg(cfg)`：

```python
def _make_bn(orig, cfg, name, bn_cls, quantize_nonlinear=False):
    from src.ops.norm import QuantizedBatchNorm2d  # (existing import context)
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = bn_cls(
        num_features=orig.num_features, eps=orig.eps,
        momentum=orig.momentum, affine=orig.affine,
        track_running_stats=orig.track_running_stats,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg),
        quantize_backprop=norm_cfg.is_training,
        name=name,
    )
    _copy_bn_state(orig, mod)
    return mod


def _make_ln(orig: nn.LayerNorm, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.norm import QuantizedLayerNorm
    normalized_shape = orig.normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = QuantizedLayerNorm(
        normalized_shape=list(normalized_shape), eps=orig.eps,
        elementwise_affine=orig.elementwise_affine,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg),
        quantize_backprop=norm_cfg.is_training,
        name=name,
    )
    if orig.elementwise_affine:
        mod.weight.data = orig.weight.data.clone()
        mod.bias.data = orig.bias.data.clone()
    return mod


def _make_gn(orig: nn.GroupNorm, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.norm import QuantizedGroupNorm
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = QuantizedGroupNorm(
        num_groups=orig.num_groups, num_channels=orig.num_channels,
        eps=orig.eps, affine=orig.affine,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg),
        quantize_backprop=norm_cfg.is_training,
        name=name,
    )
    if orig.affine:
        mod.weight.data = orig.weight.data.clone()
        mod.bias.data = orig.bias.data.clone()
    return mod


def _make_rms_norm(orig, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.norm import QuantizedRMSNorm
    normalized_shape = orig.normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    norm_cfg = _nonlinear_true_cfg(cfg) if quantize_nonlinear else _non_matmul_cfg(cfg)
    mod = QuantizedRMSNorm(
        normalized_shape=list(normalized_shape), eps=orig.eps,
        elementwise_affine=orig.elementwise_affine,
        cfg=norm_cfg,
        inner_scheme=_norm_inner_scheme(cfg),
        quantize_backprop=norm_cfg.is_training,
        name=name,
    )
    if orig.elementwise_affine:
        mod.weight.data = orig.weight.data.clone()
    return mod
```

### Step 3: 修改 matmul `_make_*` 函数签名（接收但不使用）

`_make_linear`, `_make_conv`, `_make_conv_transpose` 接受 `quantize_nonlinear` 但不改变行为（matmul 不受影响）：

```python
def _make_linear(orig: nn.Linear, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    # ... unchanged ...

def _make_conv(orig, cfg, name, conv_cls, quantize_nonlinear=False):
    # ... unchanged ...
```

### Step 4: 更新 `_MODULE_MAPPING` 中的 lambda 签名

```python
_MODULE_MAPPING = {
    nn.Linear: lambda orig, cfg, name, **kw: _make_linear(orig, cfg, name, **kw),
    nn.Conv1d: lambda orig, cfg, name, **kw: _make_conv(orig, cfg, name, QuantizedConv1d, **kw),
    # ... all others use **kw to forward quantize_nonlinear
    nn.BatchNorm1d: lambda orig, cfg, name, **kw: _make_bn(orig, cfg, name, QuantizedBatchNorm1d, **kw),
    nn.BatchNorm2d: lambda orig, cfg, name, **kw: _make_bn(orig, cfg, name, QuantizedBatchNorm2d, **kw),
    nn.BatchNorm3d: lambda orig, cfg, name, **kw: _make_bn(orig, cfg, name, QuantizedBatchNorm3d, **kw),
    nn.LayerNorm: lambda orig, cfg, name, **kw: _make_ln(orig, cfg, name, **kw),
    nn.GroupNorm: lambda orig, cfg, name, **kw: _make_gn(orig, cfg, name, **kw),
    nn.Sigmoid: lambda orig, cfg, name, **kw: _make_sigmoid(orig, cfg, name, **kw),
    # ... etc
}
```

### Step 5: Commit

```bash
git add src/session/_model.py
git commit -m "feat(session): wire quantize_nonlinear through _replace_module to make fns"
```

---

## Task 3: Norm 模块集成测试

**Files:**
- Modify: `src/tests/test_session_unit.py`

### Step 1: 添加 norm True 模式 cfg 字段验证测试

在 `TestQuantizeNonLinearSwitch` 中新增：

```python
def test_quantize_nonlinear_true_norm_has_per_block_input(self):
    """True + per_block compute: norm receives cfg.input = per_block (not stripped)."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.LayerNorm(8),
        nn.ReLU(),
    )
    qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)

    # BatchNorm should have per_block input/weight
    bn = qmodel[1]
    assert bn.cfg.input is not None
    assert bn.cfg.input.granularity.mode.name == "PER_BLOCK"
    assert bn.cfg.weight is not None
    assert bn.cfg.weight.granularity.mode.name == "PER_BLOCK"

    # LayerNorm should have per_block input/weight
    ln = qmodel[2]
    assert ln.cfg.input is not None
    assert ln.cfg.input.granularity.mode.name == "PER_BLOCK"
    assert ln.cfg.weight is not None
    assert ln.cfg.weight.granularity.mode.name == "PER_BLOCK"

    # Backward fields should be populated (storage exists)
    assert bn.cfg.is_training is True
    assert ln.cfg.is_training is True
    assert bn.quantize_backprop is True
    assert ln.quantize_backprop is True

def test_quantize_nonlinear_true_norm_inner_scheme_unchanged(self):
    """True: norm inner_scheme stays at storage (not upgraded to per_block)."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = nn.Sequential(nn.LayerNorm(8))
    qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)

    ln = qmodel[0]
    # inner_scheme must be storage, NOT per_block
    assert ln.inner_scheme is not None
    assert ln.inner_scheme.granularity.mode.name == "PER_TENSOR"
    # The inner_scheme should be the storage scheme
    assert ln.inner_scheme is cfg.storage

def test_quantize_nonlinear_true_vs_false_bit_exact_no_per_block(self):
    """Without per_block compute, True and False produce bit-exact same output."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="int8", w_granularity="per_tensor",
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    ).eval()

    x = torch.randn(1, 3, 8, 8)

    torch.manual_seed(42)
    model_false = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=False)
    out_false = model_false(x)

    torch.manual_seed(42)
    model_true = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)
    out_true = model_true(x)

    assert torch.equal(out_true, out_false), \
        "True and False must be bit-exact when no per_block compute"
```

### Step 2: 运行全部回归测试

```bash
pytest src/tests/test_session_unit.py -q -m "not slow"
# Expected: all pass, no regression
```

### Step 3: Commit

```bash
git add src/tests/test_session_unit.py
git commit -m "test(session): norm quantize_nonlinear=True cfg field + bit-exact parity tests"
```

---

## Task 4: Activation/Softmax/Pooling 入口量化分离

**Files:**
- Modify: `src/ops/_mixin.py`
- Modify: `src/ops/activations.py`
- Modify: `src/ops/softmax.py`
- Modify: `src/ops/pooling.py`
- Modify: `src/session/_model.py`

### Step 1: `_QuantizedModuleMixin` 新增 `_entry_quantize` 辅助方法

```python
# In _mixin.py, add to _QuantizedModuleMixin:

def _entry_quantize(self, input):
    """Apply storage → compute entry quantization for quantize_nonlinear=True.

    The two-stage entry quant is applied BEFORE the Function's inner_scheme
    vec_ops, so that operand entry gets per_block compute while internal
    vec_ops stay at storage level.
    """
    from src.quantize import quantize
    if self._entry_storage is not None:
        input = quantize(input, self._entry_storage)
    if self._entry_compute is not None:
        input = quantize(input, self._entry_compute)
    return input
```

并在 `_init_quant_cfg` 中初始化默认值：

```python
def _init_quant_cfg(self, cfg, inner_scheme, quantize_backprop, name):
    # ... existing code ...
    self._entry_storage = None
    self._entry_compute = None
```

### Step 2: 修改每个 Activation/Softmax/Pooling forward()

每个 activation 的 `forward()` 在调用 Function.apply 之前调用 `_entry_quantize`：

```python
# Sigmoid forward():
def forward(self, input):
    input = self._entry_quantize(input)
    inner_scheme = self.cfg.input
    quantize_backprop = self.cfg.grad_input is not None
    emit_fn = self._emit if self._observers else None
    result = SigmoidFunction.apply(
        input, inner_scheme, quantize_backprop, self._analysis_name, emit_fn,
    )
    return result

# Tanh forward()
# ReLU forward()
# ReLU6 forward()
# LeakyReLU forward()
# SiLU forward()
# GELU forward()
# Softmax forward()
# AdaptiveAvgPool2d forward()
# All follow the same pattern: input = self._entry_quantize(input) as first line
```

### Step 3: 修改 `_make_*` 函数在 True 时设置 `_entry_compute`

```python
def _make_sigmoid(orig: nn.Sigmoid, cfg: OpQuantConfig, name: str, quantize_nonlinear=False):
    from src.ops.activations import QuantizedSigmoid
    act_cfg = _activation_cfg(cfg)
    mod = QuantizedSigmoid(cfg=act_cfg, name=name)
    if quantize_nonlinear and cfg.storage is not None and cfg.input is not None:
        mod._entry_storage = cfg.storage
        mod._entry_compute = cfg.input  # per_block
    return mod
```

所有 activation/softmax/pooling `_make_*` 函数同样模式：构造 cfg 仍用 `_activation_cfg(cfg)`（inner_scheme=storage），`True` 时额外设置 `_entry_storage` 和 `_entry_compute`。

### Step 4: Commit

```bash
git add src/ops/_mixin.py src/ops/activations.py src/ops/softmax.py src/ops/pooling.py src/session/_model.py
git commit -m "feat(ops): separate entry quant from inner_scheme for nonlinear ops"
```

---

## Task 5: Activation/Softmax False 回归测试

### Step 1: 添加测试确认 `_entry_quantize` 在 False 时为 no-op

```python
def test_entry_quantize_is_noop_when_false(self):
    """_entry_compute is None when quantize_nonlinear=False → _entry_quantize no-op."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = nn.Sequential(nn.ReLU(), nn.Sigmoid(), nn.SiLU())
    qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=False)

    # All activations should have no _entry_compute
    for mod in qmodel:
        assert getattr(mod, '_entry_compute', None) is None
        assert getattr(mod, '_entry_storage', None) is None

    # Forward pass should still work (regression)
    x = torch.randn(1, 16)
    out = qmodel(x)
    assert out.shape == x.shape
```

### Step 2: 运行现有测试确保无回归

```bash
pytest src/tests/test_session_unit.py -q -m "not slow"
pytest src/tests/test_ops_equiv_activations.py -q
pytest src/tests/test_ops_equiv_norm.py -q
# Expected: all pass
```

### Step 3: Commit

```bash
git add src/tests/test_session_unit.py
git commit -m "test(session): verify _entry_quantize no-op when quantize_nonlinear=False"
```

---

## Task 6: E2E 集成测试 — quantize_nonlinear=True 完整流程

**Files:**
- Modify: `src/tests/test_session_unit.py`

### Step 1: 添加完整 E2E 测试

```python
def test_quantize_nonlinear_true_e2e_forward_backward(self):
    """Full model forward+backward with quantize_nonlinear=True + per_block compute."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = _make_model_with_all_op_types()
    qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)
    qmodel.train()

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    out = qmodel(x)
    loss = out.sum()
    loss.backward()

    # All params should have non-zero gradients
    for name, param in qmodel.named_parameters():
        assert param.grad is not None, f"{name} has None grad"
        assert param.grad.abs().sum() > 0, f"{name} has zero grad"

    # Input gradient must be non-zero
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

def test_quantize_nonlinear_true_mx_no_storage_e2e(self):
    """MX per_block without storage (bfloat=0): True mode forward+backward."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        calibrator="max",
    ).to_op_config()

    model = _make_model_with_all_op_types()
    qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)
    qmodel.train()

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    out = qmodel(x)
    loss = out.sum()
    loss.backward()

    # Should complete without NaN or inf
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    for param in qmodel.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

def test_quantize_nonlinear_true_vs_false_snr_in_range(self):
    """With per_block compute, True vs False SNR should be within expected range.

    True adds per_block quant on operand entry → output differs from False.
    But the difference should be bounded (SNR > 20 dB for fp8 per_block).
    """
    import copy
    from src.session._model import quantize_model
    from src.report._metrics import compute_snr

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128, 32),
        nn.LayerNorm(32),
        nn.SiLU(),
        nn.Linear(32, 10),
        nn.Softmax(dim=1),
    ).eval()

    x = torch.randn(4, 3, 8, 8)

    torch.manual_seed(42)
    model_false = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=False)
    out_false = model_false(x)

    torch.manual_seed(42)
    model_true = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)
    out_true = model_true(x)

    snr = compute_snr(out_true, out_false)
    assert snr > 20.0, f"SNR {snr:.1f} dB too low between True and False"
```

### Step 2: 运行全量测试

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"
# Expected: all pass, no regression
```

### Step 3: Commit

```bash
git add src/tests/test_session_unit.py
git commit -m "test(session): E2E quantize_nonlinear=True forward/backward + SNR tests"
```

---

## Task 7: 属性测试 — 幂等性与无 NaN

### Step 1: 添加属性测试

```python
def test_quantize_nonlinear_true_idempotent_output(self):
    """Repeated forward with same input produces identical output."""
    import copy
    from src.session._model import quantize_model

    cfg = QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_bits=16, storage_kind="bfloat",
        calibrator="max",
    ).to_op_config()

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    ).eval()

    qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)
    x = torch.randn(1, 3, 8, 8)

    out1 = qmodel(x)
    out2 = qmodel(x)
    assert torch.equal(out1, out2)

def test_quantize_nonlinear_true_no_nan_inf(self):
    """quantize_nonlinear=True produces no NaN or inf in outputs across formats."""
    import copy
    from src.session._model import quantize_model

    formats = [
        ("fp8_e4m3", "per_block", 32),
        ("fp8_e5m2", "per_block", 32),
        ("int8", "per_tensor", None),
        ("fp4_e2m1", "per_block", 32),
    ]

    model = _make_model_with_all_op_types().eval()
    x = torch.randn(2, 3, 8, 8)

    for fmt, gran, bs in formats:
        cfg = QuantConfig(
            w_format=fmt, w_granularity=gran, w_block_size=bs,
            a_format=fmt, a_granularity=gran, a_block_size=bs,
            storage_bits=16, storage_kind="bfloat",
            calibrator="max",
        ).to_op_config()

        qmodel = quantize_model(copy.deepcopy(model), cfg, quantize_nonlinear=True)
        out = qmodel(x)
        assert not torch.isnan(out).any(), f"NaN in output for format={fmt}"
        assert not torch.isinf(out).any(), f"inf in output for format={fmt}"
```

### Step 2: 运行

```bash
pytest src/tests/test_session_unit.py::TestQuantizeNonLinearSwitch -v
```

### Step 3: Commit

```bash
git add src/tests/test_session_unit.py
git commit -m "test(session): idempotence + no-NaN property tests for quantize_nonlinear=True"
```

---

## 判断标准（验收门）

- [ ] `_nonlinear_true_cfg()` 保留 per_block compute 字段 + storage 级 backward
- [ ] `_replace_module()` 将 `quantize_nonlinear` 传递给 `_make_*` 函数
- [ ] Norm 模块在 `True` 时 `cfg.input`/`cfg.weight` 为 per_block，`inner_scheme` 仍为 storage
- [ ] Activation/Softmax/Pooling 在 `True` 时 `_entry_compute` 为 per_block，`inner_scheme` 为 storage
- [ ] 无 per_block compute 时 `True` ≡ `False`（bit-exact）
- [ ] 全量测试 2,069+ passed，无回归
