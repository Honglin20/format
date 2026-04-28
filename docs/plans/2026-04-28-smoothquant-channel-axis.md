# SmoothQuant Configurable Channel Axis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `channel_axis` parameter to `SmoothQuantTransform`, `SmoothQuantWeightTransform`, and `compute_smoothquant_scale` so Conv2d (channel at dim=1) works correctly, while maintaining backward compatibility for Linear (channel at dim=-1).

**Architecture:** Three parameters control axis semantics:
- `compute_smoothquant_scale(..., act_channel_axis=-1, w_channel_axis=1)` — which dims to reduce for per-channel max
- `SmoothQuantTransform(scale, channel_axis=-1)` — which dim of `x` the scale broadcasts along (activation channel)
- `SmoothQuantWeightTransform(scale, channel_axis=1)` — which dim of `W` the scale broadcasts along (weight input channel)

All new params default to current hardcoded values (backward compatible). Equality and hashing are extended to include `channel_axis`.

**Tech Stack:** PyTorch, pytest

---

### The Bug Being Fixed

For Linear input `(N, C)` or `(B, S, C)`:

| Operation | Axis | Result |
|---|---|---|
| `compute_smoothquant_scale` reduces activation | dim=-1 = C | scale `[C]` ✓ |
| `SmoothQuantTransform.forward(x)` | dim=-1 = C | `x / s` ✓ |

For Conv2d input `(N, C, H, W)`:

| Operation | Axis | Result |
|---|---|---|
| `compute_smoothquant_scale` reduces activation | dim=-1 = **W** | scale `[W]` **WRONG** |
| `SmoothQuantTransform.forward(x)` | dim=-1 = **W** | `x / s` on width **WRONG** |
| `SmoothQuantWeightTransform.forward(W)` | dim=1 = IC | tries `[W]` scale on `[IC]` dim → runtime error or silent nonsense |

---

### Task 1: Add `channel_axis` to `SmoothQuantTransform`

**Files:**
- Modify: `src/transform/smooth_quant.py:91-189`
- Create: tests in `src/tests/test_transform_smooth_quant.py`

**Step 1: Write failing tests for `channel_axis` on `SmoothQuantTransform`**

Add to `TestSmoothQuantTransform` class:

```python
def test_channel_axis_conv2d_activation(self):
    """channel_axis=1: scale broadcasts to dim=1 for Conv2d-style input (N, C, H, W)."""
    _, SmoothQuantTransform = _try_import()
    scale = torch.tensor([2.0, 4.0, 1.0])  # 3 channels
    t = SmoothQuantTransform(scale, channel_axis=1)
    # Conv2d-style input: (N=2, C=3, H=2, W=2)
    x = torch.tensor([
        [[[1.0, 2.0], [3.0, 4.0]],
         [[5.0, 6.0], [7.0, 8.0]],
         [[9.0, 10.0], [11.0, 12.0]]],
        [[[13.0, 14.0], [15.0, 16.0]],
         [[17.0, 18.0], [19.0, 20.0]],
         [[21.0, 22.0], [23.0, 24.0]]]
    ])
    result = t.forward(x)
    # Channel 0 divided by 2.0, channel 1 by 4.0, channel 2 by 1.0
    expected_c0 = x[:, 0:1, :, :] / 2.0
    expected_c1 = x[:, 1:2, :, :] / 4.0
    expected_c2 = x[:, 2:3, :, :] / 1.0
    assert torch.equal(result[:, 0:1], expected_c0)
    assert torch.equal(result[:, 1:2], expected_c1)
    assert torch.equal(result[:, 2:3], expected_c2)

def test_channel_axis_default_is_minus_one(self):
    """Default channel_axis=-1 preserves backward compatibility."""
    _, SmoothQuantTransform = _try_import()
    t1 = SmoothQuantTransform(torch.tensor([2.0, 4.0]))
    t2 = SmoothQuantTransform(torch.tensor([2.0, 4.0]), channel_axis=-1)
    assert t1 == t2

def test_channel_axis_roundtrip_conv2d(self):
    """Roundtrip with channel_axis=1 recovers original (Conv2d layout)."""
    _, SmoothQuantTransform = _try_import()
    scale = torch.tensor([0.5, 2.0, 1.0, 4.0])
    t = SmoothQuantTransform(scale, channel_axis=1)
    x = torch.randn(2, 4, 4, 4)  # (N=2, C=4, H=4, W=4)
    xr = t.inverse(t.forward(x))
    assert torch.equal(xr, x), f"Roundtrip failed with channel_axis=1"

def test_eq_different_channel_axis(self):
    """Transforms with same scale but different channel_axis are not equal."""
    _, SmoothQuantTransform = _try_import()
    scale = torch.tensor([0.5, 2.0, 1.0])
    t1 = SmoothQuantTransform(scale, channel_axis=-1)
    t2 = SmoothQuantTransform(scale, channel_axis=1)
    assert t1 != t2

def test_hash_different_channel_axis(self):
    """Transforms with same scale but different channel_axis have different hash."""
    _, SmoothQuantTransform = _try_import()
    scale = torch.tensor([0.5, 2.0, 1.0])
    t1 = SmoothQuantTransform(scale, channel_axis=-1)
    t2 = SmoothQuantTransform(scale, channel_axis=1)
    assert hash(t1) != hash(t2)

def test_channel_axis_rejects_zero(self):
    """channel_axis=0 should be valid (unusual but allowed)."""
    _, SmoothQuantTransform = _try_import()
    scale = torch.tensor([2.0, 4.0])
    t = SmoothQuantTransform(scale, channel_axis=0)
    x = torch.randn(2, 3)  # channel on dim 0
    xr = t.inverse(t.forward(x))
    assert torch.equal(xr, x)
```

**Step 2: Run tests to confirm they fail**

```bash
pytest src/tests/test_transform_smooth_quant.py::TestSmoothQuantTransform::test_channel_axis_conv2d_activation -v
# Expected: TypeError: SmoothQuantTransform.__init__() got an unexpected keyword argument 'channel_axis'
```

**Step 3: Implement `channel_axis` in `SmoothQuantTransform`**

Changes in `src/transform/smooth_quant.py`:

```python
class SmoothQuantTransform(TransformBase):
    """Pre-quantization SmoothQuant activation smoothing (immutable scale).

    Applies per-channel scaling to activations before quantization::

        forward(x) = x / scale
        inverse(x_q) = x_q * scale

    The scale is immutable after construction. Use :meth:`from_calibration`
    to create from activation statistics and a weight tensor; this avoids
    the "uncalibrated" illegal state that a mutable design would allow.

    Args:
        scale: 1D tensor of per-channel smoothing factors.
        channel_axis: The dimension of the input tensor ``x`` that
            corresponds to the channel dimension.  Default ``-1`` (last
            dim, matching ``nn.Linear`` activations).  Use ``1`` for
            ``nn.Conv2d`` (NCHW layout) or ``nn.Conv1d`` (NCL layout).
    """

    invertible = True

    def __init__(self, scale: Tensor, channel_axis: int = -1):
        """Create SmoothQuantTransform with a pre-computed per-channel scale.

        Args:
            scale: 1D tensor of per-channel smoothing factors. Must be
                   strictly positive (values <= 0 will produce NaNs in
                   forward/inverse).
            channel_axis: Dimension of ``x`` that is the channel axis.
                   Default ``-1`` (last dim).
        """
        object.__setattr__(self, "_scale", scale.detach().clone())
        object.__setattr__(self, "_channel_axis", channel_axis)

    @property
    def scale(self) -> Tensor:
        """The per-channel smoothing factor (read-only)."""
        return self._scale

    @property
    def channel_axis(self) -> int:
        """The channel dimension index."""
        return self._channel_axis

    def _broadcast_scale(self, x: Tensor) -> Tensor:
        """Reshape ``self._scale`` to broadcast against ``x``.

        Places scale at ``self._channel_axis`` and size-1 everywhere else.
        """
        shape = [1] * x.ndim
        shape[self._channel_axis] = -1
        return self._scale.view(*shape)

    # forward, inverse unchanged (delegate to _broadcast_scale)

    # from_calibration — updated in Task 3

    def __eq__(self, other) -> bool:
        if not isinstance(other, SmoothQuantTransform):
            return False
        return (self._channel_axis == other._channel_axis
                and torch.equal(self._scale, other._scale))

    def __hash__(self) -> int:
        return hash((self._channel_axis,
                     tuple(self._scale.flatten().tolist())))
```

Key changes:
1. `__init__` takes `channel_axis: int = -1` — stored immutably via `object.__setattr__`
2. New `channel_axis` property (read-only)
3. `_broadcast_scale` uses `self._channel_axis` instead of hardcoded `-1`:
   - Before: `shape = [1] * (x.ndim - 1) + [-1]`
   - After: `shape = [1] * x.ndim; shape[self._channel_axis] = -1`
4. `__eq__` includes `self._channel_axis == other._channel_axis`
5. `__hash__` includes `self._channel_axis`

**Step 4: Run tests to confirm they pass**

```bash
pytest src/tests/test_transform_smooth_quant.py::TestSmoothQuantTransform -v
# Expected: all pass (existing + new)
```

**Step 5: Commit**

```bash
git add src/transform/smooth_quant.py src/tests/test_transform_smooth_quant.py
git commit -m "feat(transform): add configurable channel_axis to SmoothQuantTransform"
```

---

### Task 2: Add `channel_axis` to `SmoothQuantWeightTransform`

**Files:**
- Modify: `src/transform/smooth_quant.py:192-264`
- New: tests in `src/tests/test_transform_smooth_quant.py`

**Step 1: Write tests for `SmoothQuantWeightTransform` (whole class is untested!)**

Add new test class `TestSmoothQuantWeightTransform`:

```python
class TestSmoothQuantWeightTransform:

    def _try_import_wt(self):
        try:
            from src.transform.smooth_quant import SmoothQuantWeightTransform
            return SmoothQuantWeightTransform
        except (ImportError, AttributeError):
            pytest.skip("SmoothQuantWeightTransform not yet implemented")

    def test_roundtrip_linear_weight(self):
        """Roundtrip for Linear weight (OC, IC)."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([0.5, 2.0, 1.0])  # 3 input channels
        t = SmoothQuantWeightTransform(scale)  # default channel_axis=1
        W = torch.randn(4, 3)  # (OC=4, IC=3)
        Wr = t.inverse(t.forward(W))
        assert torch.equal(Wr, W)

    def test_roundtrip_conv_weight(self):
        """Roundtrip for Conv2d weight (OC, IC, KH, KW)."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([0.5, 2.0])
        t = SmoothQuantWeightTransform(scale, channel_axis=1)
        W = torch.randn(8, 2, 3, 3)  # (OC=8, IC=2, KH=3, KW=3)
        Wr = t.inverse(t.forward(W))
        assert torch.equal(Wr, W)

    def test_forward_linear_weight(self):
        """forward(W) = W * scale along dim=1."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantWeightTransform(scale)
        W = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        expected = torch.tensor([[2.0, 8.0],
                                 [6.0, 16.0]])
        assert torch.equal(t.forward(W), expected)

    def test_forward_conv_weight(self):
        """forward(W) = W * scale along dim=1 for Conv weight."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0])
        t = SmoothQuantWeightTransform(scale, channel_axis=1)
        W = torch.ones(4, 1, 2, 2)  # (OC=4, IC=1, H=2, W=2)
        result = t.forward(W)
        assert torch.equal(result, W * 2.0)

    def test_inverse_linear_weight(self):
        """inverse(W_q) = W_q / scale along dim=1."""
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t = SmoothQuantWeightTransform(scale)
        W_q = torch.tensor([[2.0, 8.0],
                            [6.0, 16.0]])
        expected = torch.tensor([[1.0, 2.0],
                                 [3.0, 4.0]])
        assert torch.equal(t.inverse(W_q), expected)

    def test_invertible_flag(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t = SmoothQuantWeightTransform(torch.tensor([1.0, 2.0]))
        assert t.invertible is True

    def test_is_transform_base(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        from src.scheme.transform import TransformBase
        assert issubclass(SmoothQuantWeightTransform, TransformBase)

    def test_scale_stored_as_clone(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        original = torch.tensor([1.0, 2.0, 4.0])
        t = SmoothQuantWeightTransform(original)
        original[0] = 999.0
        W = torch.randn(4, 3)
        Wr = t.inverse(t.forward(W))
        assert torch.equal(Wr, W)

    def test_channel_axis_default_is_one(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t1 = SmoothQuantWeightTransform(torch.tensor([2.0, 4.0]))
        t2 = SmoothQuantWeightTransform(torch.tensor([2.0, 4.0]), channel_axis=1)
        assert t1 == t2

    def test_eq_different_channel_axis(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t1 = SmoothQuantWeightTransform(scale, channel_axis=0)
        t2 = SmoothQuantWeightTransform(scale, channel_axis=1)
        assert t1 != t2

    def test_hash_different_channel_axis(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        scale = torch.tensor([2.0, 4.0])
        t1 = SmoothQuantWeightTransform(scale, channel_axis=0)
        t2 = SmoothQuantWeightTransform(scale, channel_axis=1)
        assert hash(t1) != hash(t2)

    def test_eq_same_scale_and_axis(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t1 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert t1 == t2

    def test_hash_consistent(self):
        SmoothQuantWeightTransform = self._try_import_wt()
        t1 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        t2 = SmoothQuantWeightTransform(torch.tensor([0.5, 2.0, 1.0]))
        assert hash(t1) == hash(t2)
```

**Step 2: Run tests to confirm they fail**

```bash
pytest src/tests/test_transform_smooth_quant.py::TestSmoothQuantWeightTransform -v
# Expected: FAIL with "SmoothQuantWeightTransform.__init__() got an unexpected keyword argument 'channel_axis'"
```

**Step 3: Implement `channel_axis` in `SmoothQuantWeightTransform`**

Changes in `src/transform/smooth_quant.py`:

```python
class SmoothQuantWeightTransform(TransformBase):
    """Weight-side SmoothQuant compensation: ``forward(W) = W * scale``.

    Companion to :class:`SmoothQuantTransform`.  While the activation side
    divides by the per-channel scale (``x / s``), the weight side multiplies
    by the same scale (``W * s``) to maintain mathematical equivalence:

        (X / s) @ (W * s) = X @ W

    The scale is the same tensor used in the activation-side transform,
    obtained via ``sq_transform.scale``.

    The weight's input-channel dimension defaults to **dim 1** (``[OC, IC]``
    for Linear, ``[OC, IC, H, W]`` for Conv).  The scale broadcasts as
    ``[1, IC, 1, ...]``.
    """

    invertible = True

    def __init__(self, scale: Tensor, channel_axis: int = 1):
        """Create SmoothQuantWeightTransform with a pre-computed per-channel scale.

        Args:
            scale: 1D tensor of per-channel smoothing factors (same scale
                   as used in ``SmoothQuantTransform``).  Cloned internally.
            channel_axis: Dimension of ``W`` that is the input-channel axis.
                   Default ``1`` (matches both Linear [OC, IC] and
                   Conv [OC, IC, H, W] weight layouts).
        """
        object.__setattr__(self, "_scale", scale.detach().clone())
        object.__setattr__(self, "_channel_axis", channel_axis)

    @property
    def scale(self) -> Tensor:
        """The per-channel compensation factor (read-only)."""
        return self._scale

    @property
    def channel_axis(self) -> int:
        """The weight input-channel dimension index."""
        return self._channel_axis

    def _broadcast_scale(self, W: Tensor) -> Tensor:
        """Reshape scale to broadcast against weight tensor.

        Places scale at ``self._channel_axis`` and size-1 everywhere else.
        """
        shape = [1] * W.ndim
        shape[self._channel_axis] = -1
        return self._scale.view(*shape)

    # forward and inverse unchanged

    def __eq__(self, other) -> bool:
        if not isinstance(other, SmoothQuantWeightTransform):
            return False
        return (self._channel_axis == other._channel_axis
                and torch.equal(self._scale, other._scale))

    def __hash__(self) -> int:
        return hash((self._channel_axis,
                     tuple(self._scale.flatten().tolist())))
```

**Step 4: Run tests to confirm they pass**

```bash
pytest src/tests/test_transform_smooth_quant.py::TestSmoothQuantWeightTransform -v
# Expected: all pass
```

**Step 5: Commit**

```bash
git add src/transform/smooth_quant.py src/tests/test_transform_smooth_quant.py
git commit -m "feat(transform): add configurable channel_axis to SmoothQuantWeightTransform"
```

---

### Task 3: Add axis params to `compute_smoothquant_scale` and `from_calibration`

**Files:**
- Modify: `src/transform/smooth_quant.py:25-189`

**Step 1: Write failing tests for axis params on `compute_smoothquant_scale`**

Add to `TestComputeSmoothQuantScale` class:

```python
def test_act_channel_axis_conv2d(self):
    """act_channel_axis=1 reduces over all dims except dim=1 for Conv2d-style activation."""
    compute_smoothquant_scale, _ = _try_import()
    torch.manual_seed(42)
    # Conv2d activation: (N=2, C=3, H=4, W=5) — channel at dim=1
    X = torch.randn(2, 3, 4, 5)
    W = torch.randn(8, 3, 3, 3)  # Conv weight (OC=8, IC=3, KH=3, KW=3)
    
    scale = compute_smoothquant_scale(X, W, alpha=0.5,
                                       act_channel_axis=1, w_channel_axis=1)
    
    assert scale.shape == (3,), f"Expected (3,), got {scale.shape}"
    
    # Verify act_channel_axis=1 gives same result as manually computing per-channel
    act_amax_manual = torch.amax(torch.abs(X), dim=(0, 2, 3))  # reduce all except dim 1
    w_amax_manual = torch.amax(torch.abs(W), dim=(0, 2, 3))    # reduce all except dim 1
    expected = (act_amax_manual.clamp(min=1e-12).pow(0.5)
                / w_amax_manual.clamp(min=1e-12).pow(0.5))
    assert torch.allclose(scale, expected, atol=1e-6)

def test_act_channel_axis_default(self):
    """Default act_channel_axis=-1 preserves backward compatibility."""
    compute_smoothquant_scale, _ = _try_import()
    torch.manual_seed(42)
    X = torch.randn(4, 16)
    W = torch.randn(8, 16)
    scale_new = compute_smoothquant_scale(X, W, act_channel_axis=-1, w_channel_axis=1)
    scale_old = compute_smoothquant_scale(X, W)
    assert torch.equal(scale_new, scale_old)

def test_w_channel_axis_zero(self):
    """w_channel_axis=0: reduce all except dim 0 (unusual weight layout)."""
    compute_smoothquant_scale, _ = _try_import()
    torch.manual_seed(42)
    # Unusual weight: [IC=4, OC=8] (transposed)
    act_stats = torch.randn(4).abs()
    W = torch.randn(4, 8)  # [IC, OC] — input channel at dim 0
    scale = compute_smoothquant_scale(act_stats, W, w_channel_axis=0)
    assert scale.shape == (4,)
```

**Step 2: Run to confirm failure**

```bash
pytest src/tests/test_transform_smooth_quant.py::TestComputeSmoothQuantScale::test_act_channel_axis_conv2d -v
# Expected: TypeError: compute_smoothquant_scale() got an unexpected keyword argument 'act_channel_axis'
```

**Step 3: Implement axis params**

In `compute_smoothquant_scale`:

```python
def compute_smoothquant_scale(
    X_act: Tensor,
    W: Tensor,
    alpha: float = 0.5,
    act_channel_axis: int = -1,
    w_channel_axis: int = 1,
) -> Tensor:
    """Compute per-channel SmoothQuant smoothing factor.

    The scale for each channel j is::

        s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

    where ``X_j`` refers to the j-th channel of the activation and ``W_j``
    refers to the j-th input channel of the weight.

    Args:
        X_act: Per-channel max absolute activation values, or raw activation
               tensor. If 1D, treated as per-channel statistics directly.
               If 2D+, reduced along all dims except ``act_channel_axis``.
        W: Weight tensor. Reduced along all dims except ``w_channel_axis``.
        alpha: Smoothing strength. 0 = all weight, 1 = all activation.
               Default 0.5.
        act_channel_axis: Channel axis of ``X_act``. Default ``-1``
               (matching ``nn.Linear`` activations). Use ``1`` for
               ``nn.Conv2d`` (NCHW).
        w_channel_axis: Input-channel axis of ``W``. Default ``1``
               (matching both ``nn.Linear`` [OC, IC] and ``nn.Conv2d``
               [OC, IC, ...] weight layouts).

    Returns:
        Scale tensor of shape ``[C]`` where ``C`` is the channel count.

    Raises:
        ValueError: If ``alpha`` is outside [0, 1].
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # --- Activation per-channel max ---
    if X_act.ndim == 1:
        act_amax = X_act.abs()
    else:
        reduce_dims = tuple(d for d in range(X_act.ndim) if d != act_channel_axis)
        act_amax = torch.amax(torch.abs(X_act), dim=reduce_dims)

    # --- Weight per-input-channel max ---
    w_reduce_dims = tuple(d for d in range(W.ndim) if d != w_channel_axis)
    w_amax = torch.amax(torch.abs(W), dim=w_reduce_dims)

    # --- SmoothQuant scale ---
    s = (
        act_amax.clamp(min=1e-12).pow(alpha)
        / w_amax.clamp(min=1e-12).pow(1.0 - alpha)
    )
    return s
```

In `SmoothQuantTransform.from_calibration`:

```python
@staticmethod
def from_calibration(
    X_act: Tensor, W: Tensor, alpha: float = 0.5,
    act_channel_axis: int = -1, w_channel_axis: int = 1,
) -> "SmoothQuantTransform":
    """Factory: compute scale from activation statistics and weight.

    Convenience wrapper around :func:`compute_smoothquant_scale` that
    returns a fully-formed transform.

    Args:
        X_act: Activation statistics or raw activation tensor.
        W: Weight tensor.
        alpha: Smoothing strength (default 0.5).
        act_channel_axis: Channel axis of ``X_act``. Default ``-1``
               (nn.Linear). Use ``1`` for nn.Conv2d (NCHW).
        w_channel_axis: Input-channel axis of ``W``. Default ``1``.

    Returns:
        A ``SmoothQuantTransform`` with the computed scale and
        ``channel_axis`` set to ``act_channel_axis``.
    """
    scale = compute_smoothquant_scale(
        X_act, W, alpha,
        act_channel_axis=act_channel_axis,
        w_channel_axis=w_channel_axis,
    )
    return SmoothQuantTransform(scale, channel_axis=act_channel_axis)
```

**Step 4: Run all SmoothQuant tests to confirm they pass**

```bash
pytest src/tests/test_transform_smooth_quant.py -v
# Expected: all pass
```

**Step 5: Commit**

```bash
git add src/transform/smooth_quant.py src/tests/test_transform_smooth_quant.py
git commit -m "feat(transform): add configurable channel axes to compute_smoothquant_scale and from_calibration"
```

---

### Task 4: Fix experiment to pass correct axis for Conv2d

**Files:**
- Modify: `examples/experiment_format_study.py:534-588`

**Step 1: No new test needed (existing experiment is the test)**

The fix ensures `_make_smoothquant_transforms` passes the right `act_channel_axis`.

**Step 2: Identify the current code** (already known)

Lines 566-587: hook registration uses `isinstance(module, (nn.Linear, nn.Conv2d))` but `from_calibration` call does not distinguish.

**Step 3: Implement**

Replace the `_make_smoothquant_transforms` function:

```python
def _make_smoothquant_transforms(
    fp32_model: nn.Module,
    calib_data: List[torch.Tensor],
) -> Dict[str, TransformBase]:
    """Create per-layer SmoothQuantTransform dict.

    Runs one forward pass through the FP32 model to capture each layer's
    activation and weight, then creates a per-layer SmoothQuantTransform
    with correctly-shaped per-channel scales.

    For ``nn.Linear``, the activation channel is dim ``-1`` (last dim).
    For ``nn.Conv2d``, the activation channel is dim ``1`` (NCHW layout).

    Args:
        fp32_model: FP32 reference model.
        calib_data: List of calibration batches (first batch used).

    Returns:
        Dict mapping layer name to ``SmoothQuantTransform`` (or
        ``IdentityTransform`` on failure).
    """
    if fp32_model is None:
        return {}

    activations: Dict[str, torch.Tensor] = {}
    weights: Dict[str, torch.Tensor] = {}
    channel_axes: Dict[str, int] = {}  # per-module activation channel axis
    hooks = []

    def _hook(name):
        def fn(module, _input, _output):
            activations[name] = _input[0].detach()
            if hasattr(module, "weight") and module.weight is not None:
                weights[name] = module.weight.data
        return fn

    for name, module in fp32_model.named_modules():
        if isinstance(module, nn.Linear):
            channel_axes[name] = -1  # activation channel = last dim
            hooks.append(module.register_forward_hook(_hook(name)))
        elif isinstance(module, nn.Conv2d):
            channel_axes[name] = 1   # activation channel = dim 1 (NCHW)
            hooks.append(module.register_forward_hook(_hook(name)))

    if not hooks:
        return {}

    with torch.no_grad():
        fp32_model.eval()
        fp32_model(calib_data[0])

    for h in hooks:
        h.remove()

    per_layer: Dict[str, TransformBase] = {}
    for name in activations:
        if name in weights:
            try:
                per_layer[name] = SmoothQuantTransform.from_calibration(
                    X_act=activations[name], W=weights[name], alpha=0.5,
                    act_channel_axis=channel_axes.get(name, -1),
                )
            except (ValueError, RuntimeError) as e:
                print(f"  Warning: SmoothQuant for {name}: {e}")
                per_layer[name] = IdentityTransform()

    return per_layer
```

Key change: `channel_axes` dict stores the activation channel axis per module name. `nn.Linear` → `-1`, `nn.Conv2d` → `1`. Passed to `from_calibration(act_channel_axis=...)`.

Also removed the `isinstance(module, (nn.Linear, nn.Conv2d))` in favor of two `isinstance` branches.

**Step 4: Verify — run the full test suite**

```bash
pytest src/tests/ -q
# Expected: all 1305 tests pass
```

**Step 5: Commit**

```bash
git add examples/experiment_format_study.py
git commit -m "fix(example): pass correct channel_axis to SmoothQuantTransform per module type"
```

---

### Task 5: Cross-check all callers for backward compatibility

**Files to verify (no changes needed, just audit):**
- `examples/06_transforms.py:87` — `from_calibration(calib_act, W=fc1.weight, alpha=0.5)` → uses Linear weight, default axis correct
- `examples/00_comprehensive.py:219` — `from_calibration(act, ref.fc1.weight.data, alpha=0.5)` → uses Linear weight, default axis correct
- `src/transform/__init__.py` — re-exports both classes, no API change

**Other callers of `SmoothQuantWeightTransform` (no changes needed):**
- `examples/experiment_format_study.py:278` — `SmoothQuantWeightTransform(sq_transform.scale)` with default `channel_axis=1` — correct for both Linear and Conv2d weights

All callers are backward-compatible because all new parameters have defaults matching the old hardcoded values.

**Step 1: Run full test suite to confirm no regressions**

```bash
pytest src/tests/ -q
# Expected: all 1305 pass (regression-free)
```

**Step 2: Commit** (if any documentation updates needed)

```bash
# Only if README or docstrings needed updating
git add README.md
git commit -m "docs: note channel_axis support in SmoothQuant transforms"
```

---

## Summary of Changes

| File | Change | Risk |
|------|--------|------|
| `src/transform/smooth_quant.py` | Add `channel_axis` to `SmoothQuantTransform`, `SmoothQuantWeightTransform`, `compute_smoothquant_scale`; update `_broadcast_scale`, `__eq__`, `__hash__` | Low — all new params have backward-compatible defaults |
| `src/tests/test_transform_smooth_quant.py` | 6 new tests for channel_axis on SmoothQuantTransform; 3 new tests for compute_smoothquant_scale axis params; 13 new tests for SmoothQuantWeightTransform (entire class) | Test-only |
| `examples/experiment_format_study.py` | `_make_smoothquant_transforms` passes `act_channel_axis` per module type | Low — only hook registration changes |
| `examples/06_transforms.py` | No changes needed | — |
| `examples/00_comprehensive.py` | No changes needed | — |

## Acceptance Criteria

1. `SmoothQuantTransform(channel_axis=1)` correctly applies scale along dim 1 for Conv2d inputs
2. `SmoothQuantTransform()` with default `channel_axis=-1` unchanged for Linear inputs
3. `compute_smoothquant_scale(act_channel_axis=1)` correctly computes per-channel max along dim 1
4. `SmoothQuantWeightTransform` has `channel_axis` support (default 1)
5. `_make_smoothquant_transforms` passes correct axis for Linear vs Conv2d
6. All existing callers work without changes (backward compatible)
7. All 1305+ tests pass
8. `SmoothQuantWeightTransform` has complete test coverage (currently zero tests)
