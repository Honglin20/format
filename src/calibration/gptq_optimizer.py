"""GPTQOptimizer: Hessian-based column-by-column weight quantization.

GPTQ (Frantar et al., 2023) quantizes weights sequentially, using
second-order (Hessian) information to compensate for quantization error
in the remaining unquantized columns.  This produces weights with lower
layer-output MSE than naive per-channel rounding.

Architecture
------------
Follows the ``LayerwiseScaleOptimizer`` pattern:

1. Enumerate quantized modules via ``_get_quantized_modules``.
2. For each module with a weight QuantScheme: collect calibration inputs,
   compute the Hessian, then run column-by-column GPTQ.
3. Store the quantized weights directly on ``module.weight.data``.

No new transforms, formats, or buffers.  The standard forward path
re-quantizes via ``quantize(w, scheme)`` — idempotent on GPTQ weights.

Restrictions (v1)
-----------------
- Weight-only: only ``cfg.weight`` is processed; activations are unchanged.
- Per-channel granularity only.  Per-block (MX) granularity requires
  different handling inside the Hessian update loop.
- ``nn.Linear`` only.  ``nn.Conv2d`` support can be added later.
"""
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.quantize.elemwise import quantize
from src.scheme.granularity import GranularityMode
from src.session._model import _get_quantized_modules


def _precompute_scale(W: torch.Tensor, scheme) -> Optional[torch.Tensor]:
    """Pre-compute the per-channel or per-tensor amax from the full weight.

    Returns the scale tensor that ``quantize(..., scale=result)`` expects,
    or ``None`` if the scheme's format computes scales differently
    (float formats, per_block granularity).

    The returned scale has the same dtype as *W*.
    """
    gran = scheme.granularity
    if gran.mode == GranularityMode.PER_CHANNEL:
        ax = gran.channel_axis if gran.channel_axis >= 0 else W.ndim + gran.channel_axis
        dims_to_reduce = tuple(d for d in range(W.ndim) if d != ax)
        amax = torch.amax(torch.abs(W), dim=dims_to_reduce, keepdim=True)
        return amax.clamp(min=1e-12)

    if gran.mode == GranularityMode.PER_TENSOR:
        amax = torch.amax(torch.abs(W))
        if not torch.isfinite(amax) or amax <= 0:
            return None
        return amax.clamp(min=1e-12)

    # PER_BLOCK / DYNAMIC_GROUP: scale computation varies per block;
    # passing a single scale is not meaningful.
    return None


class GPTQOptimizer:
    """GPTQ: column-by-column weight quantization with Hessian compensation.

    Args:
        block_size: Number of columns processed per Hessian update.
            Larger = faster but coarser compensation.  Common: 128, 64.
        damp_percent: Hessian diagonal damping as a fraction of the mean
            diagonal.  Must be in (0, 1].  Default 0.01 (1%).
        act_order: If True, quantize columns in descending order of
            Hessian diagonal (high-impact columns first).  Improves
            accuracy on some architectures at a small runtime cost.
        num_batches: Maximum number of calibration batches to use when
            ``eval_fn`` is None and ``calib_data`` is a list of tensors.
    """

    def __init__(
        self,
        block_size: int = 128,
        damp_percent: float = 0.01,
        act_order: bool = False,
        num_batches: int = 8,
    ):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if damp_percent <= 0 or damp_percent > 1:
            raise ValueError(
                f"damp_percent must be in (0, 1], got {damp_percent}"
            )
        if num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")

        self.block_size = block_size
        self.damp_percent = damp_percent
        self.act_order = act_order
        self.num_batches = num_batches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        qmodel: nn.Module,
        calib_data: Any,
        *,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run GPTQ weight quantization on every eligible module.

        Args:
            qmodel: Quantized model (from ``quantize_model``).
            calib_data: Calibration data.  When ``eval_fn`` is None, must
                be ``List[Tensor]`` for direct iteration.  When ``eval_fn``
                is provided, can be any type the user's function accepts.
            eval_fn: ``(model, data) -> Any``.  Controls model interaction
                during input collection.  When None, falls back to
                ``for batch in calib_data[:num_batches]: model(batch)``.

        Returns:
            Dict mapping module name to metadata dict with keys:
            ``"mse_before"``, ``"mse_after"`` (weight MSE vs fp32 before
            and after GPTQ).
        """
        modules = _get_quantized_modules(qmodel)
        results: Dict[str, Dict[str, float]] = {}

        for name, module in modules:
            weight_scheme = getattr(module.cfg, "weight", None)
            if weight_scheme is None:
                continue

            # Skip modules without weight (e.g. ReLU, LayerNorm, Softmax)
            if not hasattr(module, "weight") or module.weight is None:
                continue

            # Conv2d not supported in v1 — would need im2col reshape
            if isinstance(module, nn.Conv2d):
                continue

            W = module.weight.data
            if W.ndim != 2:
                continue  # only 2D weight (Linear) supported in v1

            # Collect calibration inputs for this layer
            inputs = self._collect_inputs(
                qmodel, module, calib_data, eval_fn=eval_fn,
            )
            if not inputs:
                continue

            X = torch.cat(
                [x.detach().reshape(-1, W.shape[1]) for x in inputs], dim=0,
            )

            mse_before = (W - quantize(W, weight_scheme)).pow(2).mean().item()

            W_q = self._gptq_quantize(W, X, weight_scheme)
            module.weight.data = W_q

            mse_after = (W - W_q).pow(2).mean().item()
            results[name] = {"mse_before": mse_before, "mse_after": mse_after}

        return results

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _gptq_quantize(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        scheme,
    ) -> torch.Tensor:
        """Apply GPTQ to a single Linear weight matrix.

        Args:
            W: Weight tensor, shape ``[out_features, in_features]``.
            X: Calibration inputs stacked, shape ``[N, in_features]``.
            scheme: ``QuantScheme`` from ``module.cfg.weight``.

        Returns:
            GPTQ-quantized weight tensor, same shape as ``W``.
        """
        in_features = W.shape[1]
        device = W.device
        dtype = W.dtype

        # 1. Hessian: H = 2 * Xᵀ @ X  (in float32 for numerical stability)
        X_f32 = X.to(device=device, dtype=torch.float32)
        H = 2.0 * X_f32.T @ X_f32  # [in_features, in_features]

        # Damping: fraction of mean diagonal
        diag_mean = torch.mean(torch.diag(H))
        damp = self.damp_percent * diag_mean
        H += damp * torch.eye(in_features, device=device, dtype=torch.float32)

        # 2. H_inv via Cholesky (most numerically stable); fall back to pinv
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            H_inv = torch.linalg.pinv(H)

        # 3. Optional activation ordering
        W_f32 = W.to(dtype=torch.float32)
        if self.act_order:
            diag = torch.diag(H_inv)
            perm = torch.argsort(diag, descending=True)
            inv_perm = torch.argsort(perm)
            W_f32 = W_f32[:, perm]
            H_inv = H_inv[perm][:, perm]
        else:
            inv_perm = None

        # 4. Pre-compute per-channel scale from the FULL weight matrix so
        #    every column block uses the same amax.  Without this, each
        #    block would compute its own amax from the block subset,
        #    breaking idempotency with the forward pass.
        full_scale = _precompute_scale(W_f32, scheme)

        block_size = min(self.block_size, in_features)
        W_q = W_f32.clone()

        for i1 in range(0, in_features, block_size):
            i2 = min(i1 + block_size, in_features)

            # Quantize this block of columns
            W_block = W_f32[:, i1:i2]  # [out, block_size]
            with torch.no_grad():
                # Slice scale to match the block when channel axis
                # aligns with the column-slicing dimension
                if full_scale is None:
                    kwargs = {}
                elif full_scale.shape[-1] == in_features:
                    kwargs = {"scale": full_scale[..., i1:i2]}
                else:
                    kwargs = {"scale": full_scale}
                Q_block = quantize(W_block, scheme, **kwargs)
            err = Q_block - W_block      # [out, block_size]
            W_q[:, i1:i2] = Q_block

            # Update remaining columns to compensate
            if i2 < in_features:
                for j_rel in range(i2 - i1):
                    col = i1 + j_rel
                    # H_inv[col, col] is a scalar
                    h_diag = H_inv[col, col]
                    if h_diag < 1e-30:
                        continue  # effectively zero — skip compensation
                    factor = H_inv[col, i2:] / h_diag  # [in_features - i2]
                    # err[:, j_rel] is [out_features]
                    # Broadcast: [out, 1] * [1, in_features-i2]
                    W_f32[:, i2:] -= err[:, j_rel : j_rel + 1] * factor.unsqueeze(0)

        if inv_perm is not None:
            W_q = W_q[:, inv_perm]

        return W_q.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Calibration input collection
    # ------------------------------------------------------------------

    def _collect_inputs(
        self,
        qmodel: nn.Module,
        module: nn.Module,
        calib_data: Any,
        *,
        eval_fn: Optional[Callable] = None,
    ) -> List[torch.Tensor]:
        """Collect inputs to *module* by running the quantized model.

        Uses a forward hook on *module* to capture ``inp[0]``, then
        runs the model over calibration data.  The quantized model's
        preceding layers include quantization noise, so these are
        "real" inputs the layer will see at inference time.
        """
        inputs: List[torch.Tensor] = []

        def hook(_mod, inp, _out):
            inputs.append(inp[0].detach().clone())

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                if eval_fn is not None:
                    eval_fn(qmodel, calib_data)
                else:
                    for batch in calib_data[: self.num_batches]:
                        qmodel(batch)
        finally:
            handle.remove()

        return inputs
