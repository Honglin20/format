"""LayerwiseScaleOptimizer: gradient-based per-layer pre-scale optimization.

Path A (BRECQ-style): sequential optimization using real quantized intermediates.
For each layer, runs the partially-quantized model to get true inputs,
then optimizes pre-scale via gradient descent to minimize MSE against
fp32 layer output.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.transform.pre_scale import PreScaleTransform


def _get_quantized_modules(model: nn.Module) -> List[tuple]:
    """Return [(name, module), ...] for all Quantized* modules with cfg."""
    result = []
    for name, module in model.named_modules():
        if hasattr(module, "cfg") and not getattr(module, "_is_passthrough", False):
            result.append((name, module))
    return result


def _replace_transform(cfg: OpQuantConfig, transform) -> OpQuantConfig:
    """Return a new OpQuantConfig with *transform* replacing all non-None schemes."""
    fields = {}
    for f_name in cfg.__dataclass_fields__:
        old = getattr(cfg, f_name)
        if old is not None and isinstance(old, QuantScheme):
            fields[f_name] = QuantScheme(
                format=old.format,
                granularity=old.granularity,
                transform=transform,
                round_mode=old.round_mode,
            )
        else:
            fields[f_name] = old
    return OpQuantConfig(**fields)


class LayerwiseScaleOptimizer:
    """Gradient-based per-layer pre-scale optimization.

    Uses Path A (sequential quantized intermediates):
    1. Collect fp32 layer outputs from fp32 model
    2. For each layer: get real quantized inputs, then gradient-optimize
       pre_scale: minimize MSE(y_fp32, y_quant)
    3. Freeze the optimized pre_scale before moving to the next layer

    Args:
        num_steps: Optimization steps per layer (default: 100).
        num_batches: Number of calibration batches to use (default: 8).
        optimizer: Optimizer name — "adam" or "sgd" (default: "adam").
        lr: Learning rate (default: 1e-3).
        loss: Loss function — "mse" (default: "mse").
    """

    _VALID_OPTIMIZERS = {"adam", "sgd"}
    _VALID_LOSSES = {"mse"}

    def __init__(
        self,
        num_steps: int = 100,
        num_batches: int = 8,
        optimizer: str = "adam",
        lr: float = 1e-3,
        loss: str = "mse",
    ):
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if optimizer not in self._VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {self._VALID_OPTIMIZERS}, got {optimizer!r}"
            )
        if loss not in self._VALID_LOSSES:
            raise ValueError(
                f"loss must be one of {self._VALID_LOSSES}, got {loss!r}"
            )

        self.num_steps = num_steps
        self.num_batches = num_batches
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss

    def optimize(
        self,
        qmodel: nn.Module,
        fp32_model: nn.Module,
        calib_batches: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run layer-wise LSQ optimization.

        Args:
            qmodel: Quantized model (from quantize_model).
            fp32_model: Original fp32 model.
            calib_batches: List of input tensors from calibration.

        Returns:
            Dict mapping module name -> optimized pre_scale tensor.
        """
        modules = _get_quantized_modules(qmodel)
        batches = calib_batches[:self.num_batches]
        optimized_scales: Dict[str, torch.Tensor] = {}

        # Collect fp32 targets for all layers in one pass
        fp32_targets = self._collect_fp32_targets(
            qmodel, fp32_model, modules, batches
        )

        for layer_idx, (name, module) in enumerate(modules):
            targets = fp32_targets.get(name)
            if not targets:
                continue

            # Initial scale from fp32 targets (per-channel absmax)
            init_scale = self._compute_initial_scale(targets, module)
            pre_scale = nn.Parameter(init_scale)

            # Temporarily inject PreScaleTransform into module's cfg
            original_cfg = module.cfg
            transform = PreScaleTransform(scale=pre_scale)
            module.cfg = _replace_transform(original_cfg, transform)

            # Get real inputs from partially-quantized model
            real_inputs = self._get_layer_inputs(qmodel, module, batches)

            # Build optimizer
            if self.optimizer == "adam":
                opt = torch.optim.Adam([pre_scale], lr=self.lr)
            else:
                opt = torch.optim.SGD([pre_scale], lr=self.lr)

            loss_fn = nn.MSELoss()

            # Optimize
            module.train()
            for _ in range(self.num_steps):
                for x_in, y_fp32 in zip(real_inputs, targets):
                    opt.zero_grad()
                    y_q = module(x_in)
                    loss = loss_fn(y_q, y_fp32)
                    loss.backward()
                    opt.step()
            module.eval()

            # Freeze: store optimized scale
            optimized_scale = pre_scale.detach()
            optimized_scales[name] = optimized_scale

            # Replace cfg transform with frozen PreScaleTransform
            frozen_transform = PreScaleTransform(scale=optimized_scale)
            module.cfg = _replace_transform(original_cfg, frozen_transform)

            # Persist as module buffer for inference
            module.register_buffer("_pre_scale", optimized_scale)

        return optimized_scales

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_initial_scale(targets, module) -> torch.Tensor:
        """Compute initial pre-scale (identity = ones).

        Returns a single-element tensor that broadcasts to any shape.
        Per-channel initialization can be added later.
        """
        return torch.ones(1)

    def _collect_fp32_targets(
        self, qmodel, fp32_model, modules, batches
    ) -> Dict[str, List[torch.Tensor]]:
        """Collect fp32 layer outputs for all quantized layers using hooks."""
        targets: Dict[str, List[torch.Tensor]] = {}
        hooks = []
        fp32_lookup = dict(fp32_model.named_modules())

        for qname, _ in modules:
            fp32_mod = fp32_lookup.get(qname)
            if fp32_mod is None:
                continue
            targets[qname] = []

            def _make_hook(name):
                def hook(_module, _inp, out):
                    targets[name].append(out.detach().clone())

                return hook

            hooks.append(fp32_mod.register_forward_hook(_make_hook(qname)))

        try:
            with torch.no_grad():
                for batch in batches:
                    fp32_model(batch)
        finally:
            for h in hooks:
                h.remove()

        return targets

    def _get_layer_inputs(
        self, qmodel, module, batches
    ) -> List[torch.Tensor]:
        """Get real inputs to a module by running qmodel with a forward hook."""
        inputs: List[torch.Tensor] = []

        def hook(_module, inp, _out):
            inputs.append(inp[0].detach().clone())

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for batch in batches:
                    qmodel(batch)
        finally:
            handle.remove()

        return inputs
