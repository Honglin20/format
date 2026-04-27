"""
CalibrationSession: context manager for calibration data collection.

Replaces the old DataLoader-driven calibrate() pattern with a context
manager that the user controls.  The old CalibrationPipeline is kept
as a thin compatibility wrapper.

Design:

  1. __enter__ registers forward hooks on all Quantized* modules.
  2. User runs forward passes manually inside ``with`` block.
  3. __exit__ removes hooks, computes scales via strategy, and (by
     default) auto-assigns them as ``_output_scale`` buffers.

  Scales can also be inspected mid-collection via :meth:`scales`.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.calibration.strategies import ScaleStrategy


class CalibrationSession:
    """Context manager for activation-scale calibration.

    Args:
        model: PyTorch model (typically the quantized model from QuantSession).
        strategy: ``ScaleStrategy`` instance used to compute final scales.
        axis: Dimension along which per-slice statistics are tracked.
        assign: If True (default), scales are auto-assigned as module
            buffers on context exit.  Set False to only collect without
            modifying the model.

    Example::

        with CalibrationSession(model, MaxScaleStrategy()) as calib:
            for batch in calib_data:
                model(batch)
        # Scales are auto-assigned on exit — model is now calibrated.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: ScaleStrategy,
        axis: int = -1,
        assign: bool = True,
    ):
        self.model = model
        self.strategy = strategy
        self.axis = axis
        self._assign = assign
        self._running_amax: Dict[str, torch.Tensor] = {}
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "CalibrationSession":
        self._running_amax.clear()
        for name, module in self.model.named_modules():
            if hasattr(module, "cfg"):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
        return self

    def __exit__(self, *args):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._assign and self._running_amax:
            s = self.scales()
            self._assign_scales(s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scales(self) -> Dict[str, torch.Tensor]:
        """Compute and return scale factors from collected statistics.

        Can be called inside or after the ``with`` block.  Each call
        re-computes from the current running-amax state.
        """
        scales: Dict[str, torch.Tensor] = {}
        for name, amax in self._running_amax.items():
            scales[name] = self.strategy.compute(amax, self.axis)
        return scales

    def assign_scales(self, scales: Optional[Dict[str, torch.Tensor]] = None) -> List[str]:
        """Register scales as ``_output_scale`` buffers on model modules.

        Args:
            scales: Dict mapping module names to scale tensors.
                If None, calls :meth:`scales` internally.

        Returns:
            List of module names that were successfully assigned.
        """
        if scales is None:
            scales = self.scales()
        return self._assign_scales(scales)

    def clear_scales(self) -> List[str]:
        """Remove all ``_output_scale`` buffers from the model.

        Returns:
            List of module names from which buffers were removed.
        """
        removed = []
        module_map = dict(self.model.named_modules())
        for name, module in module_map.items():
            if hasattr(module, "_output_scale"):
                del module._output_scale
                removed.append(name)
        return removed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assign_scales(self, scales: Dict[str, torch.Tensor]) -> List[str]:
        assigned = []
        module_map = dict(self.model.named_modules())
        for name, scale in scales.items():
            if name in module_map:
                module_map[name].register_buffer("_output_scale", scale)
                assigned.append(name)
        return assigned

    def _make_hook(self, name: str):
        def _hook(module, _input, output):
            x = output.detach()
            amax = torch.amax(torch.abs(x), dim=self.axis, keepdim=True)
            if name in self._running_amax:
                self._running_amax[name] = torch.max(
                    self._running_amax[name], amax
                )
            else:
                self._running_amax[name] = amax
        return _hook


# ------------------------------------------------------------------
# Backward-compatible wrapper
# ------------------------------------------------------------------

class CalibrationPipeline(CalibrationSession):
    """Legacy DataLoader-driven pipeline — kept for backward compatibility.

    Prefer :class:`CalibrationSession` for new code.
    """

    def __init__(self, model, strategy, num_batches=64, axis=-1):
        super().__init__(model, strategy, axis=axis, assign=False)
        self.num_batches = num_batches

    def calibrate(self, dataloader) -> Dict[str, torch.Tensor]:
        """Run calibration over *dataloader* and return per-layer scales.

        Legacy wrapper — opens a context-manager session internally.
        """
        with self:
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= self.num_batches:
                        break
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    self.model(inputs)
        return self.scales()
