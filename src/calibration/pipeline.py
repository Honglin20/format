"""
CalibrationPipeline: iterate calibration data, collect activation statistics,
and compute scale factors using a pluggable ScaleStrategy.

Design (v1 -- running-amax based):

  1. Register forward hooks on all model modules that have a ``cfg``
     attribute (i.e. ``QuantizedXxx`` layers).
  2. Walk the DataLoader up to ``num_batches``, running each batch through
     the model under ``torch.no_grad()``.
  3. For each forward call, capture the output activation and update a
     running absolute-maximum (``running_amax``) per layer.  The running
     max is taken element-wise across batches.
  4. Remove all hooks.
  5. Apply ``strategy.compute(running_amax, axis)`` to obtain the final
     per-layer scale tensor.

  The returned dictionary maps module names (as seen by ``named_modules()``)
  to scale tensors.

  Future versions may add histogram-based collection for MSE/KL strategies
  on full activation tensors.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.calibration.strategies import ScaleStrategy


class CalibrationPipeline:
    """Calibration data collector and scale-factor computer.

    Args:
        model: PyTorch model to calibrate (any ``nn.Module``).
        strategy: ``ScaleStrategy`` instance used to compute final scales
            from collected statistics.
        num_batches: Maximum number of DataLoader batches to process.
            Default 64.
        axis: Dimension along which per-slice statistics are tracked and
            scales are computed.  Supports NumPy-style negative indexing.
            Default -1 (last dimension).

    Example::

        pipeline = CalibrationPipeline(model, MaxScaleStrategy(), num_batches=8)
        scales = pipeline.calibrate(dataloader)
        # scales = {"layer_name": scale_tensor, ...}
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: ScaleStrategy,
        num_batches: int = 64,
        axis: int = -1,
    ):
        self.model = model
        self.strategy = strategy
        self.num_batches = num_batches
        self.axis = axis
        # Internal state: running absolute max per layer
        self._running_amax: Dict[str, torch.Tensor] = {}

    def assign_scales(self, scales: Dict[str, torch.Tensor]) -> list:
        """Assign pre-computed scale tensors to model modules as buffers.

        For each ``(name, scale)`` in *scales*, looks up the module by
        name via ``model.named_modules()`` and registers the scale as a
        persistent buffer named ``_output_scale``.  Registered buffers
        survive ``state_dict()`` / ``load_state_dict()`` round-trips.

        Args:
            scales: Dictionary mapping module names to scale tensors,
                as returned by :meth:`calibrate`.

        Returns:
            List of module names that were successfully assigned.
        """
        assigned = []
        module_map = dict(self.model.named_modules())
        for name, scale in scales.items():
            if name in module_map:
                module_map[name].register_buffer("_output_scale", scale)
                assigned.append(name)
        return assigned

    def calibrate(self, dataloader) -> Dict[str, torch.Tensor]:
        """Run calibration and return per-layer scale factors.

        Iterates through ``dataloader``, running forward passes under
        ``torch.no_grad()``.  Collects activation statistics from every
        module that has a ``cfg`` attribute (i.e. quantized layers).

        Args:
            dataloader: A PyTorch ``DataLoader`` yielding batches.
                Each batch may be:
                - A ``(input,)`` or ``(input, target, ...)`` tuple
                - A plain ``torch.Tensor``

        Returns:
            Dictionary mapping module names (``str``) to scale tensors
            (``torch.Tensor``).  Modules without a ``cfg`` attribute are
            skipped and do not appear in the output.
        """
        # Reset internal state from any previous calibration call
        self._running_amax.clear()

        # 1. Register forward hooks on quantized layers
        hooks = []
        for name, module in self.model.named_modules():
            if hasattr(module, "cfg"):
                hook = module.register_forward_hook(self._make_hook(name))
                hooks.append(hook)

        if not hooks:
            return {}

        try:
            # 2. Iterate through calibration data
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= self.num_batches:
                        break

                    # Unpack the batch: handle both tensor and tuple/list inputs
                    if isinstance(batch, (list, tuple)):
                        # (input,) or (input, target, ...)
                        inputs = batch[0]
                    else:
                        inputs = batch

                    # Run forward pass (hooks capture activations)
                    self.model(inputs)
        finally:
            # 3. Always clean up hooks, even if forward pass raises
            for hook in hooks:
                hook.remove()

        # 4. If no data was processed, no hooks fired → empty result
        if not self._running_amax:
            return {}

        # 5. Compute final scales from collected statistics
        scales: Dict[str, torch.Tensor] = {}
        for name, amax in self._running_amax.items():
            scales[name] = self.strategy.compute(amax, self.axis)

        return scales

    def _make_hook(self, name: str):
        """Create a forward hook that tracks running amax for *name*."""

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
