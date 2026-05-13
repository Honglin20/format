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
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.calibration.strategies import ScaleStrategy
from src.scheme.granularity import GranularityMode


class CalibrationSession:
    """Context manager for activation-scale calibration.

    Args:
        model: PyTorch model (typically the quantized model from _QuantSession).
        strategy: ``ScaleStrategy`` instance used to compute final scales.
        axis: Dimension along which per-slice statistics are tracked.
        assign: If True (default), scales are auto-assigned as module
            buffers on context exit.  Set False to only collect without
            modifying the model.

    Example::

        with CalibrationSession(model, MaxScaleStrategy()) as calib:
            eval_fn(model, calib_data)
        # Scales are auto-assigned on exit — model is now calibrated.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: ScaleStrategy,
        axis: int = -1,
        assign: bool = True,
        track_input: bool = False,
    ):
        self.model = model
        self.strategy = strategy
        self.axis = axis
        self._assign = assign
        self._track_input = track_input
        self._running_amax: Dict[str, torch.Tensor] = {}
        self._running_input_amax: Dict[str, torch.Tensor] = {}
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
        if self._assign:
            if self._running_amax:
                self._assign_scales(self.scales())
            if self._track_input and self._running_input_amax:
                self._assign_input_scales(self.input_scales())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scales(self) -> Dict[str, torch.Tensor]:
        """Compute and return scale factors from collected statistics.

        Can be called inside or after the ``with`` block.  Each call
        re-computes from the current running-amax state.

        The running amax is already correctly shaped (scalar for PER_TENSOR,
        (C,) for PER_CHANNEL).  Strategies are not re-applied here because
        they expect raw-data tensors, not pre-computed amax.
        """
        scales: Dict[str, torch.Tensor] = {}
        for name, amax in self._running_amax.items():
            scales[name] = amax.clamp(min=1e-12)
        return scales

    def input_scales(self) -> Dict[str, torch.Tensor]:
        """Return input amax scale factors (only when ``track_input=True``)."""
        scales: Dict[str, torch.Tensor] = {}
        for name, amax in self._running_input_amax.items():
            scales[name] = amax.clamp(min=1e-12)
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

    def assign_input_scales(self, scales: Optional[Dict[str, torch.Tensor]] = None) -> List[str]:
        """Register input scales as ``_input_scale`` buffers.

        Args:
            scales: Dict mapping module names to scale tensors.
                If None, calls :meth:`input_scales` internally.

        Returns:
            List of module names that were successfully assigned.
        """
        if scales is None:
            scales = self.input_scales()
        return self._assign_input_scales(scales)

    def save_scales(self, filepath: str, scales: Optional[Dict[str, torch.Tensor]] = None) -> str:
        """Save scale factors to disk.

        .. deprecated::
           Use the standalone ``save_scales(scales, filepath)`` function instead.

        Args:
            filepath: Path to save the scales dict (e.g. ``"scales.pt"``).
            scales: Dict mapping module names to scale tensors.
                If None, calls :meth:`scales` internally.

        Returns:
            The filepath (for chaining).
        """
        if scales is None:
            scales = self.scales()
        torch.save(scales, filepath)
        return filepath

    def load_scales(
        self,
        filepath: str,
        assign: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Load scales from disk and optionally assign to model modules.

        .. deprecated::
           Use the standalone ``load_scales(filepath)`` function instead.

        Args:
            filepath: Path to the saved scales file (e.g. ``"scales.pt"``).
            assign: If True (default), assign scales as ``_output_scale``
                buffers on corresponding model modules.

        Returns:
            Dict mapping module names to scale tensors.
        """
        scales = torch.load(filepath, weights_only=False)
        if assign:
            self._assign_scales(scales)
        return scales

    @staticmethod
    def load_scales_from(filepath: str) -> Dict[str, torch.Tensor]:
        """Load scales from disk (standalone, no model required).

        .. deprecated::
           Use the standalone ``load_scales(filepath)`` function instead.

        Args:
            filepath: Path to the saved scales file.

        Returns:
            Dict mapping module names to scale tensors.
        """
        return torch.load(filepath, weights_only=False)

    def clear_scales(self) -> List[str]:
        """Remove all ``_output_scale`` and ``_input_scale`` buffers from the model.

        Returns:
            List of module names from which buffers were removed.
        """
        removed = []
        module_map = dict(self.model.named_modules())
        for name, module in module_map.items():
            for buf_name in ("_output_scale", "_input_scale"):
                if hasattr(module, buf_name):
                    delattr(module, buf_name)
                    if name not in removed:
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

    def _assign_input_scales(self, scales: Dict[str, torch.Tensor]) -> List[str]:
        assigned = []
        module_map = dict(self.model.named_modules())
        for name, scale in scales.items():
            if name in module_map:
                module_map[name].register_buffer("_input_scale", scale)
                assigned.append(name)
        return assigned

    def _make_hook(self, name: str):
        def _hook(module, _input, output):
            x = output.detach()
            # Compute amax with shape determined by output granularity.
            # PER_TENSOR: scalar. PER_CHANNEL: (C,) — reduce all dims
            # except channel_axis so each channel has its own scale.
            mode, channel_axis = self._output_granularity(module)
            if mode == GranularityMode.PER_TENSOR:
                amax = torch.amax(torch.abs(x))
            elif mode == GranularityMode.PER_CHANNEL:
                ax = channel_axis if channel_axis >= 0 else x.ndim + channel_axis
                dims_to_reduce = [i for i in range(x.ndim) if i != ax]
                amax = torch.amax(
                    torch.abs(x), dim=tuple(dims_to_reduce), keepdim=True,
                )
            else:
                # PER_BLOCK / DYNAMIC_GROUP: per-element or caller-driven
                amax = torch.amax(torch.abs(x), dim=self.axis, keepdim=True)

            if name in self._running_amax:
                self._running_amax[name] = torch.max(
                    self._running_amax[name], amax
                )
            else:
                self._running_amax[name] = amax

            # Track input amax when static_input_scale is requested
            if self._track_input and _input and isinstance(_input[0], torch.Tensor):
                inp = _input[0].detach()
                imode, ich_axis = self._input_granularity(module)
                if imode is not None:
                    if imode == GranularityMode.PER_TENSOR:
                        inp_amax = torch.amax(torch.abs(inp))
                    elif imode == GranularityMode.PER_CHANNEL:
                        ax = ich_axis if ich_axis >= 0 else inp.ndim + ich_axis
                        dims = [i for i in range(inp.ndim) if i != ax]
                        inp_amax = torch.amax(
                            torch.abs(inp), dim=tuple(dims), keepdim=True,
                        )
                    else:
                        inp_amax = torch.amax(torch.abs(inp), dim=self.axis, keepdim=True)

                    if name in self._running_input_amax:
                        self._running_input_amax[name] = torch.max(
                            self._running_input_amax[name], inp_amax,
                        )
                    else:
                        self._running_input_amax[name] = inp_amax
        return _hook

    @staticmethod
    def _output_granularity(module):
        """Return (mode, channel_axis) for the module's output quant scheme."""
        if hasattr(module, "cfg") and module.cfg.output is not None:
            g = module.cfg.output.granularity
            return g.mode, g.channel_axis
        return GranularityMode.PER_TENSOR, 0

    @staticmethod
    def _input_granularity(module):
        """Return (mode, channel_axis) for the module's input quant scheme.

        Returns (None, 0) when input scheme is absent or uses a mode
        where static scale is not applicable (per_block MX, float formats).
        """
        if not hasattr(module, "cfg") or module.cfg.input is None:
            return None, 0
        s = module.cfg.input
        g = s.granularity
        if g.mode == GranularityMode.PER_BLOCK:
            return None, 0
        if s.format.ebits > 0:
            return None, 0
        return g.mode, g.channel_axis


# ------------------------------------------------------------------
# Standalone persistence helpers (no session/model required)
# ------------------------------------------------------------------


def save_scales(scales: Dict[str, torch.Tensor], filepath: str) -> str:
    """Save a scales dict to disk.

    Args:
        scales: Dict mapping module names to scale tensors.
        filepath: Path to save (e.g. ``"scales.pt"``).

    Returns:
        The filepath (for chaining).
    """
    torch.save(scales, filepath)
    return filepath


def load_scales(filepath: str) -> Dict[str, torch.Tensor]:
    """Load a scales dict from disk.

    Args:
        filepath: Path to the saved scales file.

    Returns:
        Dict mapping module names to scale tensors.
    """
    return torch.load(filepath, weights_only=False)


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

    def calibrate(self, dataloader, *, eval_fn: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
        """Run calibration over *dataloader* and return per-layer scales.

        Legacy wrapper — opens a context-manager session internally.

        Args:
            dataloader: Iterable of batches.
            eval_fn: ``(model, data) -> Any``. Controls how the model is
                called during calibration. When None, falls back to
                ``self.model(inputs)`` for each batch.
        """
        with self:
            with torch.no_grad():
                if eval_fn is not None:
                    eval_fn(self.model, dataloader)
                else:
                    for i, batch in enumerate(dataloader):
                        if i >= self.num_batches:
                            break
                        if isinstance(batch, (list, tuple)):
                            inputs = batch[0]
                        else:
                            inputs = batch
                        self.model(inputs)
        return self.scales()
