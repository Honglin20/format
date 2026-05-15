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
  4. When ``sparse=True``, per-sample activations are collected and
     ``compute_sparse_mask()`` is called on exit to produce static
     sparse masks and per-group scales.

  Scales can also be inspected mid-collection via :meth:`scales`.
"""
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.calibration.strategies import ScaleStrategy
from src.scheme.granularity import GranularityMode, GranularitySpec


# ---------------------------------------------------------------------------
# Static sparse scale computation (ADR-012)
# ---------------------------------------------------------------------------


def _adjust_gran_axes_for_calibration(gran):
    """Return a GranularitySpec with axes adjusted for calibration samples.

    Calibration samples (after :func:`torch.cat` flattening the batch dim) are
    one rank lower than the inference-time tensors that the spec's axes target.
    Positive axis indices must be decremented by 1 to point to the same
    semantic dimension.  Negative indices auto-adjust with rank and are left
    unchanged.

    This is only needed for sparse calibration (mask and per-group scale
    computation), which processes individual per-sample tensors without a
    batch dimension.
    """
    if gran is None:
        return gran
    bank_axis = gran.bank_axis - 1 if gran.bank_axis > 0 else gran.bank_axis
    channel_axis = gran.channel_axis - 1 if gran.channel_axis > 0 else gran.channel_axis
    block_axis = gran.block_axis - 1 if gran.block_axis > 0 else gran.block_axis
    return GranularitySpec(
        mode=gran.mode,
        block_size=gran.block_size,
        channel_axis=channel_axis,
        block_axis=block_axis,
        bank_size=gran.bank_size,
        bank_axis=bank_axis,
        outlier_ratio=gran.outlier_ratio,
    )


def _compute_sparse_scales(
    x_calib: torch.Tensor,
    mask: torch.Tensor,
    gran,
) -> tuple:
    """Compute per-group amax scales from calibration data and a fixed mask.

    Args:
        x_calib: Calibration samples stacked along dim 0 (S, D1, D2, ...).
        mask: Boolean mask with shape (D1, D2, ...), True = outlier.
        gran: GranularitySpec with outlier_ratio > 0.

    Returns:
        (scale_n, scale_o) — amax tensors with shapes matching the
        granularity mode (scalar for per_tensor, per-channel shape for
        per_channel, etc.).
    """
    S = x_calib.shape[0]
    mode = gran.mode

    normal_amax = None
    outlier_amax = None
    for s in range(S):
        x_s = x_calib[s]
        n_a = _compute_group_amax(x_s, mask, invert=True, mode=mode, gran=gran)
        o_a = _compute_group_amax(x_s, mask, invert=False, mode=mode, gran=gran)
        normal_amax = torch.max(normal_amax, n_a) if normal_amax is not None else n_a
        outlier_amax = torch.max(outlier_amax, o_a) if outlier_amax is not None else o_a

    return normal_amax.clamp(min=1e-12), outlier_amax.clamp(min=1e-12)


def _compute_group_amax(x, mask, invert, mode, gran):
    """Compute amax of masked values for a given group, respecting granularity.

    Args:
        x: Single-sample tensor.
        mask: Boolean mask, True = outlier.
        invert: If True, compute for normal group (mask=False elements).
                If False, compute for outlier group (mask=True elements).
        mode: GranularityMode.
        gran: GranularitySpec.

    Returns:
        amax tensor with shape determined by granularity.
    """
    sel = (~mask) if invert else mask
    x_sel = x * sel.float()

    if mode == GranularityMode.PER_TENSOR:
        return torch.amax(torch.abs(x_sel))

    if mode == GranularityMode.PER_CHANNEL:
        axis = gran.channel_axis
        if axis < 0:
            axis = x_sel.ndim + axis
        dims_to_reduce = [i for i in range(x_sel.ndim) if i != axis]
        if not dims_to_reduce:
            return torch.abs(x_sel)
        return torch.amax(torch.abs(x_sel), dim=tuple(dims_to_reduce), keepdim=True)

    if mode == GranularityMode.BANK:
        axis = gran.bank_axis
        if axis < 0:
            axis = x_sel.ndim + axis
        bank_size = gran.bank_size
        N_along = x_sel.shape[axis]
        num_banks = N_along // bank_size
        new_shape = list(x_sel.shape)
        new_shape[axis] = num_banks
        new_shape.insert(axis + 1, bank_size)
        x_r = x_sel.reshape(new_shape)
        dims_to_reduce = [i for i in range(x_r.ndim) if i != axis]
        if not dims_to_reduce:
            return torch.abs(x_r)
        return torch.amax(torch.abs(x_r), dim=tuple(dims_to_reduce), keepdim=True)

    raise ValueError(f"Unsupported granularity mode for static sparse: {mode}")


# ---------------------------------------------------------------------------
# CalibrationSession
# ---------------------------------------------------------------------------


class CalibrationSession:
    """Context manager for activation-scale calibration.

    Args:
        model: PyTorch model (typically the quantized model from quantize_model()).
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
        sparse: bool = False,
        sq_mode: Optional[str] = None,
    ):
        self.model = model
        self.strategy = strategy
        self.axis = axis
        self._assign = assign
        self._track_input = track_input
        self._sparse = sparse
        self._sq_mode = sq_mode  # "weight" or "activation_static"
        self._running_amax: Dict[str, torch.Tensor] = {}
        self._running_input_amax: Dict[str, torch.Tensor] = {}
        self._output_samples: Dict[str, List[torch.Tensor]] = {}
        self._input_samples: Dict[str, List[torch.Tensor]] = {}
        self._sq_inputs: Dict[str, List[torch.Tensor]] = {}
        self._sq_outputs: Dict[str, List[torch.Tensor]] = {}
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "CalibrationSession":
        self._running_amax.clear()
        self._output_samples.clear()
        self._input_samples.clear()
        self._sq_inputs.clear()
        self._sq_outputs.clear()
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
            if self._sparse:
                self._compute_and_assign_sparse_state()
                self._compute_and_assign_group_sparse_state()
            if self._sq_mode:
                self._compute_and_assign_sq_state()

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
            # BANK: reshape to expose bank dim, reduce all except bank dim.
            mode, channel_axis, gran = self._output_granularity(module)
            if mode == GranularityMode.PER_TENSOR:
                amax = torch.amax(torch.abs(x))
            elif mode == GranularityMode.PER_CHANNEL:
                ax = channel_axis if channel_axis >= 0 else x.ndim + channel_axis
                dims_to_reduce = [i for i in range(x.ndim) if i != ax]
                amax = torch.amax(
                    torch.abs(x), dim=tuple(dims_to_reduce), keepdim=True,
                )
            elif mode == GranularityMode.BANK and gran is not None:
                amax = self._compute_bank_amax(x, gran)
            else:
                # PER_BLOCK / DYNAMIC_GROUP: per-element or caller-driven
                amax = torch.amax(torch.abs(x), dim=self.axis, keepdim=True)

            if name in self._running_amax:
                self._running_amax[name] = torch.max(
                    self._running_amax[name], amax
                )
            else:
                self._running_amax[name] = amax

            # Collect per-sample outputs for static sparse (ADR-012 + ADR-013)
            if self._sparse and gran is not None:
                scheme = getattr(module.cfg, "output", None)
                needs_sparse = gran.outlier_ratio > 0.0
                needs_group = (scheme is not None
                               and scheme.group_format is not None
                               and scheme.group_ratio > 0.0)
                if needs_sparse or needs_group:
                    if name not in self._output_samples:
                        self._output_samples[name] = []
                    self._output_samples[name].append(x.clone())

            # Track input amax when static_input_scale is requested
            if self._track_input and _input and isinstance(_input[0], torch.Tensor):
                inp = _input[0].detach()
                imode, ich_axis, igran = self._input_granularity(module)
                if imode is not None:
                    if imode == GranularityMode.PER_TENSOR:
                        inp_amax = torch.amax(torch.abs(inp))
                    elif imode == GranularityMode.PER_CHANNEL:
                        ax = ich_axis if ich_axis >= 0 else inp.ndim + ich_axis
                        dims = [i for i in range(inp.ndim) if i != ax]
                        inp_amax = torch.amax(
                            torch.abs(inp), dim=tuple(dims), keepdim=True,
                        )
                    elif imode == GranularityMode.BANK and igran is not None:
                        inp_amax = self._compute_bank_amax(inp, igran)
                    else:
                        inp_amax = torch.amax(torch.abs(inp), dim=self.axis, keepdim=True)

                    if name in self._running_input_amax:
                        self._running_input_amax[name] = torch.max(
                            self._running_input_amax[name], inp_amax,
                        )
                    else:
                        self._running_input_amax[name] = inp_amax

                    # Collect per-sample inputs for static sparse (ADR-012 + ADR-013)
                    if self._sparse and igran is not None:
                        ischeme = getattr(module.cfg, "input", None)
                        ineeds_sparse = igran.outlier_ratio > 0.0
                        ineeds_group = (ischeme is not None
                                        and ischeme.group_format is not None
                                        and ischeme.group_ratio > 0.0)
                        if ineeds_sparse or ineeds_group:
                            if name not in self._input_samples:
                                self._input_samples[name] = []
                            self._input_samples[name].append(inp.clone())

            # SQ-format: collect inputs for Hessian (weight SQ) and outputs for
            # activation averages (activation_static SQ).
            if self._sq_mode and hasattr(module, "cfg"):
                w_scheme = getattr(module.cfg, "weight", None)
                a_scheme = getattr(module.cfg, "input", None)
                if (self._sq_mode == "weight"
                        and w_scheme is not None
                        and w_scheme.sq_importance
                        and _input and isinstance(_input[0], torch.Tensor)):
                    inp = _input[0].detach()
                    if name not in self._sq_inputs:
                        self._sq_inputs[name] = []
                    self._sq_inputs[name].append(inp.clone())
                if (self._sq_mode == "activation_static"
                        and a_scheme is not None
                        and a_scheme.sq_importance):
                    x_out = output.detach()
                    if name not in self._sq_outputs:
                        self._sq_outputs[name] = []
                    self._sq_outputs[name].append(x_out.clone())
        return _hook

    # ------------------------------------------------------------------
    # Static sparse state (ADR-012)
    # ------------------------------------------------------------------

    def _compute_and_assign_sparse_state(self):
        """Compute static sparse masks and per-group scales from collected samples.

        For each module with collected output samples:
          1. Stack samples → call compute_sparse_mask()
          2. Compute per-group amax (max across samples)
          3. Store _output_mask, _output_scale (normal), _output_scale_o

        For each module with collected input samples:
          Same process → _input_mask, _input_scale, _input_scale_o

        Also computes weight sparse masks from the weight tensor directly.
        """
        module_map = dict(self.model.named_modules())

        for name, samples in self._output_samples.items():
            module = module_map.get(name)
            if module is None or not samples:
                continue
            gran, fmt, outlier_fmt = self._sparse_config(module, role="output")
            if gran is None:
                continue
            mask, scale_n, scale_o = self._compute_sparse_state(
                samples, fmt, gran,
            )
            module.register_buffer("_output_mask", mask)
            module.register_buffer("_output_scale", scale_n)
            module.register_buffer("_output_scale_o", scale_o)

        for name, samples in self._input_samples.items():
            module = module_map.get(name)
            if module is None or not samples:
                continue
            gran, fmt, outlier_fmt = self._sparse_config(module, role="input")
            if gran is None:
                continue
            mask, scale_n, scale_o = self._compute_sparse_state(
                samples, fmt, gran,
            )
            module.register_buffer("_input_mask", mask)
            module.register_buffer("_input_scale", scale_n)
            module.register_buffer("_input_scale_o", scale_o)

    @staticmethod
    def _sparse_config(module, role="output"):
        """Return (granularity, format, outlier_format) for sparse mask computation.

        Returns (None, None, None) when sparse is not applicable.
        """
        if not hasattr(module, "cfg"):
            return None, None, None
        scheme = getattr(module.cfg, role, None)
        if scheme is None:
            return None, None, None
        g = scheme.granularity
        if g.outlier_ratio <= 0.0:
            return None, None, None
        if g.mode == GranularityMode.PER_BLOCK:
            return None, None, None
        if scheme.format.ebits > 0:
            return None, None, None
        return g, scheme.format, scheme.outlier_format

    @staticmethod
    def _compute_sparse_state(samples, fmt, gran):
        """Compute mask, normal amax, and outlier amax from per-sample data.

        Args:
            samples: List of tensors, each the output of one forward pass.
                     Each tensor may include a batch dimension; all batch
                     elements are treated as independent calibration samples.
            fmt: FormatBase instance for the main format.
            gran: GranularitySpec with outlier_ratio > 0.

        Returns:
            (mask, scale_n, scale_o) tuple.
            mask has shape ``(1, D1, D2, ...)`` — a batch dim of size 1
            that broadcasts with any inference batch size.
        """
        from src.quantize._sparse_mask import compute_sparse_mask

        # Flatten per-batch samples into independent per-element samples.
        # samples = [ (B1, *spatial), (B2, *spatial), ... ]
        # → x_calib = (ΣBi, *spatial)
        x_calib = torch.cat(samples, dim=0)

        # Per-sample tensors inside this routine lack the batch dim; adjust
        # positive axis indices so they point to the same semantic dimension.
        calib_gran = _adjust_gran_axes_for_calibration(gran)

        mask = compute_sparse_mask(x_calib, fmt, calib_gran, calib_gran.outlier_ratio)
        # mask shape: (*spatial) — per-position, no batch dim

        scale_n, scale_o = _compute_sparse_scales(
            x_calib, mask, calib_gran,
        )

        # Add batch dim of size 1 so the mask broadcasts with any inference
        # batch size during forward (see _quantize_per_bank_static_sparse).
        mask = mask.unsqueeze(0)  # (1, *spatial)
        return mask, scale_n, scale_o

    # ------------------------------------------------------------------
    # SQ-format state (ADR-014)
    # ------------------------------------------------------------------

    def _compute_and_assign_sq_state(self):
        """Compute SQ-format artifacts from collected calibration data.

        sq_mode="weight": compute Hessian diagonal from inputs, then
            Hessian importance I = W²·H², store as _sq_importance buffer.
        sq_mode="activation_static": compute per-channel activation
            averages, then A·W channel importance, store as _sq_activation_mask.
        """
        module_map = dict(self.model.named_modules())

        if self._sq_mode == "weight":
            for name, inputs in self._sq_inputs.items():
                module = module_map.get(name)
                if module is None or not inputs:
                    continue
                if not hasattr(module, "weight") or module.weight is None:
                    continue
                w = module.weight.detach()
                hessian_diag = self._compute_hessian_diag(inputs)
                importance = self._compute_weight_importance(w, hessian_diag)
                module.register_buffer("_sq_importance", importance)

        elif self._sq_mode == "activation_static":
            for name, outputs in self._sq_outputs.items():
                module = module_map.get(name)
                if module is None or not outputs:
                    continue
                if not hasattr(module, "weight") or module.weight is None:
                    continue
                w = module.weight.detach()
                act_avg = self._compute_activation_average(outputs)
                mask = self._compute_activation_mask(act_avg, w)
                module.register_buffer("_sq_activation_mask", mask)

    @staticmethod
    def _compute_hessian_diag(inputs: list) -> torch.Tensor:
        """Compute Hessian diagonal from calibration inputs.

        H_i = mean(x²)_i over calibration set.
        """
        from src.formats._sq_importance import compute_hessian_diag_from_inputs
        return compute_hessian_diag_from_inputs(inputs)

    @staticmethod
    def _compute_weight_importance(weight: torch.Tensor,
                                    hessian_diag: torch.Tensor) -> torch.Tensor:
        """Compute I = W² · H² per-element weight importance."""
        from src.formats._sq_importance import compute_hessian_importance
        return compute_hessian_importance(weight, hessian_diag)

    @staticmethod
    def _compute_activation_average(outputs: list) -> torch.Tensor:
        """Compute per-channel average activation across calibration samples."""
        all_flat = []
        for x in outputs:
            x_flat = x.reshape(-1, x.shape[-1])
            all_flat.append(x_flat)
        stacked = torch.cat(all_flat, dim=0)
        return stacked.abs().mean(dim=0)  # |A| average per channel

    @staticmethod
    def _compute_activation_mask(act_avg: torch.Tensor,
                                  weight: torch.Tensor) -> torch.Tensor:
        """Compute per-channel importance scores for activation SQ.

        I_j = |Ā_j| · Σ_i |W_{j,i}| — channel importance.
        Returns boolean mask: True = high-precision channel.
        """
        from src.formats._sq_importance import compute_activation_channel_importance
        importance = compute_activation_channel_importance(act_avg, weight)
        # Top-50% channels by importance → high precision
        k = max(1, int(importance.numel() * 0.5))
        _, top_idx = torch.topk(importance, k)
        mask = torch.zeros(importance.shape, dtype=torch.bool)
        mask.scatter_(0, top_idx, True)
        return mask

    # ------------------------------------------------------------------
    # Static group sparse state (ADR-013)
    # ------------------------------------------------------------------

    def _compute_and_assign_group_sparse_state(self):
        """Compute static group masks from collected samples.

        For each module with collected output samples and group_format set:
          1. Stack samples → call compute_group_mask()
          2. Store _output_group_mask

        For each module with collected input samples:
          Same process → _input_group_mask
        """
        module_map = dict(self.model.named_modules())

        for name, samples in self._output_samples.items():
            module = module_map.get(name)
            if module is None or not samples:
                continue
            gran, group_ratio = self._group_sparse_config(module, role="output")
            if gran is None:
                continue
            group_mask = self._compute_group_sparse_state(samples, gran, group_ratio)
            module.register_buffer("_output_group_mask", group_mask)

        for name, samples in self._input_samples.items():
            module = module_map.get(name)
            if module is None or not samples:
                continue
            gran, group_ratio = self._group_sparse_config(module, role="input")
            if gran is None:
                continue
            group_mask = self._compute_group_sparse_state(samples, gran, group_ratio)
            module.register_buffer("_input_group_mask", group_mask)

    @staticmethod
    def _group_sparse_config(module, role="output"):
        """Return (granularity, group_ratio) for group mask computation.

        Returns (None, None) when group sparse is not applicable.
        """
        if not hasattr(module, "cfg"):
            return None, None
        scheme = getattr(module.cfg, role, None)
        if scheme is None:
            return None, None
        if scheme.group_format is None or scheme.group_ratio <= 0.0:
            return None, None
        return scheme.granularity, scheme.group_ratio

    @staticmethod
    def _compute_group_sparse_state(samples, gran, group_ratio):
        """Compute group mask from per-sample data.

        Args:
            samples: List of tensors, each the output of one forward pass.
            gran: GranularitySpec instance.
            group_ratio: Fraction of groups to mark as H.

        Returns:
            Boolean group_mask tensor.
        """
        from src.quantize._group_mask import compute_group_mask

        x_calib = torch.stack(samples, dim=0)  # (S, D1, D2, ...)
        return compute_group_mask(x_calib, gran, group_ratio)

    @staticmethod
    def _output_granularity(module):
        """Return (mode, channel_axis, granularity) for the module's output quant scheme."""
        if hasattr(module, "cfg") and module.cfg.output is not None:
            g = module.cfg.output.granularity
            return g.mode, g.channel_axis, g
        return GranularityMode.PER_TENSOR, 0, None

    @staticmethod
    def _input_granularity(module):
        """Return (mode, channel_axis, granularity) for the module's input quant scheme.

        Returns (None, 0, None) when input scheme is absent or uses a mode
        where static scale is not applicable (per_block MX, float formats).
        """
        if not hasattr(module, "cfg") or module.cfg.input is None:
            return None, 0, None
        s = module.cfg.input
        g = s.granularity
        if g.mode == GranularityMode.PER_BLOCK:
            return None, 0, None
        if s.format.ebits > 0:
            return None, 0, None
        return g.mode, g.channel_axis, g

    @staticmethod
    def _compute_bank_amax(x: torch.Tensor, gran) -> torch.Tensor:
        """Compute per-bank amax with shape broadcastable with the reshaped tensor.

        Reshapes x to expose the bank dimension, then reduces all dims
        except the bank dim.  The result has the same shape as the
        intermediate reshaped tensor inside _quantize_per_bank, with
        size 1 on all non-bank dims.
        """
        import warnings
        axis = gran.bank_axis
        if axis < 0:
            axis = x.ndim + axis
        if not (0 <= axis < x.ndim):
            raise ValueError(
                f"bank_axis={gran.bank_axis} out of range "
                f"for tensor with ndim={x.ndim}"
            )
        bank_size = gran.bank_size
        N_along = x.shape[axis]
        if N_along % bank_size != 0:
            warnings.warn(
                f"Dimension {axis} size {N_along} not divisible by "
                f"bank_size {bank_size}. Falling back to scalar amax. "
                f"This calibration scale will not match _quantize_per_bank "
                f"which requires divisibility.",
                stacklevel=2,
            )
            return torch.amax(torch.abs(x))
        num_banks = N_along // bank_size
        new_shape = list(x.shape)
        new_shape[axis] = num_banks
        new_shape.insert(axis + 1, bank_size)
        x_r = x.reshape(new_shape)
        dims_to_reduce = [i for i in range(x_r.ndim) if i != axis]
        return torch.amax(torch.abs(x_r), dim=tuple(dims_to_reduce), keepdim=True)


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
