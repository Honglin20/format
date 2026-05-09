"""
Observer base classes for analysis.

ObserverBase: abstract base for fully custom observers.
SliceAwareObserver: granularity-aware base with vectorized per-mode dispatch.

Per-mode strategy (no Python loop over slices):
  PER_TENSOR   → single _measure() call on the whole tensor
  PER_CHANNEL  → movedim+reshape to [N_ch, -1], batch _measure_per_unit()
  PER_BLOCK    → movedim+reshape to [N_units, bs], single _measure_batch()
                  with valid_counts for correct partial-block measurement
  DYNAMIC_GROUP→ mask loop _measure() (variable group sizes, can't batch)

Subclasses implement _measure() for per-slice logic.
Override _measure_per_unit() / _measure_batch() for vectorized implementations.
"""
from abc import ABC, abstractmethod
from typing import List

import torch

from src.observer.events import QuantEvent
from src.scheme.granularity import GranularityMode

SliceKey = tuple  # (tag, *identifiers); first element is always a str tag


class ObserverBase(ABC):
    """Abstract observer base class. Override on_event for custom logic."""

    @abstractmethod
    def on_event(self, event: QuantEvent) -> None:
        ...

    def report(self) -> dict:
        """Return nested dict: {layer_name: {role: {stage: {slice_key: metric}}}}"""
        return {}

    def reset(self):
        """Clear accumulated state."""


class SliceAwareObserver(ObserverBase):
    """Granularity-aware observer. Vectorized per-mode dispatch, no slice loop.

    Required: _measure(key, fp32, quant) -> dict  (per single slice)
    Optional overrides for speed:
      _measure_per_unit(fp32_2d, quant_2d)  — used for PER_CHANNEL
      _measure_batch(fp32_2d, quant_2d, valid_counts=None)  — used for PER_BLOCK
    """

    def __init__(self):
        self._buffer: dict = {}

    @abstractmethod
    def _measure(self, key, fp32, quant) -> dict:
        """Compute metrics for a single slice. fp32 / quant may be any shape."""
        ...

    def _measure_per_unit(self, fp32_2d: torch.Tensor, quant_2d: torch.Tensor) -> List[dict]:
        """Compute metrics for each row of [N, D] tensors.

        Default: loop over _measure (correct but slow).
        Override for vectorized implementations (e.g. mean over dim=1).
        """
        return [self._measure(None, fp32_2d[i], quant_2d[i]) for i in range(fp32_2d.shape[0])]

    def _measure_batch(self, fp32_2d: torch.Tensor, quant_2d: torch.Tensor,
                       valid_counts: torch.Tensor = None) -> dict:
        """Compute aggregated metrics for all rows of [N, D] tensors (PER_BLOCK).

        Default: flatten into one slice (fast, distribution-preserving).
        When valid_counts is provided, padded positions are excluded from
        the flattened view so that zero-padding does not contaminate metrics.

        Override for per-block aggregate stats (mean/std/min/max across blocks).
        """
        if valid_counts is not None:
            mask = torch.arange(fp32_2d.shape[1], device=fp32_2d.device) < valid_counts.unsqueeze(1)
            return self._measure(None, fp32_2d[mask], quant_2d[mask])
        return self._measure(None, fp32_2d.reshape(-1), quant_2d.reshape(-1))

    def on_event(self, event: QuantEvent):
        fp32 = event.fp32_tensor
        quant = event.quant_tensor
        g = event.scheme.granularity
        mode = g.mode

        dst = (self._buffer
               .setdefault(event.layer_name, {})
               .setdefault(event.role, {})
               .setdefault(f"{event.stage}[{event.pipeline_index}]", {}))

        # Guard against QuantizeContext tensor patches.  When the model
        # forward is wrapped in QuantizeContext, torch.Tensor.__sub__,
        # __add__, __mul__, etc. are patched to apply quantization —
        # which would corrupt observer arithmetic (e.g. fp32 - quant
        # would quantize both operands before subtraction, always
        # producing 0).  _enter_quantize() / _exit_quantize() signal
        # the patches to use the original (unquantized) operations.
        from src.quantize.elemwise import _enter_quantize, _exit_quantize
        _enter_quantize()
        try:
            if mode == GranularityMode.PER_TENSOR:
                dst[("tensor",)] = self._measure(("tensor",), fp32, quant)

            elif mode == GranularityMode.PER_CHANNEL:
                axis = g.channel_axis
                if axis < 0:
                    axis = fp32.ndim + axis
                n_ch = fp32.shape[axis]
                fp32_2d = fp32.movedim(axis, 0).reshape(n_ch, -1)
                quant_2d = quant.movedim(axis, 0).reshape(n_ch, -1)
                for i, m in enumerate(self._measure_per_unit(fp32_2d, quant_2d)):
                    dst[("channel", i)] = m

            elif mode == GranularityMode.PER_BLOCK:
                bs = g.block_size
                axis = g.block_axis
                if axis < 0:
                    axis = fp32.ndim + axis
                dim_size = fp32.shape[axis]
                n_blocks = (dim_size + bs - 1) // bs

                # Move block axis to last so reshape can fold all other dims into N_units
                fp32_moved = fp32.movedim(axis, -1)
                quant_moved = quant.movedim(axis, -1)

                if dim_size % bs != 0:
                    pad = n_blocks * bs - dim_size
                    fp32_moved = torch.nn.functional.pad(fp32_moved, (0, pad))
                    quant_moved = torch.nn.functional.pad(quant_moved, (0, pad))

                # [*other_dims, n_blocks*bs] → [N_units, bs]
                fp32_2d = fp32_moved.reshape(-1, bs)
                quant_2d = quant_moved.reshape(-1, bs)

                if dim_size % bs != 0:
                    valid_counts = torch.full((fp32_2d.shape[0],), bs,
                                              dtype=torch.float32, device=fp32.device)
                    # Every n_blocks-th row is a partial last block
                    valid_counts[n_blocks - 1::n_blocks] = dim_size % bs
                else:
                    valid_counts = None

                dst[("block_agg",)] = self._measure_batch(fp32_2d, quant_2d,
                                                           valid_counts=valid_counts)

            elif mode == GranularityMode.DYNAMIC_GROUP:
                if event.group_map is None:
                    raise ValueError(
                        "DYNAMIC_GROUP granularity requires group_map to be set. "
                        "Make sure the format's quantize() returns the group_map tensor."
                    )
                for gid in event.group_map.unique().tolist():
                    mask = (event.group_map == gid)
                    gid_int = int(gid)
                    dst[("group", gid_int)] = self._measure(
                        ("group", gid_int), fp32[mask], quant[mask]
                    )

            else:
                raise ValueError(f"Unknown granularity mode: {mode}")
        finally:
            _exit_quantize()

    def report(self) -> dict:
        return self._buffer

    def reset(self):
        self._buffer.clear()
