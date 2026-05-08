"""Per-layer optimal transform selection.

PerLayerOpt is a post-processing tool: given SessionResults for different
transform variants (None / Hadamard / SmoothQuant), select the best transform
per layer by QSNR and re-run with a per-layer OpQuantConfig.

Absorbs logic from ``src/pipeline/format_study.py`` L282-330 and L540-643,
adapted to use ``SessionResult`` and ``QuantConfig``.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.analysis.observers import MSEObserver, QSNRObserver
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.transform import IdentityTransform
from src.session._config import QuantConfig
from src.session._quant import _QuantSession
from src.session._result import SessionResult
from src.session._session import _extract_qsnr_mse, _make_calibrator, _run_model
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import (
    SmoothQuantTransform,
    fuse_smoothquant_weights,
)
from src.viz._helpers import _compute_best_transform_per_layer

_logger = logging.getLogger(__name__)


def per_layer_optimal(
    part_results: List[SessionResult],
    calib_data,
    fp32_model: nn.Module,
    eval_fn: Callable,
    *,
    eval_data=None,
    sq_transforms: Optional[Dict[str, SmoothQuantTransform]] = None,
) -> SessionResult:
    """Select best transform per layer by QSNR and re-run.

    Takes the per-layer QSNR results from several transform variants of the
    same format (e.g. INT8-None, INT8-Hadamard, INT8-SmoothQuant), chooses
    the best transform for each layer, builds a per-layer OpQuantConfig,
    and runs a fresh calibration + analysis + evaluation.

    Args:
        part_results: SessionResults from running transform variants of the
            same format.  The ``.config.transform`` attribute is used to
            identify each variant.
        calib_data: Calibration data for re-running.
        fp32_model: Original FP32 model (used for SQ re-calibration if
            needed).  Not mutated.
        eval_fn: ``(model, data) -> Dict[str, float]``.
        eval_data: Evaluation data.  If ``None``, only calibration is run
            (no metrics computed).
        sq_transforms: Pre-computed SQ transforms for reuse (C1 fix).
            When not provided and SQ layers are selected, transforms are
            computed via ``SmoothQuantTransform.from_model_calibration``.

    Returns:
        ``SessionResult`` from running the per-layer optimal config.

    Raises:
        IndexError: If ``part_results`` is empty.
    """
    if not part_results:
        raise IndexError("part_results must contain at least one SessionResult")

    # ------------------------------------------------------------------
    # 1. Extract variant QSNR
    # ------------------------------------------------------------------
    variant_qsnr: Dict[str, Dict[str, float]] = {}
    for r in part_results:
        tx = r.config.transform
        variant_qsnr[tx] = r.qsnr_per_layer

    # ------------------------------------------------------------------
    # 2. Determine base config
    # ------------------------------------------------------------------
    base_cfg = part_results[0].config

    # ------------------------------------------------------------------
    # 3. Best transform per layer
    # ------------------------------------------------------------------
    best_per_layer: Dict[str, str] = _compute_best_transform_per_layer(
        variant_qsnr,
    )

    # ------------------------------------------------------------------
    # 4. Build per-layer OpQuantConfig dict
    # ------------------------------------------------------------------
    per_layer_cfgs: Dict[str, OpQuantConfig] = {}
    sq_layers: List[str] = []
    sq_winning_layers: List[str] = []

    for layer_name, best_tx in best_per_layer.items():
        if best_tx == "none":
            tx = IdentityTransform()
        elif best_tx == "hadamard":
            tx = HadamardTransform()
        elif best_tx == "smoothquant":
            sq_winning_layers.append(layer_name)
            if sq_transforms and layer_name in sq_transforms:
                tx = sq_transforms[layer_name]
                sq_layers.append(layer_name)
            else:
                # Will be filled after SQ computation below
                tx = IdentityTransform()
                if not sq_transforms:
                    _logger.warning(
                        "Layer %s selected SmoothQuant but no cached transform "
                        "found, falling back to Identity pending SQ computation",
                        layer_name,
                    )
        else:
            tx = IdentityTransform()

        # Start from the base per-role OpQuantConfig
        op_cfg = base_cfg.to_op_config()

        if base_cfg.weight_only:
            # Weight-only: override weight scheme transform
            if op_cfg.weight is not None:
                new_weight = QuantScheme(
                    format=op_cfg.weight.format,
                    granularity=op_cfg.weight.granularity,
                    transform=tx,
                    round_mode=op_cfg.weight.round_mode,
                    scale_storage=op_cfg.weight.scale_storage,
                )
                per_layer_cfgs[layer_name] = OpQuantConfig(weight=new_weight)
            else:
                per_layer_cfgs[layer_name] = op_cfg
        else:
            # Normal: override input scheme transform
            if op_cfg.input is not None:
                new_input = QuantScheme(
                    format=op_cfg.input.format,
                    granularity=op_cfg.input.granularity,
                    transform=tx,
                    round_mode=op_cfg.input.round_mode,
                    scale_storage=op_cfg.input.scale_storage,
                )
                per_layer_cfgs[layer_name] = OpQuantConfig(
                    input=new_input,
                    weight=op_cfg.weight,
                    output=op_cfg.output,
                    storage=op_cfg.storage,
                )
            else:
                per_layer_cfgs[layer_name] = op_cfg

    # ------------------------------------------------------------------
    # 5. Compute SQ transforms if needed (C1 fix: reuse cached)
    # ------------------------------------------------------------------
    if sq_winning_layers and not sq_transforms:
        sq_transforms = SmoothQuantTransform.from_model_calibration(
            fp32_model,
            calib_data,
            alpha=base_cfg.sq_alpha,
            eval_fn=eval_fn,
        )
        # Update per_layer_cfgs with real SQ transforms for all winning layers
        for layer_name in sq_winning_layers:
            if layer_name not in sq_transforms:
                _logger.warning(
                    "Layer %s selected SmoothQuant but from_model_calibration "
                    "did not return a transform, falling back to Identity",
                    layer_name,
                )
                continue
            if layer_name not in sq_transforms:
                continue
            sq_t = sq_transforms[layer_name]
            op_cfg = per_layer_cfgs[layer_name]
            if base_cfg.weight_only:
                if op_cfg.weight is not None:
                    new_weight = QuantScheme(
                        format=op_cfg.weight.format,
                        granularity=op_cfg.weight.granularity,
                        transform=sq_t,
                        round_mode=op_cfg.weight.round_mode,
                        scale_storage=op_cfg.weight.scale_storage,
                    )
                    per_layer_cfgs[layer_name] = OpQuantConfig(
                        weight=new_weight,
                    )
            else:
                if op_cfg.input is not None:
                    new_input = QuantScheme(
                        format=op_cfg.input.format,
                        granularity=op_cfg.input.granularity,
                        transform=sq_t,
                        round_mode=op_cfg.input.round_mode,
                        scale_storage=op_cfg.input.scale_storage,
                    )
                    per_layer_cfgs[layer_name] = OpQuantConfig(
                        input=new_input,
                        weight=op_cfg.weight,
                        output=op_cfg.output,
                        storage=op_cfg.storage,
                    )

    # ------------------------------------------------------------------
    # 6. Fuse SQ weights
    # ------------------------------------------------------------------
    if sq_winning_layers and sq_transforms:
        model = fuse_smoothquant_weights(
            fp32_model, sq_transforms, layer_names=sq_winning_layers,
        )
    else:
        model = copy.deepcopy(fp32_model)

    # ------------------------------------------------------------------
    # 7. Create metadata QuantConfig for the result
    # ------------------------------------------------------------------
    opt_config = QuantConfig(
        name=f"{base_cfg.name}-PerLayerOpt",
        w_format=base_cfg.w_format,
        a_format=base_cfg.a_format,
        w_granularity=base_cfg.w_granularity,
        w_block_size=base_cfg.w_block_size,
        a_granularity=base_cfg.a_granularity,
        a_block_size=base_cfg.a_block_size,
        transform="none",  # Handled per-layer
        scale_storage=base_cfg.scale_storage,
        calibrator=base_cfg.calibrator,
        weight_only=base_cfg.weight_only,
        sq_alpha=base_cfg.sq_alpha,
    )

    # ------------------------------------------------------------------
    # 8. Create _QuantSession with per-layer configs
    # ------------------------------------------------------------------
    # Uses _QuantSession directly (not Session) because Session works with
    # a single QuantConfig→OpQuantConfig, while per-layer optimal requires
    # per-layer OpQuantConfig dicts.
    calibrator = _make_calibrator(base_cfg.calibrator)
    qs = _QuantSession(
        model,
        per_layer_cfgs,
        calibrator=calibrator,
        keep_fp32=True,
    )

    # ------------------------------------------------------------------
    # 9. Calibrate (reuses _run_model from _session.py)
    # ------------------------------------------------------------------
    with qs.calibrate():
        _run_model(qs, calib_data, eval_fn=eval_fn)

    # ------------------------------------------------------------------
    # 10. Analyze
    # ------------------------------------------------------------------
    with qs.analyze(observers=[QSNRObserver(), MSEObserver()]) as ctx:
        _run_model(qs, calib_data, eval_fn=eval_fn)
    report = ctx.report()
    observers_data = report._raw
    qsnr_per_layer, mse_per_layer = _extract_qsnr_mse(observers_data)

    # ------------------------------------------------------------------
    # 11. Evaluate
    # ------------------------------------------------------------------
    fp32_metrics: Optional[Dict[str, float]] = None
    quant_metrics: Optional[Dict[str, float]] = None
    delta: Optional[Dict[str, float]] = None

    if eval_data is not None:
        if qs.fp32_model is not None:
            fp32_metrics = eval_fn(qs.fp32_model, eval_data)
        quant_metrics = eval_fn(qs, eval_data)
        if fp32_metrics is not None:
            delta = {
                k: fp32_metrics[k] - quant_metrics[k]
                for k in fp32_metrics
            }

    # ------------------------------------------------------------------
    # 12. Cost
    # ------------------------------------------------------------------
    cost = qs.estimate_cost(fp32=False)
    cost_fp32 = qs.estimate_cost(fp32=True) if qs.fp32_model else None

    # ------------------------------------------------------------------
    # 13. Return result
    # ------------------------------------------------------------------
    return SessionResult(
        name=opt_config.name,
        config=opt_config,
        fp32_metrics=fp32_metrics,
        quant_metrics=quant_metrics,
        delta=delta,
        qsnr_per_layer=qsnr_per_layer,
        mse_per_layer=mse_per_layer,
        observers_data=observers_data,
        cost=cost,
        cost_fp32=cost_fp32,
        sq_transforms=sq_transforms,
    )
