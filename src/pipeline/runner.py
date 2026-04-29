from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from src.pipeline.config import resolve_config
from src.session import QuantSession
from src.calibration.strategies import MSEScaleStrategy
from src.analysis.observers import QSNRObserver, MSEObserver


def _extract_metric_per_layer(report, metric: str) -> Dict[str, float]:
    """Extract per-layer average of a metric from Report."""
    df = report.to_dataframe()
    if isinstance(df, list):
        result = {}
        for row in df:
            name = row.get("layer", "unknown")
            val = row.get(metric)
            if val is not None:
                result.setdefault(name, []).append(val)
        return {k: sum(v) / len(v) for k, v in result.items()}
    else:
        grouped = df.groupby("layer")[metric].mean()
        return grouped.to_dict()


class ExperimentRunner:
    """Thin grid-search scheduler over a search space of quantization configs.

    Iterates over every config in the search space, quantizes the model,
    runs calibration/analysis/evaluation via a single user-provided eval_fn,
    and returns structured results.

    The runner does NOT own the inference loop -- eval_fn controls all
    model interaction.  This makes the runner compatible with arbitrary
    model architectures and inference patterns.
    """

    def __init__(self, search_space: dict):
        self._search_space = search_space

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable[[nn.Module, Any], Dict[str, float]],
        calib_data: Any = None,
        analyze_data: Any = None,
        eval_data: Any = None,
        observers: list | None = None,
    ) -> Dict[str, dict]:
        """Execute the full quantize -> calibrate -> analyze -> evaluate flow.

        Args:
            fp32_model: Reference FP32 model (deep-copied, not mutated).
            eval_fn: ``(model, data) -> dict[str, float]``. Called in all
                three phases.  During calibration/analysis only forward
                side-effects are used (return value ignored).  During
                evaluation the returned dict is used for delta computation.
            calib_data: Data passed to eval_fn for calibration forward passes.
                None skips calibration.
            analyze_data: Data passed to eval_fn for analysis forward passes.
                Defaults to calib_data if both are needed. None skips analysis.
            eval_data: Data passed to eval_fn for fp32 vs quant metric comparison.
            observers: Observer instances for analysis. Default: QSNR + MSE.

        Returns:
            Dict mapping config_name to dict with keys:
            fp32, quant, delta, report, qsnr_per_layer, mse_per_layer.
        """
        if observers is None:
            observers = [QSNRObserver(), MSEObserver()]

        results = {}
        for part_name, part_def in self._search_space.items():
            configs = part_def.get("configs", {})
            for cfg_name, cfg_desc in configs.items():
                full_name = f"{part_name}/{cfg_name}"

                # Resolve descriptor to OpQuantConfig
                if isinstance(cfg_desc, dict):
                    cfg = resolve_config(cfg_desc)
                else:
                    cfg = cfg_desc  # Already an OpQuantConfig

                # Quantize -- deepcopy model to avoid mutating fp32 reference
                session = QuantSession(
                    copy.deepcopy(fp32_model), cfg,
                    calibrator=MSEScaleStrategy(),
                    keep_fp32=True,
                )

                # Phase 1: Calibrate
                if calib_data is not None:
                    with session.calibrate():
                        if isinstance(calib_data, (list, tuple)):
                            for batch in calib_data:
                                eval_fn(session, batch)
                        else:
                            eval_fn(session, calib_data)

                # Phase 2: Analyze
                report = None
                analyze_input = analyze_data if analyze_data is not None else calib_data
                if analyze_input is not None:
                    with session.analyze(observers=observers) as ctx:
                        if isinstance(analyze_input, (list, tuple)):
                            for batch in analyze_input:
                                eval_fn(session, batch)
                        else:
                            eval_fn(session, analyze_input)
                    report = ctx.report()

                # Phase 3: Evaluate
                fp32_model_copy = copy.deepcopy(fp32_model)
                fp32_metrics = eval_fn(fp32_model_copy, eval_data)
                quant_metrics = eval_fn(session, eval_data)
                delta = {k: quant_metrics.get(k, 0.0) - fp32_metrics.get(k, 0.0)
                         for k in fp32_metrics}

                entry = {
                    "fp32": fp32_metrics,
                    "quant": quant_metrics,
                    "delta": delta,
                    "report": report,
                }
                if report is not None:
                    entry["qsnr_per_layer"] = _extract_metric_per_layer(report, "qsnr_db")
                    entry["mse_per_layer"] = _extract_metric_per_layer(report, "mse")

                results[full_name] = entry

        return results
