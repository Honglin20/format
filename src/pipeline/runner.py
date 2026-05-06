from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from src.pipeline.config import resolve_config
from src.session import QuantSession
from src.calibration.strategies import MSEScaleStrategy
from src.analysis.observers import QSNRObserver, MSEObserver


@dataclass
class ExperimentResult:
    """Result of a single quantization experiment (one config)."""
    name: str
    fp32_metrics: Optional[Dict[str, float]] = None
    quant_metrics: Optional[Dict[str, float]] = None
    delta: Optional[Dict[str, float]] = None
    qsnr_per_layer: Dict[str, float] = field(default_factory=dict)
    mse_per_layer: Dict[str, float] = field(default_factory=dict)
    cost: Any = None
    cost_fp32: Any = None

    @property
    def avg_qsnr(self) -> float:
        if not self.qsnr_per_layer:
            return float("nan")
        return sum(self.qsnr_per_layer.values()) / len(self.qsnr_per_layer)

    @property
    def avg_mse(self) -> float:
        if not self.mse_per_layer:
            return float("nan")
        return sum(self.mse_per_layer.values()) / len(self.mse_per_layer)


def extract_metric_per_layer(report, metric: str) -> Dict[str, float]:
    """Extract per-layer average of a metric from Report."""
    df = report.to_dataframe()
    if isinstance(df, list):
        result: Dict[str, list] = {}
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
    """Execute a search space of quantization configs against a model.

    For each config: resolve -> create Session -> calibrate -> (LSQ) -> analyze -> evaluate.
    Pure execution - no print, no file I/O.
    """

    def __init__(self, search_space: dict, *, skip_parts: Optional[set] = None):
        self._search_space = search_space
        self._skip = skip_parts or set()

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable,
        calib_data: Any,
        eval_data: Any = None,
        observers: list | None = None,
        on_config_done: Optional[Callable] = None,
        model_for_part: Optional[Callable[[str], nn.Module]] = None,
    ) -> Dict[str, List[ExperimentResult]]:
        if observers is None:
            observers = [QSNRObserver(), MSEObserver()]
        if eval_data is None:
            eval_data = calib_data

        all_results: Dict[str, List[ExperimentResult]] = {}

        for part_name, part_cfg in self._search_space.items():
            if part_name in self._skip:
                continue

            configs = part_cfg.get("configs", [])
            part_results: List[ExperimentResult] = []

            for cfg_desc in configs:
                op_cfg = resolve_config(cfg_desc)

                if model_for_part is not None:
                    model = model_for_part(part_name)
                else:
                    model = copy.deepcopy(fp32_model)

                session = QuantSession(
                    model, op_cfg,
                    calibrator=MSEScaleStrategy(),
                    keep_fp32=True,
                )

                # Phase 1: LSQ (optional)
                lsq_steps = cfg_desc.get("lsq_steps", 0)
                if lsq_steps > 0:
                    from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer
                    session.initialize_pre_scales(
                        calib_data,
                        init=cfg_desc.get("lsq_init", "ones"),
                        pot=cfg_desc.get("lsq_pot", False),
                    )
                    opt = LayerwiseScaleOptimizer(
                        num_steps=lsq_steps,
                        num_batches=len(calib_data) if isinstance(calib_data, list) else 1,
                        optimizer="adam",
                        lr=cfg_desc.get("lsq_lr", 1e-3),
                        pot=cfg_desc.get("lsq_pot", False),
                    )
                    session.optimize_scales(opt, calib_data, eval_fn=eval_fn)

                # Phase 2: Calibrate
                with session.calibrate():
                    eval_fn(session, calib_data)

                # Phase 3: Analyze
                with session.analyze(observers=observers) as ctx:
                    eval_fn(session, calib_data)
                report = ctx.report()

                # Phase 4: Evaluate
                fp32_copy = copy.deepcopy(fp32_model)
                fp32_metrics = eval_fn(fp32_copy, eval_data)
                quant_metrics = eval_fn(session, eval_data)
                delta = {
                    k: quant_metrics.get(k, 0.0) - fp32_metrics.get(k, 0.0)
                    for k in fp32_metrics
                }

                result = ExperimentResult(
                    name=cfg_desc["name"],
                    fp32_metrics=fp32_metrics,
                    quant_metrics=quant_metrics,
                    delta=delta,
                    qsnr_per_layer=extract_metric_per_layer(report, "qsnr_db"),
                    mse_per_layer=extract_metric_per_layer(report, "mse"),
                    cost=session.estimate_cost(),
                    cost_fp32=session.estimate_cost(fp32=True),
                )
                part_results.append(result)

                if on_config_done:
                    on_config_done(result)

            all_results[part_name] = part_results

        return all_results
