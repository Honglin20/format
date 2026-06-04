"""Study: aggregate multiple quantization runs into a comparison report.

Study is pure aggregation -- zero quantization logic, zero transform awareness.
It simply loops over configs, runs ``run_quantization()`` for each, and collects results.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Dict, List, Optional, Union

import torch.nn as nn

from src.session._config import QuantConfig
from src.session._result import SessionResult
from src.session._session import run_quantization


class Study:
    """Compare multiple QuantConfigs by running quantization for each.

    Study is NOT a kind of quantized model. It is an aggregator that runs N
    configs and collects their results. Zero quantization logic lives here.

    Args:
        configs: List of QuantConfigs to compare.
        model: FP32 model (deep-copied per run).
    """

    def __init__(
        self,
        configs: List[QuantConfig],
        *,
        model: nn.Module,
    ):
        self._configs = configs
        self._model = model

    def run(
        self,
        calib_data,
        *,
        eval_data=None,
        eval_fn: Optional[Callable] = None,
        outputs: Union[str, List[str]] = "default",
        model_factory: Optional[Callable[[QuantConfig], nn.Module]] = None,
        observers: Optional[list] = None,
    ) -> "StudyReport":
        """Run all configs and return a StudyReport.

        Args:
            calib_data: Calibration data.
            eval_data: Optional evaluation data.
            eval_fn: ``(model, data) -> Dict[str, float]``.
            outputs: Output keys (``"default"`` / ``"all"`` / list).
            model_factory: Optional per-config model factory. When provided,
                called with each config to produce a fresh model.
            observers: Optional list of ObserverBase instances to attach
                to each config's Session.

        Returns:
            ``StudyReport`` from ``src.report``.
        """
        results: Dict[str, List[SessionResult]] = {}
        n_configs = len(self._configs)

        print(f"\nStudy: {n_configs} config(s), outputs={outputs}")

        for idx, cfg in enumerate(self._configs):
            t0 = time.perf_counter()
            print(f"  [{idx + 1}/{n_configs}] {cfg.name} ... ", end="", flush=True)

            if model_factory is not None:
                model = model_factory(cfg)
            else:
                model = copy.deepcopy(self._model)

            _qmodel, _fp32, result = run_quantization(
                model, cfg,
                calib_data,
                eval_data=eval_data,
                eval_fn=eval_fn,
                outputs=outputs,
                observers=observers,
            )
            results.setdefault(cfg.name, []).append(result)

            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")

        from src.report._study_report import StudyReport

        return StudyReport(results)
