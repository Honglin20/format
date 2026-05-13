"""Study: aggregate multiple Session runs into a comparison report.

Study is pure aggregation -- zero quantization logic, zero transform awareness.
It simply loops over configs, creates a Session for each, and collects results.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Dict, List, Optional, Union

import torch.nn as nn

from src.session._config import QuantConfig
from src.session._result import SessionResult
from src.session._session import Session


class Study:
    """Compare multiple QuantConfigs by running a Session for each.

    Study is NOT a kind of Session. It is an aggregator that runs N Sessions
    and collects their results. Zero quantization logic lives here.

    Args:
        configs: List of QuantConfigs to compare.
        model: FP32 model passed to each Session (deep-copied per run).
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
    ) -> "StudyReport":
        """Run all configs and return a StudyReport.

        Args:
            calib_data: Calibration data passed to each Session.
            eval_data: Optional evaluation data passed to each Session.
            eval_fn: ``(model, data) -> Dict[str, float]`` passed to each
                Session.
            outputs: Output keys passed to each Session
                (``"default"`` / ``"all"`` / list).
            model_factory: Optional per-config model factory. When provided,
                called with each config to produce a fresh model. When
                ``None``, ``copy.deepcopy(self._model)`` is used for every
                config.

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

            session = Session(model, cfg)
            result = session.run(
                calib_data,
                eval_data=eval_data,
                eval_fn=eval_fn,
                outputs=outputs,
            )
            results.setdefault(cfg.name, []).append(result)

            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")

        # Lazy import to avoid module-level circular dependency (ADR-008 SS5.2)
        from src.report._study_report import StudyReport  # noqa: PLC0415

        return StudyReport(results)
