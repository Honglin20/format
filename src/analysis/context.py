import torch.nn as nn
from src.analysis.mixin import ObservableMixin
from src.analysis.report import Report


class AnalysisContext:
    """Context manager that attaches observers to ObservableMixin modules.

    Usage:
        with AnalysisContext(model, [QSNRObserver()]) as ctx:
            for batch in data:
                model(batch)
        report = ctx.report()
    """

    def __init__(self, model: nn.Module, observers=None,
                 warmup_batches: int = 0):
        self.model = model
        self.observers = observers or []
        self.warmup_batches = warmup_batches
        self._batch_count = 0

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ObservableMixin):
                module._observers = self.observers
                module._analysis_name = name
        return self

    def __exit__(self, *args):
        for module in self.model.modules():
            if isinstance(module, ObservableMixin):
                module._observers = []

    def report(self):
        """Aggregate all observer data into a Report.

        Uses deep merge so metrics from different observers on the same
        slice are combined rather than overwritten.
        """
        raw = {}
        for obs in self.observers:
            for layer, role_map in obs.report().items():
                layer_data = raw.setdefault(layer, {})
                for role, stages in role_map.items():
                    role_data = layer_data.setdefault(role, {})
                    for stage, slices in stages.items():
                        stage_data = role_data.setdefault(stage, {})
                        for slice_key, metrics in slices.items():
                            stage_data.setdefault(slice_key, {}).update(metrics)
        return Report(raw)

    def step(self):
        """Mark one batch complete. Warmup batches reset observers."""
        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            for obs in self.observers:
                obs.reset()
