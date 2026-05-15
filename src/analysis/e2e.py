"""
End-to-end comparison tools for fp32 vs quantized model evaluation.

Two modes:

  **Auto** — ``compare_models()`` runs both models on a DataLoader::

      result = compare_models(fp32_model, qmodel, eval_loader, eval_fn=my_eval)

  **Manual** — ``Comparator`` context manager for user-controlled loops::

      with Comparator() as cmp:
          for inputs, labels in data:
              ...
              cmp.record(fp32_out, q_out, labels)
      result = cmp.evaluate(eval_fn)
"""
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


def _default_accuracy(logits, labels):
    return {"accuracy": (logits.argmax(-1) == labels).float().mean().item()}


class Comparator:
    """Collect fp32/quant output pairs and compute user-defined metrics.

    Context manager for the **manual** comparison path.  The user controls
    the inference loop and calls :meth:`record` for each batch.

    Example::

        with Comparator() as cmp:
            for inputs, labels in eval_loader:
                fp32_out = fp32_model(inputs)
                q_out = qmodel(inputs)
                cmp.record(fp32_out, q_out, labels)
        result = cmp.evaluate(my_eval_fn, directions={"acc": "higher"})

    Args:
        device: If set, tensors are moved to this device before metric
            computation (useful when collecting on CPU for large datasets).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self._fp32_outputs: List[torch.Tensor] = []
        self._q_outputs: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._device = device

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Comparator":
        self._fp32_outputs.clear()
        self._q_outputs.clear()
        self._labels.clear()
        return self

    def __exit__(self, *args):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, fp32_out: torch.Tensor, quant_out: torch.Tensor,
               labels: torch.Tensor) -> None:
        """Record one batch of outputs.

        Args:
            fp32_out: Reference (fp32) model output.
            quant_out: Quantized model output.
            labels: Ground-truth labels / targets.
        """
        self._fp32_outputs.append(fp32_out.detach().cpu())
        self._q_outputs.append(quant_out.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def evaluate(
        self,
        eval_fn: Callable[..., Dict[str, float]],
        directions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Concatenate collected outputs and run *eval_fn* on both.

        Args:
            eval_fn: ``(logits, labels) -> dict[str, float]``.
                Must return a flat dictionary of metric names to floats.
            directions: Optional dict mapping metric names to ``"higher"``
                or ``"lower"``.  Used for display only (e.g. sorting).

        Returns:
            ``{"fp32": {...}, "quant": {...}, "delta": {...}}``
        """
        fp32_cat = torch.cat(self._fp32_outputs)
        q_cat = torch.cat(self._q_outputs)
        labels_cat = torch.cat(self._labels)

        if self._device is not None:
            fp32_cat = fp32_cat.to(self._device)
            q_cat = q_cat.to(self._device)
            labels_cat = labels_cat.to(self._device)

        fp32_metrics = eval_fn(fp32_cat, labels_cat)
        q_metrics = eval_fn(q_cat, labels_cat)

        delta = {}
        for k in fp32_metrics:
            delta[k] = q_metrics[k] - fp32_metrics[k]

        result = {
            "fp32": fp32_metrics,
            "quant": q_metrics,
            "delta": delta,
        }
        if directions:
            result["_directions"] = directions
        return result

    @property
    def num_samples(self) -> int:
        return sum(t.shape[0] for t in self._labels)


def compare_models(
    fp32_model: nn.Module,
    qmodel: nn.Module,
    eval_dataloader,
    eval_fn: Callable[..., Dict[str, float]] = _default_accuracy,
    directions: Optional[Dict[str, str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Auto-mode: run both models on *eval_dataloader* and compare.

    Args:
        fp32_model: Original (unquantized) model.
        qmodel: Quantized model.
        eval_dataloader: DataLoader yielding ``(inputs, labels)``.
        eval_fn: ``(logits, labels) -> dict[str, float]``.
        directions: Optional ``{"metric": "higher"|"lower"}`` hints.
        device: Optional device override for metric computation.

    Returns:
        ``{"fp32": {...}, "quant": {...}, "delta": {...}}``
    """
    fp32_model.eval()
    qmodel.eval()

    cmp = Comparator(device=device)
    with cmp:
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs, labels = batch[0], batch[1]
                fp32_out = fp32_model(inputs)
                q_out = qmodel(inputs)
                cmp.record(fp32_out, q_out, labels)

    return cmp.evaluate(eval_fn, directions=directions)


def compare_sessions(
    arg1,
    arg2: Dict[str, Any],
    arg3=None,
    eval_fn: Callable[..., Dict[str, float]] = None,
    directions: Optional[Dict[str, str]] = None,
    fp32_label: str = "fp32",
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple quantized models against a shared fp32 baseline.

    Accepts two calling conventions:

    **New API**: ``compare_sessions(fp32_model, {"name": qmodel}, dl)``
    **Old API (backward-compat)**: ``compare_sessions({"name": session}, dl)``
      where each value is a ``Session`` compat object with ``.fp32_model``
      and ``.qmodel`` attributes.
    """
    # Detect old API: first arg is a dict, second is a DataLoader
    if isinstance(arg1, dict):
        sessions: Dict[str, Any] = arg1
        eval_dataloader = arg2
        if eval_fn is None:
            eval_fn = _default_accuracy
        # Derive fp32_model from first session
        fp32_model = next(iter(sessions.values())).fp32_model
        if fp32_model is None:
            raise RuntimeError("Session has no fp32_model (keep_fp32=False?)")
        qmodels = {name: s.qmodel for name, s in sessions.items()}
    else:
        fp32_model = arg1
        qmodels = arg2
        eval_dataloader = arg3
        if eval_fn is None:
            eval_fn = _default_accuracy

    fp32_model.eval()
    for qm in qmodels.values():
        qm.eval()

    cmp_fp32 = Comparator()
    cmp_quant: Dict[str, Comparator] = {
        name: Comparator() for name in qmodels
    }

    with cmp_fp32, torch.no_grad():
        for batch in eval_dataloader:
            inputs, labels = batch[0], batch[1]

            fp32_out = fp32_model(inputs)
            cmp_fp32.record(fp32_out, fp32_out, labels)

            for name, qm in qmodels.items():
                q_out = qm(inputs)
                cmp_quant[name].record(fp32_out, q_out, labels)

    results: Dict[str, Dict[str, Any]] = {}
    results[fp32_label] = cmp_fp32.evaluate(eval_fn)["fp32"]

    for name, cmp in cmp_quant.items():
        r = cmp.evaluate(eval_fn, directions=directions)
        results[name] = r

    return results
