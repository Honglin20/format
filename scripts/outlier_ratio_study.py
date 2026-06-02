"""
Outlier Ratio Sweep: w4a4 int4 per_channel + outlier_format/group_format=int8.

Compares ADR-012 (element sparse / outlier_format) and ADR-013 (group sparse /
group_format) across a sweep of ratios [0, 1], with w8a8 per_channel as the
upper-bound reference. Generates a plot at scripts/figures/outlier_ratio.png.

Interface::

    from scripts.outlier_ratio_study import run_outlier_ratio_study
    report = run_outlier_ratio_study(model, calib_data, eval_data, eval_fn)

Run:  PYTHONPATH=. python scripts/outlier_ratio_study.py [--model mnist|transformer] [--plot]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.session import Study, QuantConfig


# ---------------------------------------------------------------------------
# Default ratio sweep — dense in [0, 0.3] and [0.35, 1.0]
# ---------------------------------------------------------------------------

DEFAULT_RATIOS = [
    0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_outlier_ratio_configs(
    ratios: Sequence[float],
    *,
    w_format: str = "int4",
    a_format: str = "int4",
    w_granularity: str = "per_channel",
    a_granularity: str = "per_channel",
    outlier_fmt: str = "int8",
    group_fmt: str = "int8",
    include_w8a8: bool = True,
    include_outlier: bool = True,
    include_group: bool = True,
    include_gptq: bool = True,
) -> List[QuantConfig]:
    """Build config list for outlier ratio sweep.

    Args:
        ratios: Outlier/group ratios to sweep (e.g. [0.0, 0.1, 0.3, 0.5]).
        w_format / a_format: Base format for weights / activations.
        w_granularity / a_granularity: Granularity mode.
        outlier_fmt: Format for outlier elements (ADR-012).
        group_fmt: Format for H groups (ADR-013).
        include_w8a8: Include w8a8 per_channel reference config.
        include_outlier: Include ADR-012 element-sparse configs.
        include_group: Include ADR-013 group-sparse configs.
        include_gptq: Include GPTQ baseline + GPTQ+outlier/group combos.

    Returns:
        List of QuantConfig in display order.
    """
    configs: List[QuantConfig] = []

    if include_w8a8:
        configs.append(
            QuantConfig(
                name="w8a8-pc",
                w_format="int8", a_format="int8",
                w_granularity="per_channel", a_granularity="per_channel",
                quantize_nonlinear=False,
            )
        )

    # GPTQ standalone baseline
    if include_gptq:
        configs.append(
            QuantConfig(
                name="w4a4-gptq",
                w_format=w_format, a_format=a_format,
                w_granularity=w_granularity, a_granularity=a_granularity,
                w_axis=-1, a_axis=-1,
                gptq=True,
                quantize_nonlinear=False,
            )
        )

    if include_outlier:
        for r in ratios:
            configs.append(
                QuantConfig(
                    name=f"w4a4-outlier{r:.2f}",
                    w_format=w_format, a_format=a_format,
                    w_granularity=w_granularity, a_granularity=a_granularity,
                    w_axis=-1, a_axis=-1,
                    outlier_ratio=r,
                    outlier_format=outlier_fmt,
                    a_outlier_format=outlier_fmt,
                    quantize_nonlinear=False,
                )
            )

    if include_group:
        for r in ratios:
            configs.append(
                QuantConfig(
                    name=f"w4a4-group{r:.2f}",
                    w_format=w_format, a_format=a_format,
                    w_granularity=w_granularity, a_granularity=a_granularity,
                    w_axis=-1, a_axis=-1,
                    group_ratio=r,
                    group_format=group_fmt,
                    a_group_format=group_fmt,
                    quantize_nonlinear=False,
                )
            )

    # GPTQ + outlier/group combos (full ratio sweep)
    if include_gptq:
        if include_outlier:
            for r in ratios:
                configs.append(
                    QuantConfig(
                        name=f"w4a4-gptq-outlier{r:.2f}",
                        w_format=w_format, a_format=a_format,
                        w_granularity=w_granularity, a_granularity=a_granularity,
                        w_axis=-1, a_axis=-1,
                        gptq=True,
                        outlier_ratio=r,
                        outlier_format=outlier_fmt,
                        a_outlier_format=outlier_fmt,
                        quantize_nonlinear=False,
                    )
                )
        if include_group:
            for r in ratios:
                configs.append(
                    QuantConfig(
                        name=f"w4a4-gptq-group{r:.2f}",
                        w_format=w_format, a_format=a_format,
                        w_granularity=w_granularity, a_granularity=a_granularity,
                        w_axis=-1, a_axis=-1,
                        gptq=True,
                        group_ratio=r,
                        group_format=group_fmt,
                        a_group_format=group_fmt,
                        quantize_nonlinear=False,
                    )
                )

    return configs


# ---------------------------------------------------------------------------
# Eval function (generic)
# ---------------------------------------------------------------------------


def _eval_classification(model: nn.Module, data) -> Dict[str, float]:
    model.eval()

    if isinstance(data, list):
        with torch.no_grad():
            for batch in data:
                model(batch)
        return {}

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Core interface
# ---------------------------------------------------------------------------


def run_outlier_ratio_study(
    model: nn.Module,
    calib_data: List[torch.Tensor],
    eval_data,
    eval_fn: Optional[Callable] = None,
    *,
    ratios: Optional[Sequence[float]] = None,
    w_format: str = "int4",
    a_format: str = "int4",
    w_granularity: str = "per_channel",
    a_granularity: str = "per_channel",
    outlier_fmt: str = "int8",
    group_fmt: str = "int8",
    include_outlier: bool = True,
    include_group: bool = True,
    include_gptq: bool = True,
    outputs: str = "default",
):
    """Run an outlier/group ratio sweep and return the StudyReport.

    Args:
        model: FP32 model (deep-copied per config).
        calib_data: List of calibration tensors (e.g. from DataLoader).
        eval_data: Evaluation DataLoader or data.
        eval_fn: ``(model, data) -> Dict[str, float]``. Defaults to
            classification accuracy if omitted.
        ratios: Ratios to sweep. Default: dense sweep [0, 1].
        w_format / a_format: Base format for weights / activations.
        w_granularity / a_granularity: Granularity mode.
        outlier_fmt: Outlier-element format (ADR-012).
        group_fmt: H-group format (ADR-013).
        include_outlier: Include ADR-012 outlier_format configs.
        include_group: Include ADR-013 group_format configs.
        include_gptq: Include GPTQ baseline + GPTQ+outlier/group combos.
        outputs: Study output keys.

    Returns:
        StudyReport from src.report.
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS
    if eval_fn is None:
        eval_fn = _eval_classification

    configs = build_outlier_ratio_configs(
        ratios,
        w_format=w_format, a_format=a_format,
        w_granularity=w_granularity, a_granularity=a_granularity,
        outlier_fmt=outlier_fmt, group_fmt=group_fmt,
        include_outlier=include_outlier, include_group=include_group,
        include_gptq=include_gptq,
    )

    print(f"\nOutlier Ratio Study: {len(configs)} configs, {len(list(ratios))} ratios")
    model_copy = copy.deepcopy(model)
    study = Study(configs, model=model_copy)

    report = study.run(
        calib_data,
        eval_data=eval_data,
        eval_fn=eval_fn,
        outputs=outputs,
    )
    return report


# ---------------------------------------------------------------------------
# Extract ratio → accuracy from a StudyReport
# ---------------------------------------------------------------------------


def extract_ratio_data(report):
    """Extract (outlier_acc, group_acc, gptq_outlier_acc, gptq_group_acc, gptq_acc, w8a8_acc) from report."""
    results: Dict[str, Dict[str, float]] = {}
    for part in report.parts:
        for r in report._results[part]:
            acc = r.quant_metrics
            results[r.name] = acc if isinstance(acc, dict) else (acc or {})

    w8a8_acc = results.get("w8a8-pc", {}).get("accuracy", None)
    gptq_acc = results.get("w4a4-gptq", {}).get("accuracy", None)

    outlier_acc: Dict[float, float] = {}
    group_acc: Dict[float, float] = {}
    gptq_outlier_acc: Dict[float, float] = {}
    gptq_group_acc: Dict[float, float] = {}

    for name, acc_dict in results.items():
        v = acc_dict.get("accuracy", None) if isinstance(acc_dict, dict) else None
        if v is None:
            continue
        if name.startswith("w4a4-gptq-outlier"):
            ratio_str = name.replace("w4a4-gptq-outlier", "")
            try:
                gptq_outlier_acc[float(ratio_str)] = v
            except ValueError:
                pass
        elif name.startswith("w4a4-gptq-group"):
            ratio_str = name.replace("w4a4-gptq-group", "")
            try:
                gptq_group_acc[float(ratio_str)] = v
            except ValueError:
                pass
        elif name.startswith("w4a4-outlier"):
            ratio_str = name.replace("w4a4-outlier", "")
            try:
                outlier_acc[float(ratio_str)] = v
            except ValueError:
                pass
        elif name.startswith("w4a4-group"):
            ratio_str = name.replace("w4a4-group", "")
            try:
                group_acc[float(ratio_str)] = v
            except ValueError:
                pass

    return outlier_acc, group_acc, gptq_outlier_acc, gptq_group_acc, gptq_acc, w8a8_acc


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def format_results_table(report) -> str:
    outlier_acc, group_acc, gptq_outlier_acc, gptq_group_acc, gptq_acc, w8a8_acc = extract_ratio_data(report)
    w4a4_acc = outlier_acc.get(0.0) or group_acc.get(0.0)

    lines = []
    sep = "=" * 90
    lines.append(sep)
    lines.append("Outlier Ratio Study — Accuracy vs Ratio (with GPTQ ablation)")
    lines.append(sep)

    if w8a8_acc is not None:
        lines.append(f"  w8a8 per_channel (upper bound):  {w8a8_acc:.4f}")
    if w4a4_acc is not None:
        lines.append(f"  w4a4 per_channel (ratio=0):      {w4a4_acc:.4f}")
    if gptq_acc is not None:
        lines.append(f"  w4a4 GPTQ (no sparse):           {gptq_acc:.4f}")
    lines.append("")

    all_ratios = sorted(set(
        list(outlier_acc.keys()) + list(group_acc.keys())
        + list(gptq_outlier_acc.keys()) + list(gptq_group_acc.keys())
    ))
    if not all_ratios:
        lines.append("  (no ratio-sweep results found)")
        return "\n".join(lines)

    has_outlier = bool(outlier_acc)
    has_group = bool(group_acc)
    has_gptq_o = bool(gptq_outlier_acc)
    has_gptq_g = bool(gptq_group_acc)

    cols: List[str] = ["Ratio"]
    if has_outlier:
        cols.append("Outlier(int8)")
    if has_gptq_o:
        cols.append("GPTQ+Outlier")
    if has_group:
        cols.append("Group(int8)")
    if has_gptq_g:
        cols.append("GPTQ+Group")
    cols.extend(["vs w8a8", "vs w4a4"])
    header = f"  {cols[0]:<10}" + " ".join(f"{c:<16}" for c in cols[1:])
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    def _ref(ratio):
        for d in [gptq_group_acc, gptq_outlier_acc, group_acc, outlier_acc]:
            if ratio in d:
                return d[ratio]
        return None

    for ratio in all_ratios:
        parts = [f"{ratio:<10.2f}"]
        if has_outlier:
            o = outlier_acc.get(ratio)
            parts.append(f"{o:.4f}" if o is not None else "—")
        if has_gptq_o:
            go = gptq_outlier_acc.get(ratio)
            parts.append(f"{go:.4f}" if go is not None else "—")
        if has_group:
            g = group_acc.get(ratio)
            parts.append(f"{g:.4f}" if g is not None else "—")
        if has_gptq_g:
            gg = gptq_group_acc.get(ratio)
            parts.append(f"{gg:.4f}" if gg is not None else "—")
        ref = _ref(ratio)
        vs_w8a8 = f"{ref - w8a8_acc:+.4f}" if (ref is not None and w8a8_acc is not None) else ""
        vs_w4a4 = f"{ref - w4a4_acc:+.4f}" if (ref is not None and w4a4_acc is not None) else ""
        parts.append(vs_w8a8)
        parts.append(vs_w4a4)
        lines.append("  " + " ".join(f"{p:<16}" for p in parts))

    # GPTQ standalone comparison
    if gptq_acc is not None:
        lines.append("")
        lines.append(f"  GPTQ standalone vs w4a4 baseline: {gptq_acc - w4a4_acc:+.4f}" if w4a4_acc is not None else "")
        lines.append(f"  GPTQ standalone vs w8a8 upper:    {gptq_acc - w8a8_acc:+.4f}" if w8a8_acc is not None else "")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_ratio_curve(
    report,
    title: str = "Accuracy vs Outlier/Group Ratio",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Generate a matplotlib line chart from the ratio sweep results.

    Args:
        report: StudyReport from run_outlier_ratio_study().
        title: Plot title.
        save_path: Path to save the figure (PNG). If None, defaults to
            ``scripts/figures/outlier_ratio.png``.
        show: If True, call plt.show() to display interactively.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    outlier_acc, group_acc, gptq_outlier_acc, gptq_group_acc, gptq_acc, w8a8_acc = extract_ratio_data(report)
    w4a4_acc = outlier_acc.get(0.0) or group_acc.get(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # -- Reference lines --
    if w8a8_acc is not None:
        ax.axhline(y=w8a8_acc, color="green", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"w8a8 per_channel ({w8a8_acc:.4f})")
    if w4a4_acc is not None:
        ax.axhline(y=w4a4_acc, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"w4a4 per_channel ({w4a4_acc:.4f})")
    if gptq_acc is not None:
        ax.axhline(y=gptq_acc, color="purple", linestyle=":", linewidth=1.2, alpha=0.8,
                   label=f"w4a4 GPTQ ({gptq_acc:.4f})")

    # -- Outlier (ADR-012) --
    if outlier_acc:
        ratios_o = sorted(outlier_acc.keys())
        accs_o = [outlier_acc[r] for r in ratios_o]
        ax.plot(ratios_o, accs_o, marker="o", color="steelblue", linewidth=1.8,
                markersize=5, label="ADR-012 outlier_format(int8)")

    # -- Group (ADR-013) --
    if group_acc:
        ratios_g = sorted(group_acc.keys())
        accs_g = [group_acc[r] for r in ratios_g]
        ax.plot(ratios_g, accs_g, marker="s", color="darkorange", linewidth=1.8,
                markersize=5, label="ADR-013 group_format(int8)")

    # -- GPTQ + Outlier --
    if gptq_outlier_acc:
        ratios_go = sorted(gptq_outlier_acc.keys())
        accs_go = [gptq_outlier_acc[r] for r in ratios_go]
        ax.plot(ratios_go, accs_go, marker="^", color="mediumpurple", linewidth=1.5,
                markersize=6, linestyle="--", label="GPTQ + outlier_format(int8)")

    # -- GPTQ + Group --
    if gptq_group_acc:
        ratios_gg = sorted(gptq_group_acc.keys())
        accs_gg = [gptq_group_acc[r] for r in ratios_gg]
        ax.plot(ratios_gg, accs_gg, marker="D", color="orchid", linewidth=1.5,
                markersize=5, linestyle="--", label="GPTQ + group_format(int8)")

    ax.set_xlabel("Outlier / Group Ratio", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, "outlier_ratio.png")

    fig.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# MNIST model loader
# ---------------------------------------------------------------------------

_MNIST_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "mnist_mlp.pt")


def load_mnist_model(weights_path: Optional[str] = None) -> nn.Module:
    path = weights_path or _MNIST_WEIGHTS
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded MNIST MLP from {path}  (saved FP32 acc: {ckpt.get('fp32_test_acc', 'N/A'):.4f})")
    return model


# ---------------------------------------------------------------------------
# Transformer model loader
# ---------------------------------------------------------------------------

_TRANSFORMER_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "transformer_agnews.pt")
_TRANSFORMER_VOCAB = os.path.join(os.path.dirname(__file__), "weights", "transformer_agnews_vocab.pt")


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class _TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 4, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256,
                 max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = _PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def load_transformer_model(weights_path: Optional[str] = None,
                           vocab_path: Optional[str] = None):
    w_path = weights_path or _TRANSFORMER_WEIGHTS
    v_path = vocab_path or _TRANSFORMER_VOCAB

    ckpt = torch.load(w_path, map_location="cpu", weights_only=False)
    vocab_data = torch.load(v_path, map_location="cpu", weights_only=False)
    vocab = vocab_data["vocab"]

    model = _TransformerClassifier(
        vocab_size=ckpt["vocab_size"], num_classes=ckpt["num_classes"],
        d_model=ckpt["d_model"], nhead=ckpt["nhead"],
        num_layers=ckpt["num_layers"], dim_feedforward=ckpt["dim_feedforward"],
        max_len=ckpt["max_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Transformer from {w_path}  (saved FP32 acc: {ckpt.get('fp32_test_acc', 'N/A'):.4f})")
    return model, vocab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Outlier Ratio Sweep: w4a4 + outlier/group int8 vs w8a8"
    )
    parser.add_argument("--model", default="mnist", choices=["mnist", "transformer"],
                        help="Model to run (default: mnist)")
    parser.add_argument("--ratios", default=None,
                        help="Comma-separated ratios, e.g. '0.0,0.1,0.3,0.5'")
    parser.add_argument("--mode", default="both", choices=["outlier", "group", "both"],
                        help="Which sparse mode to sweep (default: both)")
    parser.add_argument("--weights", default=None, help="Override weights path")
    parser.add_argument("--vocab", default=None, help="Override vocab path (transformer)")
    parser.add_argument("--outputs", default="default", help="Study output keys")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate accuracy-vs-ratio plot (default: True)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-gptq", action="store_true", help="Skip GPTQ ablation configs")
    parser.add_argument("--plot-path", default=None, help="Custom path for plot PNG")
    args = parser.parse_args()

    do_plot = args.plot and not args.no_plot

    if args.ratios:
        ratios = [float(x.strip()) for x in args.ratios.split(",")]
    else:
        ratios = DEFAULT_RATIOS

    include_outlier = args.mode in ("outlier", "both")
    include_group = args.mode in ("group", "both")

    # -- Load model + data --
    if args.model == "mnist":
        from torchvision import datasets, transforms

        model = load_mnist_model(args.weights)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"  Loaded model FP32 test accuracy: {correct / total:.4f}")

        calib_samples = []
        for x, _y in train_loader:
            calib_samples.append(x)
            if len(calib_samples) >= 8:
                break

        eval_data = test_loader
        eval_fn = _eval_classification
        plot_title = "MNIST MLP — w4a4 int4 + Outlier/Group int8 + GPTQ Sweep"

    else:  # transformer
        import csv
        import urllib.request

        model, vocab = load_transformer_model(args.weights, args.vocab)

        data_dir = "/tmp/agnews_data"
        os.makedirs(data_dir, exist_ok=True)
        train_csv = os.path.join(data_dir, "train.csv")
        test_csv = os.path.join(data_dir, "test.csv")
        for url, path in [
            ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv", train_csv),
            ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", test_csv),
        ]:
            if not os.path.exists(path):
                print(f"Downloading {url} ...")
                urllib.request.urlretrieve(url, path)

        PAD, UNK = "<pad>", "<unk>"

        def tokenise(text: str, max_len: int = 64) -> list:
            ids = [vocab.get(t, vocab[UNK]) for t in text.lower().split()]
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [vocab[PAD]] * (max_len - len(ids))
            return ids

        def load_csv(path: str, max_len: int = 64, limit: int | None = None):
            texts, labels = [], []
            with open(path, encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 3:
                        continue
                    texts.append(tokenise(row[1] + " " + row[2], max_len))
                    labels.append(int(row[0]) - 1)
                    if limit and len(labels) >= limit:
                        break
            return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        train_x, train_y = load_csv(train_csv, max_len=64, limit=512)
        test_x, test_y = load_csv(test_csv, max_len=64, limit=7600)

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=128)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"  Loaded model FP32 test accuracy: {correct / total:.4f}")

        calib_samples = []
        for x, _y in train_loader:
            calib_samples.append(x)
            if len(calib_samples) >= 8:
                break

        eval_data = test_loader
        eval_fn = _eval_classification
        plot_title = "Transformer AG News — w4a4 int4 + Outlier/Group int8 + GPTQ Sweep"

    # -- Run study --
    print(f"\nRatios: {list(ratios)}")
    print(f"Modes: outlier={include_outlier}, group={include_group}")

    report = run_outlier_ratio_study(
        model,
        calib_samples,
        eval_data,
        eval_fn,
        ratios=ratios,
        include_outlier=include_outlier,
        include_group=include_group,
        include_gptq=not args.no_gptq,
        outputs=args.outputs,
    )

    # -- Print results --
    print("\n" + report.summary())
    print(format_results_table(report))

    # -- Plot --
    if do_plot:
        save_path = args.plot_path
        if save_path is None:
            figures_dir = os.path.join(os.path.dirname(__file__), "figures")
            os.makedirs(figures_dir, exist_ok=True)
            save_path = os.path.join(figures_dir, f"outlier_ratio_{args.model}.png")
        plot_ratio_curve(report, title=plot_title, save_path=save_path)


if __name__ == "__main__":
    main()
