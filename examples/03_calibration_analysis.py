"""
03 — Calibration Strategies & Analysis Observers.

Compares all four ScaleStrategies and all four Observers on the same model.
Run:  PYTHONPATH=. python examples/03_calibration_analysis.py
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from _model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.session import QuantSession
from src.calibration.strategies import (
    MaxScaleStrategy,
    PercentileScaleStrategy,
    MSEScaleStrategy,
    KLScaleStrategy,
)
from src.analysis.observers import (
    QSNRObserver,
    MSEObserver,
    HistogramObserver,
    DistributionObserver,
)


def make_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_channel(axis=-1))
    return OpQuantConfig(input=scheme, weight=scheme, output=scheme)


def make_data(n_samples=32):
    return torch.randn(n_samples, 128)


def calibrate_and_measure(strategy, model, calib_data):
    """Run calibration + analysis for a given strategy, return QSNR/MSE summary."""
    session = QuantSession(model, make_cfg(), calibrator=strategy)
    session.eval()

    with session.calibrate():
        for i in range(0, len(calib_data), 4):
            session(calib_data[i : i + 4])

    with session.analyze(observers=[QSNRObserver(), MSEObserver()]) as ctx:
        for i in range(0, len(calib_data), 4):
            session(calib_data[i : i + 4])

    report = ctx.report()
    # Compute average QSNR/MSE across all layers
    total_qsnr, total_mse, n = 0.0, 0.0, 0
    for layer_name in report.keys():
        for role in report._raw[layer_name]:
            for stage in report._raw[layer_name][role]:
                for sk, metrics in report._raw[layer_name][role][stage].items():
                    if "qsnr_db" in metrics and isinstance(metrics["qsnr_db"], (int, float)):
                        total_qsnr += metrics["qsnr_db"]
                        n += 1
                    if "mse" in metrics and isinstance(metrics["mse"], (int, float)):
                        total_mse += metrics["mse"]
    return total_qsnr / n if n > 0 else 0, total_mse / n if n > 0 else 0


def main():
    print("=" * 55)
    print("Calibration Strategies & Analysis Observers")
    print("=" * 55)

    # ── Part 1: Compare calibration strategies ────────────────────
    print("\n1. Calibration strategy comparison (int8, per_channel)")

    calib_data = make_data()
    strategies = [
        ("Max", MaxScaleStrategy()),
        ("Percentile(q=99)", PercentileScaleStrategy(q=99.0)),
        ("MSE(n_steps=20)", MSEScaleStrategy(n_steps=20)),
        ("KL(n_bins=32)", KLScaleStrategy(n_bins=32, n_steps=10)),
    ]

    results = []
    for name, strat in strategies:
        model = ToyMLP()
        avg_qsnr, avg_mse = calibrate_and_measure(strat, model, calib_data)
        results.append((name, avg_qsnr, avg_mse))

    print(f"   {'Strategy':<22} {'Avg QSNR':>10} {'Avg MSE':>12}")
    print(f"   {'-'*22} {'-'*10} {'-'*12}")
    for name, qsnr, mse in results:
        print(f"   {name:<22} {qsnr:>8.1f} dB {mse:>12.6f}")

    # ── Part 2: All four observers ────────────────────────────────
    print("\n2. Observer output samples (MaxScaleStrategy)")

    model = ToyMLP()
    session = QuantSession(model, make_cfg())
    session.eval()

    all_observers = [
        QSNRObserver(),
        MSEObserver(),
        HistogramObserver(n_bins=32),
        DistributionObserver(),
    ]

    with session.calibrate():
        for i in range(0, len(calib_data), 4):
            session(calib_data[i : i + 4])

    with session.analyze(observers=all_observers) as ctx:
        for i in range(0, len(calib_data), 4):
            session(calib_data[i : i + 4])

    report = ctx.report()
    # Show one sample row to demonstrate observer output structure
    sample_layer = sorted(report.keys())[0]
    sample_role_data = report._raw[sample_layer]
    first_role = list(sample_role_data.keys())[0]
    first_stage = list(sample_role_data[first_role].keys())[0]
    first_slice = list(sample_role_data[first_role][first_stage].keys())[0]
    sample_metrics = sample_role_data[first_role][first_stage][first_slice]
    print(f"   Sample: layer={sample_layer}, role={first_role}, stage={first_stage}")
    for k, v in sample_metrics.items():
        print(f"     {k}: {v}")

    # ── Part 3: Per-layer breakdown ───────────────────────────────
    print("\n3. Per-layer QSNR / MSE breakdown")
    print(f"   {'Layer':<20} {'Role':<10} {'QSNR (dB)':>10} {'MSE':>12}")
    print(f"   {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
    seen = set()
    for layer_name in sorted(report.keys()):
        for role in report._raw[layer_name]:
            for stage in report._raw[layer_name][role]:
                for sk, metrics in report._raw[layer_name][role][stage].items():
                    key = (layer_name, role)
                    if key in seen:
                        continue
                    seen.add(key)
                    qsnr = metrics.get("qsnr_db", "N/A")
                    mse = metrics.get("mse", "N/A")
                    q_str = f"{qsnr:>8.1f} dB" if isinstance(qsnr, float) else f"{'N/A':>8}"
                    m_str = f"{mse:>12.6f}" if isinstance(mse, float) else f"{'N/A':>12}"
                    print(f"   {layer_name:<20} {role:<10} {q_str} {m_str}")

    print("\n" + "=" * 55)
    print("Calibration & analysis comparison complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
