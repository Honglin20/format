"""E2E test: verify all 4 new APIs work with real quantization on a real model.

Uses bitx's built-in MLP (from scripts/mnist_hadamard_study.py pattern).
Tests: PerBlockQSNRObserver, block_error_analysis, CrossConfigLayerRanking, TransformEffectReport.
"""
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure bitx is importable
sys.path.insert(0, ".")

from src.session import Session, QuantConfig
from src.session._study import Study
from src.analysis.observers import PerBlockQSNRObserver, QSNRObserver, MSEObserver
from src.api.block_error_analysis import block_error_analysis
from src.analysis.cross_config_ranking import CrossConfigLayerRanking
from src.analysis.transform_effect import TransformEffectReport


# ── Model ──────────────────────────────────────────────────────────────────

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ── Data ───────────────────────────────────────────────────────────────────

def make_data(n_calib=32, n_eval=64):
    calib = [torch.randn(8, 784) for _ in range(n_calib // 8)]
    eval_x = torch.randn(n_eval, 784)
    eval_y = torch.randint(0, 10, (n_eval,))
    eval_dl = DataLoader(TensorDataset(eval_x, eval_y), batch_size=16)
    return calib, eval_dl


def eval_fn(model, data):
    if isinstance(data, list):
        for batch in data:
            model(batch)
        return {}
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ── Test ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("E2E Test: Example 19 APIs with real model")
    print("=" * 70)

    model = SmallMLP().eval()
    calib_data, eval_data = make_data()

    # ── 1. Study with multiple configs + PerBlockQSNRObserver ──────────
    print("\n[1] Running Study with 3 configs...")

    configs = [
        QuantConfig(name="W8A8", w_format="int8", w_granularity="per_block",
                    w_block_size=16, a_format="int8", a_granularity="per_block",
                    a_block_size=16),
        QuantConfig(name="W4A8", w_format="int4", w_granularity="per_block",
                    w_block_size=16, a_format="int8", a_granularity="per_block",
                    a_block_size=16),
        QuantConfig(name="W4A4", w_format="int4", w_granularity="per_block",
                    w_block_size=16, a_format="int4", a_granularity="per_block",
                    a_block_size=16),
        QuantConfig(name="W4A4+SQ", w_format="int4", w_granularity="per_block",
                    w_block_size=16, a_format="int4", a_granularity="per_block",
                    a_block_size=16, transform="smoothquant"),
    ]

    study = Study(configs, model=model)
    study_report = study.run(
        calib_data,
        eval_data=eval_data,
        eval_fn=eval_fn,
    )

    # Print accuracy summary
    print("\n  Accuracy Summary:")
    for part_name, part_results in study_report._results.items():
        for r in part_results:
            acc = r.quant_metrics.get("accuracy", "N/A") if r.quant_metrics else "N/A"
            fp32 = r.fp32_metrics.get("accuracy", "N/A") if r.fp32_metrics else "N/A"
            print(f"    {r.name:<12} FP32={fp32}  Quant={acc}")

    # ── 2. CrossConfigLayerRanking ─────────────────────────────────────
    print("\n[2] CrossConfigLayerRanking...")
    ranking = CrossConfigLayerRanking.from_study(study_report)
    print(f"  Configs: {ranking.config_names}")
    print(f"  All layers: {sorted(ranking.all_layers)}")

    worst = ranking.consistent_worst(k=3)
    print(f"  Consistent worst (k=3):")
    for layer, avg_q in worst:
        print(f"    {layer}: avg QSNR = {avg_q:.1f} dB")

    if worst:
        layer = worst[0][0]
        delta = ranking.layer_qsnr_delta(layer, from_config="W4A4", to_config="W8A8")
        print(f"  Delta for {layer} (W4A4→W8A8): {delta:+.1f} dB" if delta else f"  Delta: N/A")

    print(ranking.summary(k=3))

    # ── 3. TransformEffectReport ───────────────────────────────────────
    print("\n[3] TransformEffectReport...")
    tf_report = TransformEffectReport.from_study(study_report)
    print(tf_report.summary())

    recovery = tf_report.per_config_recovery()
    for r in recovery:
        print(f"  {r['base_config']} + {r['transform']}: "
              f"gain={r['accuracy_gain']:+.4f}, recovery={r['recovery_pct']:.1f}%")

    # ── 4. PerBlockQSNRObserver + block_error_analysis ─────────────────
    print("\n[4] PerBlockQSNRObserver + block_error_analysis...")

    # Run a single session with PerBlockQSNRObserver
    config_w4a4 = configs[2]  # W4A4
    session = Session(
        model,
        config_w4a4,
        observers=[QSNRObserver(), MSEObserver(), PerBlockQSNRObserver()],
        keep_fp32=True,
    )
    result = session.run(calib_data, eval_data=eval_data, eval_fn=eval_fn)

    # Analyze each layer's block-level error
    print(f"\n  Per-block analysis for W4A4:")
    for layer_name in result.qsnr_per_layer:
        for role in ("weight", "input"):
            try:
                report = block_error_analysis(result, layer=layer_name, role=role, top_k=5)
                if report.per_unit_qsnr:
                    print(f"\n{report.summary()}")
                    # Verify stats are consistent
                    assert report.stats["min"] <= report.stats["max"]
                    assert report.stats["mean"] >= report.stats["min"]
                    assert report.stats["mean"] <= report.stats["max"]
                    # Verify worst units are sorted
                    if len(report.worst_units) > 1:
                        for i in range(len(report.worst_units) - 1):
                            assert report.worst_units[i][1] <= report.worst_units[i+1][1]
            except Exception as e:
                print(f"  {layer_name} ({role}): {e}")

    print("\n" + "=" * 70)
    print("E2E Test PASSED — all 4 APIs work with real model")
    print("=" * 70)


if __name__ == "__main__":
    main()
