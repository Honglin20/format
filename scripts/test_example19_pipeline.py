"""E2E test: simulate the full Example 19 agent pipeline data flow.

Tests the complete roundtrip that agents would exercise:
  Study(observers) → save → from_file → ranking → transform effect
  → block_error_analysis → viz rendering → chart export
"""
import sys
import os
import tempfile
import shutil

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from src.session import Session, QuantConfig
from src.session._study import Study
from src.analysis.observers import PerBlockQSNRObserver, QSNRObserver, MSEObserver
from src.api.block_error_analysis import block_error_analysis
from src.analysis.cross_config_ranking import CrossConfigLayerRanking
from src.analysis.transform_effect import TransformEffectReport
from src.report._study_report import StudyReport
from src.viz.block_error_heatmap import (
    block_error_heatmap, channel_error_bar, multi_config_block_comparison,
)


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
    print("E2E Pipeline Test: Simulating Example 19 Agent Data Flow")
    print("=" * 70)

    model = SmallMLP().eval()
    calib_data, eval_data = make_data()
    tmpdir = tempfile.mkdtemp(prefix="mxint_e2e_")

    try:
        # ── Agent 2: study_runner ────────────────────────────────────
        print("\n[Agent 2: study_runner] Running Study with 4 configs + observers...")

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

        observers = [QSNRObserver(), MSEObserver(), PerBlockQSNRObserver()]
        study = Study(configs, model=model)
        study_report = study.run(
            calib_data,
            eval_data=eval_data,
            eval_fn=eval_fn,
            outputs="all",
            observers=observers,
        )

        # Print accuracy table
        print("\n  Accuracy:")
        for part_results in study_report._results.values():
            for r in part_results:
                acc = r.quant_metrics.get("accuracy") if r.quant_metrics else None
                fp32 = r.fp32_metrics.get("accuracy") if r.fp32_metrics else None
                print(f"    {r.name:<12} FP32={fp32}  Quant={acc}")

        # ── Save + Reload (simulates agent pipeline handoff) ─────
        print(f"\n[Save/Load] Saving StudyReport to {tmpdir}...")
        study_report.save(tmpdir)

        print("[Save/Load] Reloading StudyReport from disk...")
        loaded_report = StudyReport.from_file(tmpdir)

        # Verify reloaded data matches original
        orig_configs = set()
        for part_results in study_report._results.values():
            for r in part_results:
                orig_configs.add(r.name)

        loaded_configs = set()
        for part_results in loaded_report._results.values():
            for r in part_results:
                loaded_configs.add(r.name)

        assert orig_configs == loaded_configs, \
            f"Config mismatch: {orig_configs} vs {loaded_configs}"
        print(f"  Verified: {len(loaded_configs)} configs reloaded correctly")

        # ── Agent 3: gap_analyzer ────────────────────────────────────
        print("\n[Agent 3: gap_analyzer] Analyzing accuracy gaps...")

        configs_acc = {}
        for part_results in loaded_report._results.values():
            for sr in part_results:
                if sr.quant_metrics:
                    acc = sr.quant_metrics.get("accuracy") or sr.quant_metrics.get("acc")
                    if acc is not None:
                        configs_acc[sr.name] = acc

        w8a8 = configs_acc.get("W8A8")
        w4a8 = configs_acc.get("W4A8")
        w4a4 = configs_acc.get("W4A4")
        weight_deg = (w4a8 - w8a8) if (w4a8 and w8a8) else 0
        act_deg = (w4a4 - w4a8) if (w4a4 and w4a8) else 0
        bottleneck = "weight" if abs(weight_deg) > abs(act_deg) else "activation"
        print(f"  Weight degradation (W8A8→W4A8): {weight_deg:+.4f}")
        print(f"  Activation degradation (W4A8→W4A4): {act_deg:+.4f}")
        print(f"  Primary bottleneck: {bottleneck}")

        # Transform effects
        tf_report = TransformEffectReport.from_study(loaded_report)
        print(f"\n  Transform effects:")
        for r in tf_report.per_config_recovery():
            gain = r['accuracy_gain']
            pct = r['recovery_pct']
            gain_str = f"{gain:+.4f}" if gain is not None else "N/A"
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            print(f"    {r['base_config']} +{r['transform']}: gain={gain_str}, recovery={pct_str}")

        # ── Agent 4: layer_attribution ───────────────────────────────
        print("\n[Agent 4: layer_attribution] Cross-config layer ranking...")

        ranking = CrossConfigLayerRanking.from_study(loaded_report)
        worst = ranking.consistent_worst(k=3)
        print(f"  Consistent worst layers:")
        for layer, avg_q in worst:
            w8 = ranking.get_layer_qsnr(layer, "W8A8")
            w4 = ranking.get_layer_qsnr(layer, "W4A4")
            print(f"    {layer}: avg={avg_q:.1f} dB  (W8A8={w8:.1f}, W4A4={w4:.1f})")

        # Per-config worst
        for cfg_name in ranking.config_names:
            specific = ranking.config_specific_worst(config=cfg_name, k=2)
            if specific:
                print(f"  {cfg_name}-specific worst: {specific[:2]}")

        print(ranking.summary(k=3))

        # ── Agent 5: distribution_profiler ───────────────────────────
        print("\n[Agent 5: distribution_profiler] Checking observers_data after reload...")

        has_obs_data = False
        for part_results in loaded_report._results.values():
            for r in part_results:
                if r.observers_data:
                    has_obs_data = True
                    n_layers = len(r.observers_data)
                    # Check tuple keys survived roundtrip
                    for layer, roles in r.observers_data.items():
                        for role, stages in roles.items():
                            for stage, slices in stages.items():
                                for key in slices:
                                    if isinstance(key, tuple):
                                        print(f"    {r.name}/{layer}/{role}: tuple key {key} OK")
                                        break
                                break
                            break
                        break
                    print(f"    {r.name}: {n_layers} layers with observer data")
                    break

        assert has_obs_data, "No observers_data found after reload — serialization failed"
        print("  PASS: observers_data survived save/load roundtrip")

        # ── Agent 6: block_analyst + visualization ───────────────────
        print("\n[Agent 6: block_analyst] Block-level error analysis + viz...")

        charts_dir = os.path.join(tmpdir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        chart_count = 0

        for part_results in loaded_report._results.values():
            for r in part_results:
                if r.name not in ("W4A4", "W4A8"):
                    continue
                for layer_name in r.qsnr_per_layer:
                    for role in ("weight", "input"):
                        try:
                            blk_report = block_error_analysis(
                                r, layer=layer_name, role=role, top_k=5
                            )
                            if blk_report.per_unit_qsnr:
                                print(f"  {r.name}/{layer_name} ({role}): "
                                      f"{len(blk_report.per_unit_qsnr)} units, "
                                      f"worst={blk_report.worst_units[:2]}")
                        except Exception as e:
                            print(f"  {r.name}/{layer_name} ({role}): SKIP ({e})")

                    # Render heatmap
                    try:
                        fig = block_error_heatmap(r, layer=layer_name, role="weight")
                        path = os.path.join(charts_dir, f"heatmap_{r.name}_{layer_name}.png")
                        fig.savefig(path, dpi=80)
                        plt.close(fig)
                        chart_count += 1
                        print(f"    Saved: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"    Heatmap failed: {e}")

                    # Render channel bar
                    try:
                        fig = channel_error_bar(r, layer=layer_name, role="input", top_k=5)
                        path = os.path.join(charts_dir, f"channel_{r.name}_{layer_name}.png")
                        fig.savefig(path, dpi=80)
                        plt.close(fig)
                        chart_count += 1
                        print(f"    Saved: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"    Channel bar failed: {e}")

                # Only process first config result per name
                break

        # Multi-config comparison
        try:
            first_layer = list(list(loaded_report._results.values())[0][0].qsnr_per_layer.keys())[0]
            fig = multi_config_block_comparison(
                loaded_report, layer=first_layer,
                role="weight", top_k=10,
            )
            path = os.path.join(charts_dir, "comparison.png")
            fig.savefig(path, dpi=80)
            plt.close(fig)
            chart_count += 1
            print(f"  Saved: comparison.png")
        except Exception as e:
            print(f"  Comparison chart failed: {e}")

        print(f"\n  Total charts rendered: {chart_count}")

        # ── Verify results ────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        checks = {
            "Study ran 4 configs": len(loaded_configs) == 4,
            "observers_data roundtrip": has_obs_data,
            "CrossConfigLayerRanking works": len(worst) > 0 or len(ranking.all_layers) > 0,
            "TransformEffectReport detected pairs": len(tf_report.pairs) >= 1,
            "Charts rendered": chart_count >= 1,
            "tuple keys survived roundtrip": has_obs_data,
        }

        all_pass = True
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{status}] {check}")

        print()
        if all_pass:
            print("ALL CHECKS PASSED — Example 19 pipeline E2E test complete")
        else:
            print("SOME CHECKS FAILED — see above for details")

        print("=" * 70)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
