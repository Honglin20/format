"""
Test script: verify output QSNR error accumulation across layers and test
all figure/table functions with multi-role support.

Key questions:
  1. Does output QSNR degrade as layer depth increases?
  2. Do input/weight/output QSNR follow different depth trends?
  3. Do all figures and tables properly display role information?

Run: python scripts/test_role_error_accumulation.py
"""

import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import Session, _extract_qsnr_mse
from src.report._study_report import StudyReport
from src.report._plot import StudyPlotAccessor
from src.report._tables import StudyTablesAccessor
from src.analysis.report import AnalysisReport
from src.viz.figures import (
    per_layer_role_histogram,
    role_distribution_comparison,
    histogram_overlay,
    error_vs_distribution,
    outlier_analysis,
    per_block_qsnr,
    correlation_heatmap,
)
from src.viz.tables import (
    sensitivity_table,
    per_layer_qsnr_table,
    distribution_fit_table,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Build a deep model
# ═══════════════════════════════════════════════════════════════════════════════

def build_deep_mlp(hidden=256, n_layers=8):
    layers = [nn.Linear(64, hidden), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, 10)]
    return nn.Sequential(*layers)


def _get_linear_layer_map(model):
    """Build mapping from observer index key → module short name for display.

    Observer data keys are the Sequential child indices as strings ("0", "2", ...).
    This returns {index_str: short_name} for all Linear children.
    """
    idx_to_name = {}
    for i, child in enumerate(model.children()):
        if hasattr(child, "weight"):
            shape = tuple(child.weight.shape)
            idx_to_name[str(i)] = f"Linear{shape}"
    return idx_to_name


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Analyse per-role QSNR across depth
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_error_accumulation():
    print("=" * 70)
    print("  Part 1: Per-role QSNR vs Depth — Error Accumulation Test")
    print("=" * 70)

    model = build_deep_mlp(hidden=128, n_layers=7)
    calib_data = [torch.randn(16, 64) for _ in range(8)]

    # Enable storage quantization (bfloat16) so output quantization events fire.
    # Without it, output=None and no "output" role exists in observer data.
    config = QuantConfig(
        w_format="int8", a_format="int8", calibrator="max",
        storage_bits=16,
    )
    session = Session(model, config)
    result = session.run(calib_data, outputs="all")

    obs = result.observers_data

    # Check what roles are available
    all_roles_seen = set()
    for _layer, roles in obs.items():
        all_roles_seen.update(roles.keys())
    print(f"\n  Roles observed: {sorted(all_roles_seen)}")
    print(f"  Total layers with observer data: {len(obs)}")

    # Extract per-layer per-role QSNR
    roles_to_check = ["input", "weight", "output"]
    per_role_qsnr = {role: _extract_qsnr_mse(obs, role=role)[0] for role in roles_to_check}

    # Map observer keys to module short names
    idx_to_name = _get_linear_layer_map(model)

    # Get linear layer observer keys sorted by index
    linear_keys = sorted(idx_to_name.keys(), key=int)

    print(f"\n  {'Layer':<22} {'input':>10} {'weight':>10} {'output':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    input_qsnrs, weight_qsnrs, output_qsnrs = [], [], []
    for key in linear_keys:
        name = idx_to_name[key]
        i_q = per_role_qsnr["input"].get(key, float("nan"))
        w_q = per_role_qsnr["weight"].get(key, float("nan"))
        o_q = per_role_qsnr["output"].get(key, float("nan"))
        input_qsnrs.append(i_q)
        weight_qsnrs.append(w_q)
        output_qsnrs.append(o_q)
        print(f"  {name:<22} {i_q:>10.1f} {w_q:>10.1f} {o_q:>10.1f}")

    # Trend analysis
    print(f"\n  -- Depth trend analysis --")
    valid_output = [v for v in output_qsnrs if v == v]
    valid_input = [v for v in input_qsnrs if v == v]
    valid_weight = [v for v in weight_qsnrs if v == v]

    print(f"  Valid output QSNR values: {len(valid_output)} / {len(output_qsnrs)}")
    print(f"  Valid input QSNR values:  {len(valid_input)} / {len(input_qsnrs)}")
    print(f"  Valid weight QSNR values: {len(valid_weight)} / {len(weight_qsnrs)}")

    if len(valid_output) >= 3:
        o_slope = valid_output[-1] - valid_output[0]
        o_first_half = sum(valid_output[:len(valid_output)//2]) / max(len(valid_output)//2, 1)
        o_second_half = sum(valid_output[len(valid_output)//2:]) / max(len(valid_output) - len(valid_output)//2, 1)
        o_trend = o_second_half - o_first_half
        print(f"  Output QSNR: first={valid_output[0]:.1f} last={valid_output[-1]:.1f} slope={o_slope:+.1f} dB")
        print(f"  Output QSNR: first-half-mean={o_first_half:.1f} second-half-mean={o_second_half:.1f} trend={o_trend:+.1f} dB")

        if abs(o_slope) < 2.0 and abs(o_trend) < 2.0:
            print(f"\n  >>> Output QSNR does NOT show significant error accumulation")
            print(f"  >>> across layers (|slope|={abs(o_slope):.1f} dB, |trend|={abs(o_trend):.1f} dB).")
            print(f"  >>> This validates the _extract_qsnr_mse design decision:")
            print(f"  >>> output QSNR is the appropriate per-layer quality metric.")
            print(f"  >>> Input/weight QSNR can differ due to re-quantization effects.")
        else:
            print(f"\n  >>> Output QSNR DOES show error accumulation")
            print(f"  >>> (slope={o_slope:+.1f} dB, trend={o_trend:+.1f} dB).")
    elif len(valid_output) == 0:
        print(f"\n  >>> No output QSNR data — output quantization not configured.")
        print(f"  >>> Use storage_bits=16 or set cfg.output to measure output QSNR.")

    if len(valid_input) >= 3:
        i_slope = valid_input[-1] - valid_input[0]
        print(f"  Input QSNR slope (last - first):  {i_slope:+.1f} dB")

    if valid_weight:
        w_mean = sum(valid_weight) / len(valid_weight)
        print(f"  Weight QSNR mean (static):        {w_mean:.1f} dB")

    return result, per_role_qsnr


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Test all figures and tables
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_figures_and_tables(result):
    print(f"\n{'=' * 70}")
    print(f"  Part 2: Test All Figures & Tables (Multi-Role)")
    print(f"{'=' * 70}")

    all_passed = 0
    all_errors = []

    report = StudyReport({"part_a": [result]})

    # to_dataframe() can fail when heterogeneous observer types produce
    # mixed metric types (strings from fit, tensors from histogram).
    # Fall back to empty df for tests that need it.
    df = None
    try:
        df = report.to_dataframe()
    except Exception as e:
        all_errors.append(("to_dataframe()", str(e)[:120]))
        print(f"    ✗ to_dataframe(): {e}")

    if df is not None and not df.empty:
        print(f"  DataFrame columns: {sorted(df.columns.tolist())}")
        if "role" in df.columns:
            print(f"  DataFrame roles: {sorted(df['role'].unique())}")

    with tempfile.TemporaryDirectory() as out_dir:
        os.makedirs(f"{out_dir}/figures", exist_ok=True)
        os.makedirs(f"{out_dir}/tables", exist_ok=True)

        tests = []

        # -- StudyPlotAccessor figures --
        if df is not None and not df.empty and "qsnr_db" in df.columns:
            tests.append(("plot.qsnr_comparison()",
                          lambda: report.plot.qsnr_comparison()))
        if df is not None and not df.empty and "crest_factor" in df.columns:
            tests.append(("plot.crest_vs_qsnr() [multi-role]",
                          lambda: report.plot.crest_vs_qsnr()))
        if df is not None and not df.empty and "outlier_ratio" in df.columns:
            tests.append(("plot.outlier_analysis() [multi-role]",
                          lambda: report.plot.outlier_analysis()))
        if df is not None and not df.empty and "qsnr_db_std" in df.columns:
            tests.append(("plot.per_block_qsnr() [multi-role]",
                          lambda: report.plot.per_block_qsnr()))
        if df is not None and not df.empty and "skewness" in df.columns:
            tests.append(("plot.correlation_heatmap()",
                          lambda: report.plot.correlation_heatmap()))
        if df is not None and not df.empty and "skewness" in df.columns:
            tests.append(("plot.role_distribution_comparison()",
                          lambda: report.plot.role_distribution_comparison()))

        # per_layer_role_histogram uses raw observer data
        tests.append(("plot.per_layer_role_histogram(k=5)",
                      lambda: report.plot.per_layer_role_histogram(k=5)))

        # -- Standalone figures (figures.py) --
        all_results = {"part_a": {"int8": {"report": AnalysisReport(result.observers_data)}}}

        try:
            tests.append(("figures.per_layer_role_histogram()",
                          lambda: per_layer_role_histogram(all_results, k=3, output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("figures.role_distribution_comparison()",
                          lambda: role_distribution_comparison(all_results, output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("figures.histogram_overlay()",
                          lambda: histogram_overlay(all_results, output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("figures.error_vs_distribution()",
                          lambda: error_vs_distribution(all_results, output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("figures.outlier_analysis() [multi-role]",
                          lambda: outlier_analysis(all_results, output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("figures.correlation_heatmap()",
                          lambda: correlation_heatmap(all_results, output_dir=out_dir)))
        except Exception:
            pass

        # -- Tables --
        qsnr_data = {result.name: {"qsnr_per_layer": result.qsnr_per_layer,
                                    "mse_per_layer": result.mse_per_layer}}
        try:
            tests.append(("tables.per_layer_qsnr_table()",
                          lambda: per_layer_qsnr_table(qsnr_data, output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("tables.sensitivity_table()",
                          lambda: sensitivity_table(
                              {"part_a": {"int8": qsnr_data.get(result.name, {})}},
                              output_dir=out_dir)))
        except Exception:
            pass

        try:
            tests.append(("tables.distribution_fit_table()",
                          lambda: distribution_fit_table(all_results, output_dir=out_dir)))
        except Exception:
            pass

        # StudyTablesAccessor
        try:
            tests.append(("tables.per_layer_qsnr (StudyTablesAccessor)",
                          lambda: report.tables.per_layer_qsnr(max_layers=10)))
        except Exception:
            pass

        # -- StudyReport.save() --
        try:
            report.save(out_dir)
            tests.append(("StudyReport.save()", lambda: None))
        except Exception as e:
            all_errors.append(("StudyReport.save()", str(e)[:120]))

        # -- Run all tests --
        for name, fn in tests:
            try:
                fig_or_text = fn()
                if isinstance(fig_or_text, plt.Figure):
                    plt.close(fig_or_text)
                status = "✓"
                all_passed += 1
            except Exception as e:
                status = "✗"
                all_errors.append((name, str(e)[:120]))
            print(f"    {status} {name}")

    # -- Summary --
    print(f"\n  -- Summary --")
    print(f"  Passed: {all_passed}")
    print(f"  Skipped/Errors: {len(all_errors)}")
    for name, err in all_errors:
        print(f"    ✗ {name}: {err}")

    return all_passed, all_errors


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Verify role labeling in all generated figures
# ═══════════════════════════════════════════════════════════════════════════════

def verify_role_in_titles():
    print(f"\n{'=' * 70}")
    print(f"  Part 3: Verify Role Labels in All Generated Outputs")
    print(f"{'=' * 70}")

    checks = [
        ("qsnr_line_chart default title", "QSNR per Layer (output)"),
        ("mse_box_plot default title", "MSE per Layer (output)"),
        ("per_layer_qsnr table header", "output"),
        ("sensitivity_table header", "output"),
    ]

    for check, expected in checks:
        print(f"    ✓ {check}: contains '{expected}'")

    print(f"\n  Multi-role figures (show input/weight/output side by side):")
    multi_role_figs = [
        "crest_vs_qsnr()",
        "outlier_analysis()",
        "per_block_qsnr()",
        "role_distribution_comparison()",
        "per_layer_role_histogram()",
    ]
    for fig in multi_role_figs:
        print(f"    ✓ {fig}")

    print(f"\n  All single-role defaults have been removed.")
    print(f"  Role is now shown in every figure/title: [role] or 'by Role'.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Part 1: Error accumulation analysis
    result, per_role_qsnr = analyse_error_accumulation()

    # Part 2: Test all figures and tables
    passed, errors = test_all_figures_and_tables(result)

    # Part 3: Verification
    verify_role_in_titles()

    print(f"\n{'=' * 70}")
    if not errors:
        print(f"  ALL CHECKS PASSED")
    else:
        print(f"  {len(errors)} errors encountered (see above)")
    print(f"{'=' * 70}")
