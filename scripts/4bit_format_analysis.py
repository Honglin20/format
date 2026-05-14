#!/usr/bin/env python3
"""
4-bit Format Analysis for Shakespeare GPT.

Loads a pretrained Shakespeare character-level GPT, computes FP32 baseline
perplexity, prepares calibration data, and serves as the entry point for a
4-part analysis of 4-bit quantization formats.

Run:  PYTHONPATH=. python scripts/4bit_format_analysis.py
"""

from __future__ import annotations

import math
import os

import torch

from scripts.transformer_quant_study import (
    MiniGPT,
    make_dataloaders,
    eval_fn,
)
from src.session import Session, Study, QuantConfig


def main() -> None:
    torch.manual_seed(42)

    print("=" * 60)
    print("  4-bit Format Analysis — Shakespeare GPT")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[Data]")
    train_loader, val_loader, vocab_size = make_dataloaders(
        block_size=128, batch_size=64
    )
    print(f"  Vocab size:              {vocab_size}")
    print(f"  Training batches:        {len(train_loader)}")
    print(f"  Validation batches:      {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[Model]")

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=192,
        n_heads=3,
        n_layers=4,
        block_size=128,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:              {n_params:,}")

    # Load pretrained state dict
    weights_path = os.path.join(
        os.path.dirname(__file__), "weights", "shakespeare_gpt.pt"
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"  Loaded weights from:     {weights_path}")

    # ------------------------------------------------------------------
    # FP32 Baseline Perplexity
    # ------------------------------------------------------------------
    print("\n[FP32 Baseline Perplexity]")

    model.eval()
    with torch.no_grad():
        fp32_result = eval_fn(model, val_loader)
    fp32_ppl = fp32_result["perplexity"]
    print(f"  FP32 validation PPL:     {fp32_ppl:.4f}")

    # ------------------------------------------------------------------
    # Calibration Data (4 batches from training loader)
    # ------------------------------------------------------------------
    print("\n[Calibration Data]")

    calib_data: list[torch.Tensor] = []
    for x, _y in train_loader:
        calib_data.append(x)
        if len(calib_data) >= 4:
            break

    print(f"  Calibration batches:     {len(calib_data)}")
    if calib_data:
        print(f"  Calibration batch shape: {calib_data[0].shape}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Vocab size:              {vocab_size}")
    print(f"  Parameters:              {n_params:,}")
    print(f"  Architecture:            MiniGPT(d=192, h=3, L=4, T=128)")
    print(f"  FP32 val PPL:            {fp32_ppl:.4f}")
    print(f"  Calibration batches:     {len(calib_data)}")
    print()

    # ------------------------------------------------------------------
    # Part 1: MXINT Precision Comparison (W8A8 / W4A8 / W4A4)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 1: MXINT Precision Comparison (W8A8 / W4A8 / W4A4)")
    print("=" * 60)

    mxint_configs = [
        QuantConfig(
            name="MXINT-W8A8",
            w_format="int8", a_format="int8",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="MXINT-W4A8",
            w_format="int4", a_format="int8",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="MXINT-W4A4",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
    ]

    study = Study(mxint_configs, model=model)
    report = study.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    print("\n--- Summary (local QSNR) ---")
    print(report.summary())

    print("\n--- Summary (accum QSNR) ---")
    print(report.summary(qsnr_type="accum"))

    # --- Markdown table ---
    df_local = report.summary_dataframe(qsnr_type="local")
    df_accum = report.summary_dataframe(qsnr_type="accum")

    print("\n--- MXINT Comparison Table ---")
    print()
    print(f"| Config | PPL | ΔPPL | QSNR (local) | QSNR (accum) |")
    print(f"|--------|-----|------|--------------|---------------|")
    print(f"| FP32 baseline | {fp32_ppl:.4f} | — | — | — |")

    if df_local is not None and df_accum is not None:
        merged = df_local.merge(
            df_accum[["config", "avg_qsnr_db"]],
            on="config", suffixes=("", "_accum")
        )
        for _, row in merged.iterrows():
            print(f"| {row['config']} | {row['quant_perplexity']:.4f} | "
                  f"{row['delta_perplexity']:+.4f} | "
                  f"{row['avg_qsnr_db']:.2f} dB | "
                  f"{row['avg_qsnr_db_accum']:.2f} dB |")

    # ------------------------------------------------------------------
    # Part 2: Root Cause Analysis (Diagnose + Characterize + Visualize)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 2: Root Cause Analysis")
    print("=" * 60)

    import copy

    import matplotlib.pyplot as plt

    analysis_dir = os.path.join(os.path.dirname(__file__), "analysis_output")
    os.makedirs(analysis_dir, exist_ok=True)

    # ---- Step 1: W4A4 Session with observers ----
    print("\n[1/6] W4A4 Detail Session (distribution + histogram observers)")

    cfg_w4a4_detail = QuantConfig(
        name="MXINT-W4A4-detail",
        w_format="int4", a_format="int4",
        w_granularity="per_block", a_granularity="per_block",
        w_block_size=32, a_block_size=32,
        quantize_nonlinear=False,
    )

    session_w4a4 = Session(model=copy.deepcopy(model), config=cfg_w4a4_detail)
    result_w4a4 = session_w4a4.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs=["accuracy", "qsnr", "distribution", "histogram"],
    )

    # ---- Step 2: Diagnose (per-role per-layer QSNR) ----
    print("\n[2/6] Diagnose — Per-Layer Per-Role QSNR")

    print("\n--- Top-10 Worst Layers (local QSNR) ---")
    for layer, qsnr in result_w4a4.top_k_qsnr(k=10, qsnr_type="local"):
        print(f"  {layer:<30} {qsnr:.2f} dB")

    if result_w4a4.accum_qsnr_per_layer:
        print("\n--- Top-10 Worst Layers (accumulated) ---")
        for layer, qsnr in result_w4a4.top_k_qsnr(k=10, qsnr_type="accum"):
            print(f"  {layer:<30} {qsnr:.2f} dB")

    print("\n--- Per-Role Attribution Table (diagnose.per_role_table) ---")
    try:
        table = result_w4a4.diagnose.per_role_table(max_layers=20)
        print(table)
    except Exception as exc:
        print(f"  (per_role_table unavailable: {exc})")

    print("\n--- Per-Role Summary (diagnose.summary) ---")
    try:
        print(result_w4a4.diagnose.summary())
    except Exception as exc:
        print(f"  (summary unavailable: {exc})")

    # ---- Step 3: Characterize (distribution degradation) ----
    print("\n[3/6] Characterize — Distribution Degradation")

    print("\n--- Causal Analysis (characterize.causal_analysis) ---")
    try:
        causal = result_w4a4.characterize.causal_analysis()
        print(causal)
    except Exception as exc:
        print(f"  (causal_analysis unavailable: {exc})")

    print("\n--- Per-Layer Per-Role Distribution Metrics ---")
    printed = 0
    target = 20
    for layer, roles in sorted(result_w4a4.observers_data.items()):
        for role in ("input", "weight", "output"):
            if printed >= target:
                break
            qsnr = result_w4a4.qsnr_by_role.get(role, {}).get(layer)
            if qsnr is None:
                continue
            try:
                diag = result_w4a4.characterize.classify(layer, role)
            except Exception:
                diag = "unknown"
            print(f"  {layer:<30} {role:<8} {qsnr:>7.1f} dB  [{diag}]")
            printed += 1
        if printed >= target:
            break

    # ---- Step 4: W8A8 comparison ----
    print("\n[4/6] W8A8 Comparison Session")

    cfg_w8a8_detail = QuantConfig(
        name="MXINT-W8A8-detail",
        w_format="int8", a_format="int8",
        w_granularity="per_block", a_granularity="per_block",
        w_block_size=32, a_block_size=32,
        quantize_nonlinear=False,
    )

    session_w8a8 = Session(model=copy.deepcopy(model), config=cfg_w8a8_detail)
    result_w8a8 = session_w8a8.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs=["accuracy", "qsnr", "distribution", "histogram"],
    )

    w4a4_ppl = (
        result_w4a4.quant_metrics.get("perplexity", float("nan"))
        if result_w4a4.quant_metrics else float("nan")
    )
    w8a8_ppl = (
        result_w8a8.quant_metrics.get("perplexity", float("nan"))
        if result_w8a8.quant_metrics else float("nan")
    )
    print(f"  W4A4 PPL: {w4a4_ppl:.4f}")
    print(f"  W8A8 PPL: {w8a8_ppl:.4f}")

    # ---- Step 5: Visualization ----
    print("\n[5/6] Visualization")

    # 5a: QSNR comparison bar chart
    print("  (a) QSNR comparison bar chart")
    try:
        fig = result_w4a4.plot.qsnr_comparison()
        fig.savefig(
            os.path.join(analysis_dir, "qsnr_comparison.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"      -> {analysis_dir}/qsnr_comparison.png")
    except Exception as exc:
        print(f"      (skipped: {exc})")

    # 5b: Per-role QSNR bars
    print("  (b) Per-role QSNR bars")
    try:
        fig = result_w4a4.plot.per_role_qsnr_bars(max_layers=30)
        fig.savefig(
            os.path.join(analysis_dir, "per_role_qsnr_bars.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"      -> {analysis_dir}/per_role_qsnr_bars.png")
    except Exception as exc:
        print(f"      (skipped: {exc})")

    # 5c: Channel heterogeneity for the worst layer
    print("  (c) Channel heterogeneity (worst layer)")
    worst = result_w4a4.top_k_qsnr(k=1, qsnr_type="local")
    if worst:
        worst_name = worst[0][0]
        worst_qsnr = worst[0][1]
        print(f"      Worst layer: {worst_name} (QSNR={worst_qsnr:.2f} dB)")
        for role in ("weight", "input", "output"):
            try:
                fig = result_w4a4.plot.channel_heterogeneity(worst_name, role=role)
                safe = worst_name.replace(".", "_")
                fig.savefig(
                    os.path.join(analysis_dir, f"channel_hetero_{role}_{safe}.png"),
                    dpi=300, bbox_inches="tight",
                )
                plt.close(fig)
                print(f"      -> channel_hetero_{role}_{safe}.png")
            except Exception as exc:
                print(f"      (skipped {role}: {exc})")

    # 5d: Histogram overlay — separate plots for W4A4 and W8A8
    print("  (d) Histogram overlay (8bit vs 4bit comparison)")
    try:
        from src.viz.figures import histogram_overlay

        # W4A4 histogram (most quantization-sensitive layers)
        print("      [W4A4]")
        w4a4_results = {
            "part_1": {
                "MXINT-W4A4-detail": {"report": result_w4a4.report},
            },
        }
        fig = histogram_overlay(
            w4a4_results, output_dir=analysis_dir, name="histogram_overlay_w4a4",
        )
        plt.close(fig)
        print(f"      -> {analysis_dir}/histogram_overlay_w4a4.png")

        # W8A8 histogram (for comparison with W4A4)
        print("      [W8A8]")
        w8a8_results = {
            "part_1": {
                "MXINT-W8A8-detail": {"report": result_w8a8.report},
            },
        }
        fig = histogram_overlay(
            w8a8_results, output_dir=analysis_dir, name="histogram_overlay_w8a8",
        )
        plt.close(fig)
        print(f"      -> {analysis_dir}/histogram_overlay_w8a8.png")

        print("      Note: Compare histogram_overlay_w4a4.png and")
        print("            histogram_overlay_w8a8.png side by side to")
        print("            visualize 8bit vs 4bit distribution differences.")
    except Exception as exc:
        print(f"      (skipped: {exc})")

    # ---- Step 6: Attribution markdown table ----
    print("\n[6/6] Attribution Table")
    print()
    print("| Layer | Role | QSNR (dB) | Degradation Type |")
    print("|-------|------|-----------|-----------------|")

    rows = []
    for role in ("input", "weight", "output"):
        for layer, qsnr in result_w4a4.qsnr_by_role.get(role, {}).items():
            if qsnr is not None and not math.isnan(qsnr) and qsnr != float("-inf"):
                try:
                    diag = result_w4a4.characterize.classify(layer, role)
                except Exception:
                    diag = "unknown"
                rows.append((qsnr, layer, role, diag))

    rows.sort(key=lambda x: x[0])  # worst first
    for qsnr, layer, role, diag in rows[:20]:
        print(f"| {layer:<28} | {role:<6} | {qsnr:>7.2f} | {diag:<18} |")
    if len(rows) > 20:
        print(f"| ... and {len(rows) - 20} more rows |")

    # ------------------------------------------------------------------
    # Part 3: MXFP / NF4 Format Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 3: MXFP / NF4 Cross-Format Comparison")
    print("=" * 60)

    xfmt_configs = [
        QuantConfig(
            name="MXINT-4",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="MXFP-4",
            w_format="fp4_e2m1", a_format="fp4_e2m1",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="NF4-W",
            w_format="nf4",
            w_granularity="per_channel",
            weight_only=True,
            quantize_nonlinear=False,
            scale_storage="fp32",
        ),
        QuantConfig(
            name="NF4-WA",
            w_format="nf4", a_format="nf4",
            w_granularity="per_channel", a_granularity="per_channel",
            weight_only=False,
            quantize_nonlinear=False,
            scale_storage="fp32",
        ),
        QuantConfig(
            name="MXFP-8",
            w_format="fp8_e4m3", a_format="fp8_e4m3",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
    ]

    study_xfmt = Study(xfmt_configs, model=model)
    report_xfmt = study_xfmt.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    print("\n--- Summary (local QSNR) ---")
    print(report_xfmt.summary())

    print("\n--- Summary (accum QSNR) ---")
    print(report_xfmt.summary(qsnr_type="accum"))

    # --- Markdown table ---
    df_xfmt_local = report_xfmt.summary_dataframe(qsnr_type="local")
    df_xfmt_accum = report_xfmt.summary_dataframe(qsnr_type="accum")

    print("\n--- MXFP / NF4 Comparison Table ---")
    print()
    print(f"| Format | PPL | ΔPPL | QSNR (local) | QSNR (accum) |")
    print(f"|--------|-----|------|--------------|---------------|")
    print(f"| FP32 baseline | {fp32_ppl:.4f} | — | — | — |")

    if df_xfmt_local is not None and df_xfmt_accum is not None:
        merged = df_xfmt_local.merge(
            df_xfmt_accum[["config", "avg_qsnr_db"]],
            on="config", suffixes=("", "_accum")
        )
        for _, row in merged.iterrows():
            print(f"| {row['config']:<12} | {row['quant_perplexity']:.4f} | "
                  f"{row['delta_perplexity']:+.4f} | "
                  f"{row['avg_qsnr_db']:.2f} dB | "
                  f"{row['avg_qsnr_db_accum']:.2f} dB |")

    # ------------------------------------------------------------------
    # Part 4: Granularity × Sparse Cross-Sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 4: Granularity × Sparse Cross-Sweep")
    print("=" * 60)

    granularities = ["per_tensor", "per_channel", "per_block"]
    outlier_ratios = [0.0, 0.01, 0.05, 0.1, 0.2]
    block_size = 32

    # Shorthand map for naming
    g_suffix = {"per_tensor": "tensor", "per_channel": "channel", "per_block": "block"}

    sparse_configs = []

    for g in granularities:
        for r in outlier_ratios:
            name = f"{g_suffix[g]}-r{r:.2f}"
            kw = dict(
                name=name,
                w_format="int4", a_format="int4",
                w_granularity=g, a_granularity=g,
                outlier_ratio=r,
                quantize_nonlinear=False,
            )
            if g == "per_block":
                kw["w_block_size"] = block_size
                kw["a_block_size"] = block_size
            sparse_configs.append(QuantConfig(**kw))

    total = len(sparse_configs)
    print(f"\n  Running {total} configs ({len(granularities)} granularities"
          f" x {len(outlier_ratios)} outlier_ratios) ...\n")

    study_sparse = Study(sparse_configs, model=model)
    report_sparse = study_sparse.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    print("\n--- Summary (local QSNR) ---")
    print(report_sparse.summary())

    print("\n--- Summary (accum QSNR) ---")
    print(report_sparse.summary(qsnr_type="accum"))

    # ---- Pivot table: rows = granularity, columns = outlier_ratio, values = PPL ----
    df_sparse = report_sparse.summary_dataframe()

    print("\n--- Pivot Table: PPL by Granularity × Outlier Ratio ---")
    print()
    # Header row
    header = f"| {'Granularity':<12} |"
    for r in outlier_ratios:
        header += f" r={r:<5} |"
    print(header)
    sep = f"|{'-'*14}|" + "".join(f"{'-'*9}|" for _ in outlier_ratios)
    print(sep)

    best_per_g: dict[str, tuple[float, float]] = {}

    if df_sparse is not None:
        for g in granularities:
            suffix = g_suffix[g]
            row_str = f"| {suffix:<12} |"
            best_ppl = float("inf")
            best_r = 0.0
            for r in outlier_ratios:
                cfg_name = f"{suffix}-r{r:.2f}"
                match = df_sparse[df_sparse["config"] == cfg_name]
                if not match.empty:
                    ppl = match.iloc[0]["quant_perplexity"]
                    row_str += f" {ppl:>7.4f} |"
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_r = r
                else:
                    row_str += f" {'N/A':>7} |"
            print(row_str)
            best_per_g[g] = (best_r, best_ppl)

    # ---- Best per granularity ----
    print("\n--- Best Outlier Ratio per Granularity (by minimum PPL) ---")
    for g in granularities:
        r, ppl = best_per_g.get(g, (float("nan"), float("nan")))
        print(f"  {g_suffix[g]:<12}  best r={r:.2f}  PPL={ppl:.4f}")

    # ---- Analysis ----
    print("\n--- Analysis ---")
    # Find the per_block(32) r=0.0 baseline for comparison
    block_baseline_ppl = float("nan")
    if df_sparse is not None:
        match = df_sparse[df_sparse["config"] == "block-r0.00"]
        if not match.empty:
            block_baseline_ppl = match.iloc[0]["quant_perplexity"]

    tensor_best = best_per_g.get("per_tensor", (0.0, float("nan")))[1]
    channel_best = best_per_g.get("per_channel", (0.0, float("nan")))[1]
    block_best = best_per_g.get("per_block", (0.0, float("nan")))[1]

    print(f"  per_block(32) r=0.00 baseline PPL:  {block_baseline_ppl:.4f}")
    print(f"  per_tensor best (r={best_per_g['per_tensor'][0]:.2f}):"
          f" PPL={tensor_best:.4f}")
    print(f"  per_channel best (r={best_per_g['per_channel'][0]:.2f}):"
          f" PPL={channel_best:.4f}")
    print(f"  per_block(32)  best (r={best_per_g['per_block'][0]:.2f}):"
          f" PPL={block_best:.4f}")

    can_match = (
        tensor_best <= block_baseline_ppl * 1.05  # within 5%
        if not (math.isnan(tensor_best) or math.isnan(block_baseline_ppl))
        else False
    )
    print()
    print(f"  Can per_tensor + sparse match per_block(32)? "
          f"{'YES' if can_match else 'NO'}")
    print(f"    (per_tensor best {tensor_best:.4f} vs per_block baseline"
          f" {block_baseline_ppl:.4f};"
          f" threshold = {block_baseline_ppl * 1.05:.4f})")
    print()
    print(f"  Optimal sparse degree by granularity:")
    for g in granularities:
        r, ppl = best_per_g.get(g, (float("nan"), float("nan")))
        print(f"    {g_suffix[g]:<12}  r={r:.2f}  (PPL={ppl:.4f})")


if __name__ == "__main__":
    main()
