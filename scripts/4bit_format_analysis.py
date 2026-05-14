#!/usr/bin/env python3
"""
4-bit Format Analysis for Shakespeare GPT.

Loads a pretrained Shakespeare character-level GPT, computes FP32 baseline
perplexity, prepares calibration data, and serves as the entry point for a
4-part analysis of 4-bit quantization formats.

Run:  PYTHONPATH=. python scripts/4bit_format_analysis.py
"""

from __future__ import annotations

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
    for layer, roles in sorted(result_w4a4.observers_data.items()):
        if printed >= 20:
            break
        for role in ("input", "weight", "output"):
            if printed >= 20:
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
        outputs=["accuracy", "qsnr"],
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

    # 5d: Histogram overlay via viz.figures
    print("  (d) Histogram overlay")
    try:
        from src.viz.figures import histogram_overlay

        all_results = {
            "part_1": {
                "MXINT-W4A4-detail": {"report": result_w4a4.report},
            },
        }
        fig = histogram_overlay(all_results, output_dir=analysis_dir)
        plt.close(fig)
        print(f"      -> {analysis_dir}/histogram_overlay.png")
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
            if qsnr is not None and qsnr == qsnr and qsnr != float("-inf"):
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
    # Part 3: Weight-only NF4 + FP8 evaluation
    # ------------------------------------------------------------------
    print("\n[Part 3] Weight-only NF4 + FP8 — TBD")

    # ------------------------------------------------------------------
    # Part 4: Best configs + comparative summary
    # ------------------------------------------------------------------
    print("\n[Part 4] Best configs + summary — TBD")


if __name__ == "__main__":
    main()
