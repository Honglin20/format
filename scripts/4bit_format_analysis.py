#!/usr/bin/env python3
"""
4-bit Format Analysis for Shakespeare GPT.

Loads a pretrained Shakespeare character-level GPT, computes FP32 baseline
perplexity, prepares calibration data, and serves as the entry point for a
4-part analysis of 4-bit quantization formats.

Run:  PYTHONPATH=. python scripts/4bit_format_analysis.py
"""

from __future__ import annotations

import copy
import os

import torch

from scripts.transformer_quant_study import (
    MiniGPT,
    make_dataloaders,
    eval_fn,
)
from src.session import Study, QuantConfig


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

    study = Study(mxint_configs, model=copy.deepcopy(model))
    report = study.run(
        calib_data,
        eval_data=val_loader,
        eval_fn=eval_fn,
        outputs="default",
    )

    print("\n--- Summary (local QSNR) ---")
    report.print_summary()

    print("\n--- Summary (accum QSNR) ---")
    report.print_summary(qsnr_type="accum")

    # --- Markdown table ---
    df_local = report.summary_dataframe(qsnr_type="local")
    df_accum = report.summary_dataframe(qsnr_type="accum")

    print("\n--- MXINT Comparison Table ---")
    print()
    print(f"| Config | PPL | ΔPPL | QSNR (local) | QSNR (accum) |")
    print(f"|--------|-----|------|--------------|---------------|")
    print(f"| FP32 baseline | {fp32_ppl:.4f} | — | — | — |")

    if df_local is not None and df_accum is not None:
        for idx in range(len(df_local)):
            config = df_local.iloc[idx]["config"]
            ppl = df_local.iloc[idx]["quant_perplexity"]
            dppl = df_local.iloc[idx]["delta_perplexity"]
            qsnr_local = df_local.iloc[idx]["avg_qsnr_db"]
            qsnr_accum = df_accum.iloc[idx]["avg_qsnr_db"]
            print(f"| {config} | {ppl:.4f} | {dppl:+.4f} | {qsnr_local:.2f} | {qsnr_accum:.2f} |")

    # ------------------------------------------------------------------
    # Part 2: W4A4 with Hadamard / SmoothQuant transforms
    # ------------------------------------------------------------------
    print("\n[Part 2] W4A4 + Hadamard / SmoothQuant — TBD")

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
