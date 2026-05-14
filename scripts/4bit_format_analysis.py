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
import sys

import torch
import torch.nn.functional as F

from scripts.transformer_quant_study import (
    MiniGPT,
    download_shakespeare,
    make_dataloaders,
    eval_fn,
)


def main() -> None:
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
    print(f"  Expected FP32 PPL:       ~7.95")
    print()

    # ------------------------------------------------------------------
    # Part 1: Uniform W4A4 — sweep over granularity (per-block, per-channel, per-tensor)
    # ------------------------------------------------------------------
    print("\n[Part 1] Uniform W4A4 — TBD")

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
