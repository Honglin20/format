"""
Format Study — User Entry Point

Replace the four functions below with your own model and data,
then run directly or call run_format_study() programmatically.

To customise the search space, edit src/pipeline/studies/format_study.py.
"""
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")

from src.pipeline.format_study import run_format_study, plot_from_results


# ---------------------------------------------------------------------------
# Replace these functions with your own model and data
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    from examples._model import ToyMLP
    return ToyMLP()


def build_conv_model() -> nn.Module:
    from examples._model import ToyConvNet
    return ToyConvNet()


def make_calib_data(num_samples: int = 256, batch_size: int = 16) -> List[torch.Tensor]:
    return [torch.randn(batch_size, 128) for _ in range(num_samples // batch_size)]


def make_eval_loader(num_samples: int = 512, batch_size: int = 16) -> DataLoader:
    x = torch.randn(num_samples, 128)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def eval_fn(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Format Precision Study")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-part-a", action="store_true")
    parser.add_argument("--skip-part-b", action="store_true")
    parser.add_argument("--skip-part-c", action="store_true")
    parser.add_argument("--skip-part-d", action="store_true")
    parser.add_argument("--skip-part-d-conv", action="store_true")
    parser.add_argument("--plot-from", default=None, metavar="RESULTS_JSON")
    args = parser.parse_args()

    if args.plot_from:
        plot_from_results(args.plot_from, args.output_dir)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        skip = {k: True for k, v in {
            "A": args.skip_part_a, "B": args.skip_part_b,
            "C": args.skip_part_c, "D": args.skip_part_d,
            "D_conv": args.skip_part_d_conv,
        }.items() if v}
        run_format_study(
            build_model,
            lambda: make_calib_data(args.calib_samples, args.batch_size),
            lambda: make_eval_loader(args.eval_samples, args.batch_size),
            eval_fn,
            build_conv_model=build_conv_model,
            output_dir=args.output_dir,
            skip_parts=skip or None,
        )
