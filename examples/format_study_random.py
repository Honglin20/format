"""
Format Study — Random Tensor Validation

用随机生成的 tensor 验证量化格式精度结论：

  1. 4-bit INT4 FP32 scale 显著优于 PoT scale（~5-8 dB QSNR 差距）
  2. 8-bit 格式差异远小于 4-bit
  3. Hadamard transform 在 per_block 格式上有增益
  4. scale_format 对比：fp32 vs pot 对 per_channel 影响最大

运行：
    python examples/format_study_random.py

不需要真实数据集。用随机输入和随机初始化的 MLP 验证量化误差。
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.format_study import run_format_study, plot_from_results


# ---------------------------------------------------------------------------
# User-provided functions — replace these with your own model and data
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64),
    )


def make_calib_data(num_samples: int = 128, batch_size: int = 16):
    torch.manual_seed(42)
    return [torch.randn(batch_size, 64) for _ in range(num_samples // batch_size)]


def make_eval_loader(num_samples: int = 256, batch_size: int = 16):
    x = torch.randn(num_samples, 64)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def eval_fn(model: nn.Module, data):
    """统一的前向推理函数 — 处理 calibration / analysis / evaluation 三种场景。

    校准和分析阶段 data 是 List[Tensor]，只需前向传播（返回值被忽略）。
    评估阶段 data 是 DataLoader[(input, label)]，返回精度指标。
    """
    model.eval()

    # Calibration / analysis: data is List[Tensor], just run forward pass
    if isinstance(data, list):
        with torch.no_grad():
            for batch in data:
                model(batch)
        return {}

    # Evaluation: data is DataLoader, compute metrics
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# Focused study config — demonstrates key conclusions on random tensors
# ---------------------------------------------------------------------------

STUDY_CONFIG = {
    # ── Core conclusion: FP32 scale vs PoT scale at 4-bit ──
    "core_4bit_scale": {
        "description": "Core: INT4 per_channel — FP32 scale vs PoT scale",
        "configs": [
            {"name": "INT4-PC-FP32", "format": "int4", "granularity": "per_channel",
             "axis": -1, "scale_format": "fp32"},
            {"name": "INT4-PC-PoT",  "format": "int4", "granularity": "per_channel",
             "axis": -1, "scale_format": "pot"},
        ],
        "output": {
            "tables": ["accuracy", "pot_delta"],
            "figures": ["qsnr_line", "pot_delta_bar"],
        },
    },

    # ── 4-bit format overview ──
    "4bit_formats": {
        "description": "4-bit Format Overview",
        "configs": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": -1, "weight_only": True},
        ],
        "output": {
            "tables": ["accuracy"],
            "figures": ["qsnr_line", "mse_box"],
        },
    },

    # ── 8-bit format overview (differences should be small) ──
    "8bit_formats": {
        "description": "8-bit Format Overview",
        "configs": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 32},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
        ],
        "output": {
            "tables": ["accuracy"],
            "figures": ["qsnr_line", "mse_box"],
        },
    },

    # ── Hadamard transform effect at 4-bit ──
    "4bit_transform": {
        "description": "Hadamard transform effect on 4-bit formats",
        "configs": [
            {"name": "MXINT-4",      "format": "int4",     "granularity": "per_block", "block_size": 32},
            {"name": "MXINT-4-Had",  "format": "int4",     "granularity": "per_block", "block_size": 32,
             "transform": "hadamard"},
            {"name": "MXFP-4",       "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32},
            {"name": "MXFP-4-Had",   "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32,
             "transform": "hadamard"},
        ],
        "output": {
            "tables": ["accuracy"],
            "figures": ["qsnr_line", "transform_delta"],
        },
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format Study — Random Tensor Validation")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: results/random_tensor_study)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--plot-from", default=None, metavar="RESULTS_JSON",
                        help="Regenerate figures from existing results.json")
    args = parser.parse_args()

    if args.plot_from:
        plot_from_results(args.plot_from, args.output_dir)
    else:
        torch.manual_seed(args.seed)
        results = run_format_study(
            build_model=build_model,
            make_calib_data=lambda: make_calib_data(args.calib_samples, args.batch_size),
            make_eval_loader=lambda: make_eval_loader(args.eval_samples, args.batch_size),
            eval_fn=eval_fn,
            config=STUDY_CONFIG,
            output_dir=args.output_dir or "results/random_tensor_study",
        )
