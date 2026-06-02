# 4bit 量化格式对比分析 — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 单个 Python 脚本完成 Shakespeare GPT 在 MXINT/MXFP/NF4 + granularity × sparse 下的全面对比分析，输出 markdown 表格和可视化图表。

**Architecture:** 纯消费层脚本，通过 `Study`/`Session`/`QuantConfig` 高层 API 编排已有能力，不修改 `src/`。模型定义从 `scripts/transformer_quant_study.py` 导入，权重从 `scripts/weights/shakespeare_gpt.pt` 加载。

**Tech Stack:** PyTorch + 项目内 `src/session`、`src/formats`、`src/scheme`、`src/viz`

---

### Task 1: 创建分析脚本骨架 + 模型加载

**Files:**
- Create: `scripts/4bit_format_analysis.py`

**Step 1: 创建脚本文件，含模型加载和验证**

```python
"""
4-bit Format Comparative Analysis — Shakespeare GPT.

Four-part analysis:
  1. MXINT precision: W8A8 / W4A8 / W4A4
  2. Root cause: per-layer per-role distribution diagnosis + visualization
  3. MXFP / NF4 cross-format comparison
  4. Granularity × Sparse cross-sweep (outlier_ratio)

Run:  PYTHONPATH=. python scripts/4bit_format_analysis.py
"""
from __future__ import annotations

import copy
import math
import os
import sys

import torch
import torch.nn.functional as F

from src.session import Study, Session, QuantConfig


# ---- Model definition (copied from scripts/transformer_quant_study.py) ----
# [MiniGPT + TransformerBlock + CausalSelfAttention classes here]
# We import from the existing script to avoid duplication:
from scripts.transformer_quant_study import (
    MiniGPT, download_shakespeare, make_dataloaders, eval_fn
)


def main():
    torch.manual_seed(42)

    # Data
    print("=== Loading data ===")
    train_loader, val_loader, vocab_size = make_dataloaders(
        block_size=128, batch_size=32)
    print(f"  Vocab: {vocab_size}, Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    # Model
    print("\n=== Loading pretrained model ===")
    weights_path = os.path.join(os.path.dirname(__file__),
                                "weights", "shakespeare_gpt.pt")
    ckpt = torch.load(weights_path, map_location="cpu")
    model = MiniGPT(vocab_size=vocab_size, d_model=192, n_heads=3,
                    n_layers=4, block_size=128)
    model.load_state_dict(ckpt)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # FP32 baseline
    val_loss, val_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            val_loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                reduction="sum").item()
            val_tokens += y.numel()
    fp32_ppl = math.exp(val_loss / val_tokens)
    print(f"  FP32 PPL: {fp32_ppl:.4f}")

    # Calibration data
    calib_data = []
    for x, _y in train_loader:
        calib_data.append(x)
        if len(calib_data) >= 4:
            break
    print(f"  Calib batches: {len(calib_data)}")

    # ---- Part 1: MXINT precision table ----
    # ... (next tasks)
```

**Step 2: 运行脚本验证加载链路**

Run: `PYTHONPATH=. python scripts/4bit_format_analysis.py`
Expected: 打印 FP32 PPL ~7.95，无报错

**Step 3: Commit**

```bash
git add scripts/4bit_format_analysis.py
git commit -m "feat: add 4bit analysis script skeleton with model loading"
```

---

### Task 2: Part 1 — MXINT 精度对比表 (W8A8 / W4A8 / W4A4)

**Files:**
- Modify: `scripts/4bit_format_analysis.py`

**Step 1: 添加 Part 1 configs 和 Study 运行**

在 `main()` 中 calib_data 之后追加：

```python
    # =====================================================================
    # Part 1: MXINT Precision Comparison (W8A8 / W4A8 / W4A4)
    # =====================================================================
    print("\n" + "=" * 60)
    print("Part 1: MXINT Precision Comparison")
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

    study1 = Study(mxint_configs, model=copy.deepcopy(model))
    report1 = study1.run(calib_data, eval_data=val_loader, eval_fn=eval_fn,
                         outputs="default")

    print("\n--- Part 1: Summary ---")
    report1.print_summary()
    report1.print_summary(qsnr_type="accum")
```

**Step 2: 将结果格式化为 markdown 表格**

在 `report1` 之后追加 markdown 表格生成代码：

```python
    # Markdown table
    df = report1.summary_dataframe()
    lines = []
    lines.append("## Part 1: MXINT Precision Comparison")
    lines.append("")
    lines.append(f"FP32 baseline perplexity: **{fp32_ppl:.4f}**")
    lines.append("")
    lines.append("| Config | PPL | ΔPPL | QSNR (local) | QSNR (accum) |")
    lines.append("|--------|-----|------|-------------|-------------|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['config']} | {row['quant_perplexity']:.4f} | "
            f"{row['delta_perplexity']:+.4f} | "
            f"{row['avg_qsnr_db']:.1f} dB | "
            f"{row.get('avg_qsnr_accum_db', 'N/A')} |"
        )
    print("\n".join(lines))
```

**Step 3: 运行验证 Part 1 输出**

Run: `PYTHONPATH=. python scripts/4bit_format_analysis.py`
Expected: 三种 config 精度递减 (W8A8 > W4A8 > W4A4)

**Step 4: Commit**

```bash
git add scripts/4bit_format_analysis.py
git commit -m "feat: add Part 1 MXINT W8A8/W4A8/W4A4 precision comparison"
```

---

### Task 3: Part 2 — 精度下降根因分析 + 可视化

**Files:**
- Modify: `scripts/4bit_format_analysis.py`

**Step 1: 添加 per-layer per-role QSNR 诊断**

```python
    # =====================================================================
    # Part 2: Root Cause Analysis — per-layer per-role diagnosis
    # =====================================================================
    print("\n" + "=" * 60)
    print("Part 2: Root Cause Analysis")
    print("=" * 60)

    # Run a single Session for detailed analysis on W4A4
    cfg_w4a4 = QuantConfig(
        name="MXINT-W4A4-detail",
        w_format="int4", a_format="int4",
        w_granularity="per_block", a_granularity="per_block",
        w_block_size=32, a_block_size=32,
        quantize_nonlinear=False,
    )
    session_w4a4 = Session(copy.deepcopy(model), cfg_w4a4)
    result_w4a4 = session_w4a4.run(calib_data, eval_data=val_loader,
                                    eval_fn=eval_fn)

    # Diagnose: per-role per-layer QSNR
    diag = result_w4a4.diagnose

    # Top-K worst layers by accumulated QSNR
    print("\nTop-10 worst layers (accumulated QSNR):")
    for name, qsnr in diag.top_k_qsnr(10, role="output"):
        print(f"  {name}: {qsnr:.1f} dB")

    # Per-role error sources (weight vs input)
    print("\nPer-role QSNR summary:")
    print(f"  Weight QSNR avg: {diag.role_summary('weight'):.1f} dB")
    print(f"  Input QSNR avg:  {diag.role_summary('input'):.1f} dB")
    print(f"  Output QSNR avg: {diag.role_summary('output'):.1f} dB")

    # Layer-type analysis
    print("\nLayer-type breakdown:")
    layer_type_qsnr = diag.by_layer_type()
    for ltype, info in layer_type_qsnr.items():
        print(f"  {ltype}: avg QSNR={info['avg_qsnr']:.1f} dB, "
              f"worst={info['worst_layer']} ({info['worst_qsnr']:.1f} dB)")

    # Error source attribution table
    print("\n--- Part 2: Error Source Attribution ---")
    sources = diag.error_sources()
    lines2 = ["| Layer | Role | QSNR (local) | QSNR (accum) | Degradation |"]
    lines2.append("|-------|------|-------------|-------------|-------------|")
    for s in sources[:20]:  # top 20 entries
        lines2.append(
            f"| {s['layer']} | {s['role']} | {s['qsnr_local']:.1f} dB | "
            f"{s['qsnr_accum']:.1f} dB | {s.get('degradation', 'N/A')} |"
        )
    print("\n".join(lines2))
```

**Step 2: 添加分布特征诊断（Characterize）**

```python
    # Characterize: distribution degradation classification
    char = result_w4a4.characterize
    print("\n--- Part 2: Distribution Degradation ---")
    degradation = char.classify()
    for layer_name, info in list(degradation.items())[:10]:
        print(f"  {layer_name}: type={info['degradation_type']}, "
              f"severity={info.get('severity', 'N/A')}, "
              f"weight_skew={info.get('weight_skewness', 'N/A'):.2f}, "
              f"input_skew={info.get('input_skewness', 'N/A'):.2f}")
```

**Step 3: 添加 8bit vs 4bit 分布叠加可视化**

```python
    # Run W8A8 session for comparison visualizations
    cfg_w8a8 = QuantConfig(
        name="MXINT-W8A8-detail",
        w_format="int8", a_format="int8",
        w_granularity="per_block", a_granularity="per_block",
        w_block_size=32, a_block_size=32,
        quantize_nonlinear=False,
    )
    session_w8a8 = Session(copy.deepcopy(model), cfg_w8a8)
    result_w8a8 = session_w8a8.run(calib_data, eval_data=val_loader,
                                    eval_fn=eval_fn)

    # Identify worst layers from W4A4
    worst_layers = [name for name, _ in diag.top_k_qsnr(5, role="output")]

    # Generate overlay histograms for worst layers
    out_dir = os.path.join(os.path.dirname(__file__), "analysis_output")
    os.makedirs(out_dir, exist_ok=True)

    for layer_name in worst_layers:
        # Weight histogram: FP32 (original) + INT8 quantized + INT4 quantized
        result_w4a4.plot.weight_histogram_overlay(
            layer_name,
            reference_result=result_w8a8,
            save_path=os.path.join(out_dir, f"hist_weight_{layer_name}.png"),
        )
        # Input activation histogram
        result_w4a4.plot.input_histogram_overlay(
            layer_name,
            reference_result=result_w8a8,
            save_path=os.path.join(out_dir, f"hist_input_{layer_name}.png"),
        )

    # Per-role QSNR comparison bar chart (W8A8 vs W4A4)
    result_w4a4.plot.per_role_qsnr_comparison(
        reference_result=result_w8a8,
        reference_label="INT8",
        quantized_label="INT4",
        save_path=os.path.join(out_dir, "per_role_qsnr_comparison.png"),
    )

    # Per-channel QSNR boxplot for worst layer
    result_w4a4.plot.channel_heterogeneity(
        worst_layers[0],
        save_path=os.path.join(out_dir, "channel_heterogeneity.png"),
    )

    print(f"\n  Visualizations saved to: {out_dir}")
```

**Step 4: 运行脚本验证 Part 2 输出**

Run: `PYTHONPATH=. python scripts/4bit_format_analysis.py`
Expected: 打印 per-layer QSNR，生成 PNG 图表文件

**Step 5: Commit**

```bash
git add scripts/4bit_format_analysis.py
git commit -m "feat: add Part 2 root cause analysis with distribution visualization"
```

---

### Task 4: Part 3 — MXFP / NF4 格式对比

**Files:**
- Modify: `scripts/4bit_format_analysis.py`

**Step 1: 添加跨格式 configs 和 Study**

```python
    # =====================================================================
    # Part 3: MXFP / NF4 Format Comparison at 4-bit
    # =====================================================================
    print("\n" + "=" * 60)
    print("Part 3: MXFP / NF4 Format Comparison")
    print("=" * 60)

    format_configs = [
        # MXINT baseline
        QuantConfig(
            name="MXINT-4",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        # MXFP-4
        QuantConfig(
            name="MXFP-4",
            w_format="fp4_e2m1", a_format="fp4_e2m1",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
        # NF4 weight-only
        QuantConfig(
            name="NF4-W",
            w_format="nf4", w_granularity="per_channel",
            weight_only=True,
            quantize_nonlinear=False,
        ),
        # NF4 weight + activation
        QuantConfig(
            name="NF4-WA",
            w_format="nf4", a_format="nf4",
            w_granularity="per_channel", a_granularity="per_channel",
            quantize_nonlinear=False,
        ),
        # MXFP-8 upper bound
        QuantConfig(
            name="MXFP-8 (ref)",
            w_format="fp8_e4m3", a_format="fp8_e4m3",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            quantize_nonlinear=False,
        ),
    ]

    study3 = Study(format_configs, model=copy.deepcopy(model))
    report3 = study3.run(calib_data, eval_data=val_loader, eval_fn=eval_fn,
                         outputs="default")

    print("\n--- Part 3: Format Comparison Summary ---")
    report3.print_summary()
    report3.print_summary(qsnr_type="accum")

    # Markdown table
    df3 = report3.summary_dataframe()
    lines3 = []
    lines3.append("## Part 3: MXFP / NF4 Format Comparison (4-bit)")
    lines3.append("")
    lines3.append("| Format | PPL | ΔPPL | QSNR (local) | QSNR (accum) |")
    lines3.append("|--------|-----|------|-------------|-------------|")
    for _, row in df3.iterrows():
        lines3.append(
            f"| {row['config']} | {row['quant_perplexity']:.4f} | "
            f"{row['delta_perplexity']:+.4f} | {row['avg_qsnr_db']:.1f} dB | "
            f"{row.get('avg_qsnr_accum_db', 'N/A')} |"
        )
    print("\n".join(lines3))
```

**Step 2: 运行验证 Part 3**

Run: `PYTHONPATH=. python scripts/4bit_format_analysis.py`
Expected: NF4-W 精度最高（weight_only），NF4-WA 看激活量化影响，MXFP-4 vs MXINT-4 对比

**Step 3: Commit**

```bash
git add scripts/4bit_format_analysis.py
git commit -m "feat: add Part 3 MXFP/NF4 cross-format comparison"
```

---

### Task 5: Part 4 — Granularity × Sparse 交叉扫描

**Files:**
- Modify: `scripts/4bit_format_analysis.py`

**Step 1: 添加交叉扫描 configs 和 Study**

```python
    # =====================================================================
    # Part 4: Granularity × Sparse Cross-Sweep
    # =====================================================================
    print("\n" + "=" * 60)
    print("Part 4: Granularity × Sparse Cross-Sweep")
    print("=" * 60)

    outlier_ratios = [0.0, 0.01, 0.05, 0.1, 0.2]
    sweep_configs = []

    # per_tensor + sparse
    for r in outlier_ratios:
        sweep_configs.append(QuantConfig(
            name=f"tensor-r{r:.2f}",
            w_format="int4", a_format="int4",
            w_granularity="per_tensor", a_granularity="per_tensor",
            outlier_ratio=r,
            quantize_nonlinear=False,
        ))

    # per_channel + sparse
    for r in outlier_ratios:
        sweep_configs.append(QuantConfig(
            name=f"channel-r{r:.2f}",
            w_format="int4", a_format="int4",
            w_granularity="per_channel", a_granularity="per_channel",
            outlier_ratio=r,
            quantize_nonlinear=False,
        ))

    # per_block + sparse
    for r in outlier_ratios:
        sweep_configs.append(QuantConfig(
            name=f"block-r{r:.2f}",
            w_format="int4", a_format="int4",
            w_granularity="per_block", a_granularity="per_block",
            w_block_size=32, a_block_size=32,
            outlier_ratio=r,
            quantize_nonlinear=False,
        ))

    study4 = Study(sweep_configs, model=copy.deepcopy(model))
    report4 = study4.run(calib_data, eval_data=val_loader, eval_fn=eval_fn,
                         outputs="default")

    print("\n--- Part 4: Granularity × Sparse Summary ---")
    report4.print_summary()
    report4.print_summary(qsnr_type="accum")

    # Build pivot table: rows=granularity, cols=outlier_ratio, values=PPL
    # Build pivot table
    df4 = report4.summary_dataframe()
    pivot = {}
    for _, row in df4.iterrows():
        name = row["config"]
        ppl = row["quant_perplexity"]
        # Parse granularity and ratio from name like "tensor-r0.05"
        parts = name.split("-r")
        gran = parts[0]
        ratio = float(parts[1])
        if gran not in pivot:
            pivot[gran] = {}
        pivot[gran][ratio] = ppl

    lines4 = []
    lines4.append("## Part 4: Granularity × Sparse Cross-Sweep (INT4)")
    lines4.append("")
    header = "| Granularity | " + " | ".join(f"r={r:.2f}" for r in outlier_ratios) + " |"
    lines4.append(header)
    sep = "|" + "|".join(["---"] * (len(outlier_ratios) + 1)) + "|"
    lines4.append(sep)
    for gran in ["tensor", "channel", "block"]:
        vals = " | ".join(f"{pivot[gran].get(r, float('nan')):.4f}"
                         for r in outlier_ratios)
        lines4.append(f"| {gran} | {vals} |")
    print("\n".join(lines4))

    # Best sparse degree per granularity
    print("\nBest outlier_ratio per granularity:")
    for gran in ["tensor", "channel", "block"]:
        best_r = min(pivot[gran], key=pivot[gran].get)
        best_ppl = pivot[gran][best_r]
        print(f"  {gran}: r={best_r:.2f} → PPL={best_ppl:.4f}")
```

**Step 2: 运行验证 Part 4**

Run: `PYTHONPATH=. python scripts/4bit_format_analysis.py`
Expected: per_block 明显优于 per_channel 优于 per_tensor，sparse 对 per_tensor 改善最明显

**Step 3: Commit**

```bash
git add scripts/4bit_format_analysis.py
git commit -m "feat: add Part 4 granularity × sparse cross-sweep"
```

---

### Task 6: 综合输出 + 最终结论

**Files:**
- Modify: `scripts/4bit_format_analysis.py`

**Step 1: 添加全部结果汇总 markdown 文件生成**

在 `main()` 末尾：

```python
    # =====================================================================
    # Consolidated output
    # =====================================================================
    output_md = os.path.join(out_dir, "analysis_report.md")
    with open(output_md, "w") as f:
        f.write("# 4-bit Quantization Format Analysis Report\n\n")
        f.write(f"**Model**: Shakespeare MiniGPT (d=192, n_layers=4)\n")
        f.write(f"**FP32 PPL**: {fp32_ppl:.4f}\n\n")

        f.write("## Part 1: MXINT Precision\n\n")
        f.write("\n".join(lines) + "\n\n")

        f.write("## Part 2: Error Source Attribution\n\n")
        f.write("\n".join(lines2) + "\n\n")

        f.write("## Part 3: Format Comparison\n\n")
        f.write("\n".join(lines3) + "\n\n")

        f.write("## Part 4: Granularity × Sparse\n\n")
        f.write("\n".join(lines4) + "\n\n")

        f.write("## Conclusions\n\n")
        f.write("1. **Best 4-bit format**: ...\n")
        f.write("2. **Optimal granularity + sparse**: ...\n")
        f.write("3. **Mixed-precision recommendation**: ...\n")

    print(f"\nReport saved to: {output_md}")
```

**Step 2: 全流程运行验证**

Run: `PYTHONPATH=. python scripts/4bit_format_analysis.py`
Expected: 四部分全部输出，markdown 报告生成，PNG 图表保存在 `scripts/analysis_output/`

**Step 3: Commit**

```bash
git add scripts/4bit_format_analysis.py
git commit -m "feat: add consolidated report output and conclusions"
```

---

### Task 7: E2E 验证

**Step 1: 清理运行**

```bash
rm -rf scripts/analysis_output
PYTHONPATH=. python scripts/4bit_format_analysis.py
```

Expected:
- FP32 PPL ~7.95
- Part 1: W8A8 PPL ~8.0, W4A8 PPL ~8.3, W4A4 PPL ~8.4
- Part 2: Top layers 应包含 attention/qkv、ffn 等模块
- Part 3: NF4-W 精度接近 W8A8，NF4-WA 下降明显
- Part 4: block > channel > tensor，sparse 对 tensor 改善最大

**Step 2: 验证图表文件**

```bash
ls scripts/analysis_output/
```

Expected: `hist_weight_*.png`, `hist_input_*.png`, `per_role_qsnr_comparison.png`, `channel_heterogeneity.png`, `analysis_report.md`

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: verify full analysis pipeline, add expected outputs"
```

---

## 测试策略

- 本脚本是分析脚本，不涉及 `src/` 变更
- 验证方式：全流程运行 + 人工审核输出合理性
- 合理性判据（基于已知 baseline 2026-05-12）：
  - Shakespeare GPT FP32 PPL = 7.95
  - INT4 per_block PPL 增量 < +1.0
  - INT8 per_block PPL 增量 < +0.3
  - NF4 weight_only PPL 增量 < +0.3
