# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建。将量化拆为**格式 × 粒度 × 变换**三个正交轴，一个 `QuantConfig` 控制一切。

## 研究示例：4-bit 格式对比分析

本分支包含对 Shakespeare GPT (1.82M params) 的完整 4-bit 格式分析，回答一个问题：**4-bit 下数据格式应该怎么选。**

```bash
cd microxcaling
PYTHONPATH=. python scripts/4bit_format_analysis.py
```

输出写入 `scripts/analysis_output/`（报告 + 可视化）。

### API 使用方式

脚本展示了库的两个核心入口——`Study`（多配置对比）和 `Session`（单配置深度分析）：

```python
from src.session import Session, Study, QuantConfig

# QuantConfig = format × granularity × transform 三轴组合
cfg_w8a8 = QuantConfig(name="W8A8", w_format="int8", a_format="int8",
                       w_granularity="per_block", a_granularity="per_block",
                       w_block_size=32, a_block_size=32)
cfg_w4a4 = QuantConfig(name="W4A4", w_format="int4", a_format="int4",
                       w_granularity="per_block", a_granularity="per_block",
                       w_block_size=32, a_block_size=32)

# ── Study API：多配置对比，一张表出结果 ──
study = Study([cfg_w8a8, cfg_w4a4], model=model)
report = study.run(calib_data, eval_data=val_loader, eval_fn=eval_fn)
print(report.summary())                          # local QSNR
print(report.summary(qsnr_type="accum"))         # end-to-end accumulated QSNR
df = report.summary_dataframe()                  # → pandas DataFrame

# ── Session API：单配置深度诊断 ──
import copy
session = Session(model=copy.deepcopy(model), config=cfg_w4a4)
result = session.run(calib_data, eval_data=val_loader, eval_fn=eval_fn,
                     outputs=["accuracy", "qsnr", "distribution", "histogram"])

# 诊断：逐层逐角色 QSNR 归因（input / weight / output）
print(result.diagnose.per_role_table(max_layers=20))
print(result.diagnose.summary())

# 表征：分布退化分类
print(result.characterize.causal_analysis())

# 可视化
result.plot.qsnr_comparison().savefig("qsnr_comparison.png", dpi=300, bbox_inches="tight")
result.plot.per_role_qsnr_bars().savefig("per_role_qsnr.png", dpi=300, bbox_inches="tight")
result.plot.channel_heterogeneity("blocks.0.attn.qkv", role="weight")

# ── 交叉格式对比：MXFP / NF4 ──
QuantConfig(name="NF4-W", w_format="nf4", w_granularity="per_channel",
            weight_only=True)
QuantConfig(name="MXFP-4", w_format="fp4_e2m1", w_granularity="per_block",
            w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
            a_block_size=32)

# ── 稀疏离群值扫描 ──
for r in [0.0, 0.01, 0.05, 0.1, 0.2]:
    QuantConfig(name=f"sparse-r{r:.2f}", w_format="int4", a_format="int4",
                w_granularity="per_block", a_granularity="per_block",
                w_block_size=32, a_block_size=32, outlier_ratio=r)
```

### 核心结论

| 结论 | 数据 |
|------|------|
| W4A8 混合精度足够好 | PPL 8.02 vs FP32 7.95（远优于 W4A4 的 8.43） |
| per_block 在 4-bit 下是必须的 | per_tensor 即使加 sparse 也只有 PPL 17.91 |
| outlier_ratio=0.10 可进一步提升 | per_block W4A4 PPL 从 8.43 → 8.15 |
| NF4 仅权重可用，激活上灾难 | NF4-W PPL 8.07 vs NF4-WA PPL 11.37 |
| MXFP-4 略优于 MXINT-4 | 差距仅 ~0.02 PPL |

详见 [`scripts/analysis_output/analysis_report.md`](scripts/analysis_output/analysis_report.md)。

## 文档导航

→ [docs/INDEX.md](docs/INDEX.md) — 全部文档索引
→ [docs/status/CURRENT.md](docs/status/CURRENT.md) — 当前进度 & 断点续传
