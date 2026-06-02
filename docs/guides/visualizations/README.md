# Visualizations

交互式 HTML 可视化与配图。浏览器直接打开即可查看。

| 文件 | 内容 |
|------|------|
| `op-quant-config-guide.html` | OpQuantConfig 完全指南 — 算子级二级量化配置的交互式文档，含 storage/compute scheme 分离、role 区分等 |
| `sq-format-algorithm.html` | SQ-Format 算法详解 (ADR-014) — SmoothQuant 逐行推演：scale 粒度、行重排、Hessian 近似等 |
| `granularity-sparse-analysis.html` | Granularity × Sparse 可视化分析 — 粒度模式与 outlier 隔离的交互式 QSNR 对比 |
| `transform-strategy-analysis.html` | Transform 策略分析 — Hadamard / SmoothQuant / GPTQ 三种变换的数值推演与 QSNR 对比 |
| `transform-strategy-report.html` | Transform 策略学术报告 — 上述三种变换的完整学术格式分析报告 |
| `sparse_qsnr_vs_ratio.png` | Sparse QSNR vs Ratio 曲线图 — 三种分布下 outlier_ratio 对 QSNR 的影响 |
| `qsnr-sweep-outlier-amplitude.png` | QSNR vs Outlier 振幅扫描图 — outlier 振幅从 1× 到 50× 时的量化精度变化 |
| `qsnr-sweep-variance.png` | QSNR vs 方差扫描图 — 基底方差从 1 到 10 时的量化精度变化 |
| `transformer_analysis.md` | 系统化误差分析示例 — Transformer + AG News + int4，ADR-010 四阶段闭环全流程 |

生成脚本：
- `scripts/granularity_sparse_analysis.py` → `granularity-sparse-analysis.html`
- `scripts/transform_strategy_analysis.py` → `transform-strategy-analysis.html`
- `scripts/transform_strategy_report.py` → `transform-strategy-report.html`
- `scripts/qsnr_sweep_plots.py` → `qsnr-sweep-*.png`
