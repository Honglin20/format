# 可视化与表格的 Role 区分规范

> 所有绘图和表格输出必须显式声明 tensor role（input / weight / output / bias）。
> 禁止在任何面向用户的输出中隐藏 role 信息。

## 核心规则

### R1: 函数签名使用 `roles`（复数）

多 role 函数接受 `roles` tuple 参数，默认值包含全部三个主要 role：

```python
# ✓ 正确
def outlier_analysis(all_results, *, roles=("input", "weight", "output"), ...):
    ...

# ✗ 错误 — 单 role 默认值隐藏了维度
def outlier_analysis(all_results, *, role="input", ...):
    ...
```

### R2: 标题/标签必须包含 role

每个子图、表格列、轴标签必须标明对应哪个 role：

```
# ✓ 正确
ax.set_title(f"{layer}\n[{role}] QSNR={qsnr:.1f}dB")

# ✗ 错误 — 用户不知道这是哪个 role 的数据
ax.set_title(f"{layer}\nQSNR={qsnr:.1f}dB")
```

Table headers 格式：`"Table N: Description (output)"` — role 在括号中。

### R3: 文件名不含 role

多 role figure 生成单个 PNG 文件，不按 role 拆分：

```
# ✓ 正确 — 一个 figure 包含所有 role 的子图
crest_vs_qsnr.png

# ✗ 错误 — 按 role 拆分导致文件数量爆炸
crest_vs_qsnr_input.png
crest_vs_qsnr_weight.png
crest_vs_qsnr_output.png
```

### R4: 聚合数据的 role 必须声明

当数据源已限定 role（如 `SessionResult.qsnr_per_layer` 来自 `_extract_qsnr_mse(role="output")`），消费者必须在标题/表头中标明：

```python
# _tables.py — 终端表格 header
"Per-Layer QSNR (dB, output) — Lower = more quantization-sensitive"

# _plot.py — matplotlib 标题
"QSNR per Layer (output)"
```

不允许出现 "QSNR per Layer" 这种让用户猜测 role 的标题。

## 设计依据

**为什么 `qsnr_per_layer` 默认取 `output`？**

- **Output QSNR** 衡量层的端到端量化质量，包含 input + weight 量化的综合效应
- **Input QSNR** 测量的是已量化数据的再量化误差（除第一层外——上层浮点输出已是量化后的近似值），是虚荣指标
- **Weight QSNR** 是静态测量，与深度/误差传播无关

实测验证（7 层 MLP + int8 量化 + bf16 storage）：
- Output QSNR 跨层斜率 ≈ 0 dB（无误差累积）
- Input QSNR 跨层斜率 ≈ +4 dB（随深度反而改善，因上层输出已量化）
- Weight QSNR ≈ 44 dB（恒定）

**当需要多 role 分析时**，使用直接访问 observer 数据的函数（如 `per_layer_role_histogram`）。

## 检查清单

新增或修改任何绘图/表格函数时，逐项确认：

- [ ] 函数签名使用 `roles` tuple 参数（非 `role` 单值）
- [ ] 所有子图标题包含 `[role]` 标识
- [ ] Table header 包含 `(output)` 或对应 role
- [ ] 单 role 数据源的 figure/table 标题声明了 role
- [ ] 多 role figure 的 PNG 文件名不含 role（一个文件含所有 role）
- [ ] 注册到 `_registry.py` 的 title 字段包含 role 标注
- [ ] `_spec.py` 的 observer 依赖正确声明

## 误差传播分析专用规范

`error_propagation()`、`accumulated_vs_local()` 和 `error_source_analysis()` 涉及两条数据源（hook 累积 QSNR + observer 本地 QSNR），额外要求：

### R5: 同时标注数据来源

每个可视化元素必须明确标注是来自 hook（累积）还是 observer（本地）：

```
# ✓ 正确
ax.set_ylabel("QSNR (dB)")；legend(["Accumulated (true_error)", "Local (observer)"])

# ✗ 错误 — 两个不同来源的数据共用同一标签
ax.set_ylabel("QSNR (dB)")；legend(["QSNR", "QSNR"])
```

### R6: 诊断阈值统一

| 诊断 | Headroom 范围 | 默认颜色 | 含义 |
|------|-------------|---------|------|
| Source | ≤ 3 dB | 绿 | 本地量化误差是主因 |
| Mixed | 3–10 dB | 橙 | 本地和传播都显著 |
| Propagated | > 10 dB | 红 | 上游传播误差主导 |

这些阈值不得在单个图表中覆盖为不同值。如需调整阈值，应在全局配置中修改。

### R7: error_source_analysis 表格式

表头格式：
```
Error Source Analysis — {config_name} [{role}]
```

列顺序必须为：
```
Layer | Accum QSNR | Local QSNR | Delta | Headroom | Diagnosis
```

Summary 行格式：
```
Summary: drop={total_drop} avg_headroom={avg}  {n_source} source, {n_mixed} mixed, {n_propagated} propagated
```

Observer-only 和 Hook-only 条目必须单独列出在表格下方，不混入 matched 区段。
