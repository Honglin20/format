# Granularity × Sparse 分析文档设计

**日期**: 2026-05-16
**状态**: 实施中

## 目标

1. README 中锚定 ADR-012/013 的 sparse 专属名：Element Sparse / Group Sparse
2. 生成一份可复现的 HTML 分析文档，以 int4 为基础格式，可视化展示 4 种粒度 × 2 种 sparse 模式

## 产出物

| 文件 | 说明 |
|------|------|
| `README.md` | 新增 Element Sparse / Group Sparse 命名段落 + QuantConfig 配置示例 |
| `scripts/granularity_sparse_analysis.py` | 生成脚本，调用库 API 跑真实量化，输出 HTML |
| `docs/guides/example/granularity-sparse-analysis.html` | 最终分析文档（单文件，内联 CSS） |

## Tensor 定义

| 名称 | Shape | 角色 | 数据来源 |
|------|-------|------|----------|
| `x1` | (1, 8, 16) | matmul input | `torch.randn` + 5-6 个 outlier (8~15)，seed=42 |
| `W` | (4, 16) | weight | `torch.randn` + 3-4 个 outlier (8~12)，seed=43 |

输出：`y = x1 @ W.T` → shape (1, 8, 4)

## 文档章节

### 0. 术语
- Element Sparse = ADR-012，per-element 离群点隔离
- Group Sparse = ADR-013，per-group 格式分配
- 核心区别：mask 形状、选择单位、组内一致性

### 1. Tensor 定义与原始值
- 3 个 tensor 的 HTML 着色网格展示
- 标注 outlier 位置

### 2. 基础粒度（4 种）
对 per_tensor / per_channel / per_block(size=8) / bank(size=4)：
- 着色网格：group 边界用背景色区分，scale 标在头部
- 量化后数值网格
- QSNR：tensor 级 + matmul 输出级
- Scale 开销：scale buffer 数量

### 3. Element Sparse
对每种粒度 + outlier_ratio={0.05, 0.1, 0.2, 0.3}：
- ratio=0.1 的着色网格：outlier 红色高亮，分裂 scale (amax_o, amax_n)
- 量化后数值 vs 无 sparse 对比
- QSNR 对比表：无 sparse vs element sparse
- ratio 扫描表：QSNR vs ratio

### 4. Group Sparse
对每种粒度 + group_ratio={0.1, 0.3, 0.5, 0.7}：
- ratio=0.3 的着色网格：H group 绿色，L group 保持原色
- 量化后数值 vs 无 sparse 对比
- QSNR 对比表：无 sparse vs group sparse
- ratio 扫描表：QSNR vs ratio

### 5. 误差热点图
关键配置的 per-element |x - x_q| 热力图

### 6. 汇总矩阵
全粒度 × 全模式 QSNR + Scale 开销表

### 7. Element Sparse vs Group Sparse 对比决策表
适用场景、优劣、推荐

## 技术约束

- 所有量化操作用 `src.quantize.quantize(x, scheme=scheme)`
- QSNR 计算：`10 * log10(signal_power / noise_power)`
- HTML 单文件，内联 CSS，无外部依赖
- 固定随机种子，可复现
