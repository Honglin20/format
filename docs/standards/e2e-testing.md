# E2E 测试规范 — 端到端回归门与回归模式库治理

本规范定义项目 E2E 测试的三个层次及其强制要求。所有开发者提交涉及 `calibration` / `quantize` /
`session` / `formats` / `transform` 的代码前，**必须通过本文规定的全部 E2E 门**。

---

## 一、E2E 测试三层结构

```
Layer A — 全局回归门 (scripts/*.py)
    每次 commit 必跑。验证核心量化路径无精度崩塌、无运行时崩溃。

Layer B — 特性级 E2E 测试 (src/tests/test_<feature>_e2e.py)
    新增功能必写。覆盖全部 granularity × ratio 边界 × 形状 × format 组合。

Layer C — 回归模式库 (docs/verification/e2e-regression-patterns.md)
    记录曾逃过 Layer A+B 的历史回归。修改相关代码前对照检查。
```

---

## 二、Layer A — 全局回归门（每次 commit 必跑）

### A1. 精度回归门

| 脚本 | 模型 | 目的 |
|------|------|------|
| `scripts/mnist_hadamard_study.py` | 3-layer MLP | 核心量化路径精度 |
| `scripts/transformer_agnews_study.py` | 2-layer Transformer | 核心量化路径精度 |

判据（基于 2026-05-11 baseline）：
- FP32 accuracy ≠ 0（防回归崩溃）
- int8-pc: `|quant - fp32| < 0.02`
- int4-pb32: `|quant - fp32| < 0.05`
- Hadamard / SmoothQuant 退化 ≤ 1%

### A2. 契约回归门

| 脚本 | 目的 |
|------|------|
| `scripts/verify_batch_independence.py` | Static sparse mask 不泄露 batch 维度 |
| `scripts/verify_sparse_consistency.py` | Session API 静态稀疏 bit-level 一致性 |
| `scripts/verify_mask_shapes.py` | 全模式 × 全算子 mask/scale shape 正确性（含 Conv BANK/PER_CHANNEL 轴对齐） |

判据：全部检查项 PASS（不允许部分通过）。

### A3. 全量单元测试门

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```

判据：全部 passed，0 failed。

---

## 三、Layer B — 特性级 E2E 测试（新增功能必写）

### 触发条件

以下任一情况，**必须**编写 `src/tests/test_<feature>_e2e.py`：

1. 新增 `GranularityMode` 枚举值
2. 新增 `FormatBase` 子类
3. 新增 `TransformBase` 子类
4. 新增 sparse 机制（outlier / group / 其他 split 策略）
5. 修改 `CalibrationSession` 的 sample collection 或 state 计算
6. 修改 `quantize()` 的分发逻辑
7. 修改 `OpQuantConfig` / `QuantScheme` 的字段语义

### 覆盖维度清单

| 维度 | 最小要求 |
|------|---------|
| Granularity | PER_TENSOR + PER_CHANNEL + PER_BLOCK + BANK，每种 ≥ 1 条 |
| Ratio 边界 | 0.01（最小） + 0.25（中等） + 1.0（退化全选），≥ 3 个值 |
| 形状多样性 | 1D(bias) + 2D(Linear weight) + 3D+(batch+spatial)，每种 ≥ 1 条 |
| 退化场景 | 通道数=1, num_banks=1, block_size > dim_size，各 ≥ 1 条 |
| Format 组合 | 同类型(int+int) + 跨类型(int+float/float+int) + 单 format(None)，≥ 3 种 |
| Transform 交互 | Hadamard + SmoothQuant + Identity，每种与 sparse 交互 ≥ 1 条 |
| 高层 API | Session.run() ≥ 1 条 + Study.run() ≥ 1 条 |
| Weight-only | weight_only=True ≥ 1 条 |
| Batch 安全 | 校准 batch ≠ 推理 batch，≥ 1 条 |

### 必须包含的测试用例类型

```
test_<feature>_basic              — 最简路径（默认参数）
test_<feature>_all_granularities  — 所有 granularity mode
test_<feature>_ratio_boundaries   — ratio 边界（0.01, 0.25, 1.0）
test_<feature>_shape_variety      — 1D/2D/3D+ 形状
test_<feature>_degenerate         — 退化场景
test_<feature>_format_combos      — format 组合
test_<feature>_transform_interact — transform 交互
test_<feature>_session_api        — Session API 全流程
test_<feature>_study_api          — Study API 全流程
test_<feature>_batch_mismatch     — 校准/推理 batch 不一致
```

---

## 四、Layer C — 回归模式库治理规则

回归模式库 `docs/verification/e2e-regression-patterns.md` 是**精选库**，不是垃圾场。
每个条目必须通过以下准入标准。

### 4.1 准入标准（必须同时满足）

| # | 条件 | 说明 |
|---|------|------|
| C1 | 曾导致 E2E 精度下降 **或** 运行时崩溃 | 仅 lint/风格问题不进入 |
| C2 | 逃过了 Layer A 全部现有门 | 被现有门捕获的 bug 不进入（门已生效） |
| C3 | 根因在 `src/` 核心路径中 | 测试脚本、文档、配置错误不进入 |
| C4 | 可能以相同形式再次出现 | 一次性 typo / 明显语法错误不进入 |

### 4.2 合并规则（减少冗余）

以下情况**必须合并为一条**，不得拆分为多个 §：

| 模式 | 示例 | 处理方式 |
|------|------|---------|
| 同一根因，不同 granularity mode | `torch.stack` 导致 PER_TENSOR / PER_CHANNEL / BANK 都有问题 | 合并为 1 条，预防规则覆盖所有 mode |
| 同一类 shape 错误，不同函数 | `x_calib` shape 错 → mask 错 + scale 错 | 合并为 1 条，根因是同一个 |
| 同一 API 误用模式 | 多处 `stack` 应改为 `cat` | 合并为 1 条，预防规则写成通用规范 |
| 兄弟 bug（修 A 必然引出 B） | 修复 shape 后 broadcast 逻辑也需要改 | 合并为 1 条，记录完整修复链 |

### 4.3 排除标准（不进入模式库）

以下情况**不进入**回归模式库：

| 情况 | 原因 | 去处 |
|------|------|------|
| 单次 typo（变量名拼错、漏写参数） | 不会以相同模式重现 | git log / CHANGELOG |
| 测试代码自身的 bug | 不影响生产路径 | 直接修，不记录 |
| 被现有 Layer A 门当场捕获的 bug | 门已生效，无需重复 | 直接修，不记录 |
| 纯配置问题（QuantConfig 参数值错误） | 不是代码逻辑缺陷 | 直接修，不记录 |
| 第三方依赖行为变更 | 不可控 | CHANGELOG 备注 |
| 文档/注释错误 | 不影响运行 | 直接修 |

### 4.4 条目生命周期

```
发现 bug → 修复 → 写 E2E 测试（Layer B）→ 评估准入
                                                        ├── 通过 C1-C4 → 写入 Layer C
                                                        └── 不通过 → 不进入
                                                                      └── 如果是重要教训 → CHANGELOG 备注
```

**已合并的条目不可删除**，只能追加更新。每个条目是一次"血的教训"，删除等于遗忘。

---

## 五、开发流程强制要求

### 提交前检查清单

```
□ Layer A1: mnist_hadamard_study.py 通过
□ Layer A1: transformer_agnews_study.py 通过
□ Layer A2: verify_batch_independence.py 通过
□ Layer A2: verify_sparse_consistency.py 通过
□ Layer A2: verify_mask_shapes.py 通过
□ Layer A3: pytest 全量通过（0 failed）
□ Layer B:  新增特性有对应 test_<feature>_e2e.py
□ Layer C:  对照 e2e-regression-patterns.md 全部 §，确认无违反
```

### 新功能开发

```
1. 对照 docs/verification/e2e-regression-patterns.md 全部 §
2. 写 Layer B E2E 测试（先失败）
3. 实现功能
4. 跑 Layer A + Layer B
5. 全部通过 → commit
```

### Bug 修复

```
1. 写重现测试（Layer B 格式，先失败）
2. 修复代码
3. 跑 Layer A + 新测试
4. 评估是否满足 Layer C 准入标准 → 是则写入
5. 全部通过 → commit
```

---

## 六、快速参考

```bash
# Layer A — 全局回归门（每次 commit 必跑）
PYTHONPATH=. python scripts/mnist_hadamard_study.py
PYTHONPATH=. python scripts/transformer_agnews_study.py
python scripts/verify_batch_independence.py
python scripts/verify_sparse_consistency.py
python scripts/verify_mask_shapes.py
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q

# Layer B — 特性级 E2E（按需）
pytest src/tests/test_<feature>_e2e.py -q

# Layer C — 回归模式库
cat docs/verification/e2e-regression-patterns.md
```
