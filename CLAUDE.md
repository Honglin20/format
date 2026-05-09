# CLAUDE.md — 项目入口

## 启动流程（每次必执行）

1. 读本文件
2. 读 `docs/status/CURRENT.md`（当前进行中的任务）
3. 读 CURRENT.md 中"断点续传必读文件"清单的文件（≤5 个）

**不要**在没有读 CURRENT.md 的情况下直接开始工作。

> 已完成任务的历史记录在 `docs/status/CHANGELOG.md`，只在需要排查历史缺陷时查阅。

---

## 项目目标

基于 microsoft/microxcaling（`mx/`）做增量式量化库重建，新代码在 `src/`。
核心：三轴量化方案（format + granularity + transform）+ 算子级配置 + 层级误差分析 + ONNX export + QAT。

---

## 不可违反的边界规则

1. `src/` 不得 `import mx`，不得依赖 `MxSpecs` dict
2. `mx/` 只读，仅通过公共 API 调用做等价性测试
3. 新增格式/granularity/transform 通过注册/实现抽象基类，不改量化核心函数
4. 测试先于实现：写失败测试 → 实现 → 通过 → commit → review agent
5. CURRENT.md 只记录当前进行中的任务（≤30 行）。子任务完成后立即将已完成条目移到 `docs/status/CHANGELOG.md`，不在 CURRENT.md 中积累历史
6. README 只放项目简介 + 一个 example + 文档链接。不放架构设计、API 参考表、多步骤教程。详细内容写到 `docs/` 对应模块文档

---

## 架构第一性原理

### 依赖层级（单向，上层依赖下层，永不反向）

```
工具层 (Tools)          calibration  analysis  cost  onnx
                          ↓ 使用
驱动层 (Integration)     session   模型生命周期编排 + 模块转换 + inline 拦截
                          ↓ 驱动
算子层 (Ops)             ops   QuantizedXxx autograd 算子
                          ↓ 调用
数学层 (Math)            quantize(tensor, scheme) → tensor
                          ↑ 三轴正交组合
                  format  ×  granularity  ×  transform

输出层 (Output)          report  viz  声明式输出 + 可视化（消费 SessionResult）
```

### 包边界原则

1. **单一概念**：一个包名 = 一个清晰的概念。禁止 `utils/`、`common/`、`misc/`、`shared/`、`tools/` 作为公共包——这些词没有排除标准，是反模式。
2. **独立变更驱动**：两个模块因为不同的原因而变更 → 分属不同的包。
3. **可发现性**：通过"我要做什么概念"找到包，不是"别人把文件放哪了"。
4. **单文件包是错的**：一个包只有一个文件 → 包边界画错了 → 合并到所属概念。
5. **私有用下划线**：`_utils/` 带 `_` = 内部辅助，不参与公共 API 约定。`from src._utils import X` 本身就是警告。
6. **横切关注点独立**：被两个不同层同时依赖的模块（如 observer → 被 ops 和 analysis 依赖）必须独立，不依附于任何一方。

### 量化统一入口

```python
# quantize() 是数学层唯一的公共量化入口。三步明确可见：
x_q = quantize(x, scheme)
#   1. x_t = scheme.transform.forward(x)       变换
#   2. x_q = scheme.format.quantize(x_t, ...)  量化
#   3. x_q = scheme.transform.inverse(x_q)     逆变换
```

- `QuantScheme` = format × granularity × transform — 三轴正交组合，改一个不改其他。
- `OpQuantConfig` = storage scheme + per-role compute schemes — 算子级二级模型。
- `quantize_mx()` 是 PER_BLOCK 的快速路径，对用户透明。

### 可扩展性模式

| 扩展需求 | 在哪里改 | 不动任何其他文件 |
|---------|---------|----------------|
| 新数值格式 | `formats/` 下新文件，extend `FormatBase` | ✅ |
| 新 Transform | `transform/` 下新文件，extend `TransformBase` | ✅ |
| 新粒度模式 | `scheme/granularity.py` 加 `GranularityMode` | 极少发生 |
| 新量化算子（如 NF4 matmul） | `ops/` 下新文件，复用 `quantize()` + `OpQuantConfig` | ✅ |
| 新近似计算（如 softmax approx） | `ops/` 下新文件，实现 `torch.autograd.Function` | ✅ |
| 新工具 | 工具层新目录 | 核心层不感知 |

---

## 场景 → 读什么

| 我要做什么 | 读哪个文件 |
|------------|-----------|
| 新终端续接上次工作 | `docs/status/CURRENT.md` |
| 查看已完成任务 / 历史 bug 修复 | `docs/status/CHANGELOG.md` |
| 开始一个全新功能 | `docs/workflow/feature-lifecycle.md` |
| 新增量化格式 | `docs/standards/adding-format.md` → `docs/architecture/001-*.md` |
| 新增 Observer | `docs/standards/adding-observer.md` → `docs/architecture/002-*.md` |
| 新增 Transform | `docs/standards/adding-transform.md` → `docs/architecture/001-*.md` |
| 写量化测试用例 | `docs/standards/quantization-testing.md` → `docs/verification/README.md` |
| 涉及数学正确性 | `docs/principles/math-verification.md` — 先推导，后验证 |
| 新增公共 API | `docs/standards/api-design.md` |
| 子任务做完了，准备收尾 | `docs/principles/review-gate.md` → `docs/workflow/subtask-lifecycle.md` |
| 提交代码 | `docs/workflow/branching-commits.md` |
| 排查历史缺陷 | `docs/reviews/INDEX.md` |
| 查公式定义 | `docs/reference/INDEX.md` |
| 了解整体进度 | `docs/workflow/phase-plan.md` |
| 文档读写规范 | `docs/principles/documentation-rules.md` |

---

## 快速参考

- **Branch**: `feature/refactor-src`（主开发），`claude/<desc>`（单任务），不推 master/main
- **Commit**: `<type>(<scope>): <描述>` — type: feat/fix/refactor/test/docs/chore
- **测试门（快速）**: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"`（1,857 passed）
- **全量测试**: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q`（含 slow，2,070 passed）
- **E2E 全算子**: `pytest src/tests/test_e2e_all_ops.py -q`（21 模块 + 10 inline op，49 parametrized）

---

## 完整文档索引

→ **[docs/INDEX.md](docs/INDEX.md)**
