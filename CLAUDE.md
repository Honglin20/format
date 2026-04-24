# CLAUDE.md — 项目开发规范

## 重要提示（每次启动必执行）

新终端启动时，按以下顺序读取，不要跳过：
1. 本文件（CLAUDE.md）— 项目总规范
2. `docs/status/CURRENT.md` — 当前活跃 task 的断点状态（<100 行）
3. CURRENT.md 中"断点续传必读文件"清单列出的具体文件（≤5 个）

**不要**在没有读 CURRENT.md 的情况下直接开始工作。

---

## 1. 项目目标

基于 microsoft/microxcaling（`mx/` 目录）做**增量式**、**高扩展性**的量化库重建，新代码全部在 `src/` 中。

**核心能力**（按 phase 推进，见第 5 节）：
- 可组合的三轴量化方案：`format + granularity + transform`
- 层级误差分析：每个被量化算子的 QSNR / MSE（Observer 模式）
- ONNX export：量化算子导出为 ONNX 图（能导出、图正确；不要求 runtime 可推理）

**不在当前范围内**：
- ORT / TensorRT runtime 推理适配
- 训练感知量化（QAT）
- 模型压缩（剪枝/蒸馏）

---

## 2. 仓库结构与边界

```
format/
├── mx/                  # Legacy 只读。作为等价性测试的黄金 reference。
│                        # 绝不修改。最终在 src/ 功能完整后删除。
├── src/                 # 所有新代码。不得 import mx 中的任何符号。
│   ├── formats/         # Format 抽象层（Strategy 模式）
│   ├── scheme/          # QuantScheme 三元组（format+granularity+transform）
│   ├── quantize/        # 量化核心函数（QuantScheme 驱动，不依赖 MxSpecs）
│   ├── ops/             # 量化算子（Linear/Conv2d/LSTM 等，Phase 3）
│   ├── analysis/        # 层级误差分析，Observer 模式（Phase 4）
│   └── onnx/            # ONNX export（Phase 5）
├── docs/
│   ├── architecture/    # 架构决策文档（ADR），命名: NNN-title.md
│   ├── plans/           # 每个 task 的实现计划，命名: YYYY-MM-DD-taskname.md
│   └── status/
│       └── CURRENT.md   # 活跃 task 断点（新终端必读）
└── CLAUDE.md            # 本文件
```

**边界约束（不可违反）**：
- `src/` 中任何文件不得 `import mx` 或依赖 `MxSpecs` dict
- `mx/` 只读，任何等价性测试通过 `mx.` 公共 API 调用，不改动 mx 源码
- 新格式、新 granularity、新 transform 通过**注册/实现抽象基类**添加，不改动量化核心函数

---

## 3. 核心架构约束

详细设计见 `docs/architecture/`。这里列**不可违反的约束**：

### 3.1 三轴量化方案（QuantScheme）

```python
@dataclass(frozen=True)
class QuantScheme:
    format: FormatBase          # 格式对象，Strategy 模式
    granularity: GranularitySpec  # 粒度规格（含 block_size / channel_axis）
    transform: TransformBase = IdentityTransform()  # 可选前后变换
    round_mode: str = "nearest"
```

- `format` 必须是 `FormatBase` 子类的实例，不是字符串
- 量化函数签名：`quantize(x: Tensor, scheme: QuantScheme) -> Tensor`
- 量化流程固定为：`transform.forward(x) → format.quantize(x, granularity, round_mode) → transform.inverse(x_q)`
- 加新格式 = 实现 `FormatBase` 子类 + 注册。不改量化核心函数。

### 3.2 Observer 分析模式

- 量化算子在关键点发事件，不做 analysis 计算
- 事件点：`input_pre_quant`、`weight_pre_quant`、`output_post_quant`
- 外部 Observer 通过上下文管理器挂载：`with AnalysisContext(model) as ctx: ...`
- 量化代码和 analysis 代码严格解耦，关闭 analysis 时零开销

### 3.3 ONNX Export

- 每个量化 `autograd.Function` 提供 `symbolic()` 方法
- 标准格式（int8/fp8）导出为 ONNX 标准 `QuantizeLinear`/`DequantizeLinear`
- MX/NVFP4 等非标准格式导出为自定义 domain（`com.microxscaling.*`）
- 目标：`onnxruntime` 能加载图，`netron` 能可视化；不要求 ORT 可执行推理

---

## 4. 开发工作流

### 分支
- 工作分支：`claude/pull-code-changes-U0kal`
- 不得推送到 `master` 或 `main`

### Commit 规范
```
<type>(<scope>): <简短描述>

[可选正文：说明 why，不说 what]
```
type: `feat` / `fix` / `refactor` / `test` / `docs` / `chore`
scope: `scheme` / `formats` / `quantize` / `ops` / `analysis` / `onnx` / `docs`

示例：
```
feat(scheme): add TransformBase + IdentityTransform to QuantScheme
refactor(quantize): replace MxSpecs dispatch with Format.quantize() Strategy
```

### 测试门（每个 task 完成前必须通过）

| Phase | 必须通过的测试 |
|---|---|
| Phase 2 修正 | `pytest src/tests/ -x`（等价性测试全绿，无 regression） |
| Phase 3 算子 | 等价性测试 + `pytest src/tests/test_ops_equiv.py` |
| Phase 4 分析 | 所有已有测试 + `pytest src/tests/test_analysis.py` |
| Phase 5 ONNX | 所有已有测试 + `pytest src/tests/test_onnx_export.py` |

---

## 5. 开发 Phase 计划

### Phase 2 修正（当前）— 扶正三轴架构
**目标**：让 src/ 的 QuantScheme 真正成为三轴可组合配置，量化函数改为 QuantScheme/Format 驱动。

子任务（按顺序）：
- [ ] P2F-1：设计并实现 `GranularitySpec`（含 block_size、channel_axis），替换当前 Granularity enum
- [ ] P2F-2：设计并实现 `TransformBase` + `IdentityTransform`；升级 `QuantScheme` 加 transform 字段
- [ ] P2F-3：升级 `FormatBase`，加 `quantize(x, granularity, round_mode) -> Tensor` 抽象方法
- [ ] P2F-4：实现各 Format 子类的 `quantize()` 方法（FP/INT/BF16 等），替换 elemwise.py 中的 if-elif 链
- [ ] P2F-5：升级 `quantize_mx_op` 使用 `QuantScheme`，消除对 MxSpecs 的依赖
- [ ] P2F-6：更新所有等价性测试，改用新 QuantScheme API；确保全部通过

计划文档：`docs/plans/2026-04-24-phase2-fix.md`（待创建）

### Phase 3 — 算子层（src/ops/）
完全基于 QuantScheme 实现 `QuantizedLinear`、`QuantizedConv2d`、`QuantizedLSTM`。
等价性对标 `mx/linear.py`、`mx/conv.py`、`mx/lstm.py`。

### Phase 4 — 层级误差分析（src/analysis/）
Observer 模式。`AnalysisContext` 上下文管理器，支持 QSNR / MSE 指标收集。

### Phase 5 — ONNX Export（src/onnx/）
`autograd.Function.symbolic()` + 混合导出策略。

---

## 6. TASK 协议（断点续传规范）

这是避免上下文爆炸、支持新终端快速续接的核心机制。

### 6.1 启动新 task 时

1. 在 `docs/plans/YYYY-MM-DD-<taskname>.md` 创建实现计划（≤200 行）：
   - 目标和验收标准
   - 涉及文件清单（带路径）
   - 子任务 checklist（有序）
2. 更新 `docs/status/CURRENT.md`（见 6.3 格式）

### 6.2 完成每个子任务后（立即更新，不批量）

在 `docs/status/CURRENT.md` 中：
- 打勾已完成子任务
- 更新"下一步"为具体的下一个子任务
- 更新"断点续传必读文件"（只列真正需要读的文件，≤5 个，带行号范围）

每完成一个子任务就 commit（小步提交），不要积累大改动。

### 6.3 CURRENT.md 固定格式

```markdown
# Current Task

**Task ID**: <Phase>-<编号>（如 P2F-3）
**Plan**: docs/plans/YYYY-MM-DD-<name>.md
**Branch**: claude/pull-code-changes-U0kal

## Progress

- [x] 子任务 1（已完成）
- [x] 子任务 2（已完成）
- [ ] **子任务 3（进行中）**
- [ ] 子任务 4

## 下一步（具体动作）

<一句话，精确到函数/文件/行号级别>

## 断点续传必读文件

1. `src/scheme/quant_scheme.py`（全文）
2. `src/formats/base.py`（全文）
3. `src/quantize/elemwise.py`（1-120 行）
4. `docs/plans/2026-04-24-phase2-fix.md`（全文）
```

### 6.4 新终端续接流程（Claude 自动执行）

```
1. 读 CLAUDE.md（本文件）
2. 读 docs/status/CURRENT.md
3. 只读 CURRENT.md 里"断点续传必读文件"清单中的文件
4. 确认当前 task 和下一步后，继续工作
```

---

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/architecture/001-three-axis-quant-scheme.md` | QuantScheme 三轴设计、Format/Granularity/Transform 接口规范 |
| `docs/architecture/002-observer-analysis.md` | 层级误差 analysis 的 Observer 模式设计 |
| `docs/architecture/003-onnx-export.md` | ONNX export 策略（混合 QDQ + 自定义 domain） |
| `docs/architecture/004-mxspecs-migration.md` | MxSpecs → QuantScheme 渐进式迁移计划 |
| `docs/status/CURRENT.md` | 活跃 task 断点（新终端必读） |
| `docs/plans/` | 各 task 详细实现计划 |
