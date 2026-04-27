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

**核心能力**（按 phase 推进，见第 6 节）：
- 可组合的三轴张量级量化方案：`format + granularity + transform`
- 可组合的算子级量化配置：`OpQuantConfig`（每个 tensor 角色接 scheme pipeline，含 forward + QAT backward）
- 层级误差分析：每个被量化算子的 QSNR / MSE / 直方图，Observer 模式，粒度感知（per-tensor / per-channel / per-block / 未来动态分组）
- ONNX export：量化算子导出为 ONNX 图（能导出、图正确；不要求 runtime 可推理）。Format.export_onnx() Strategy 模式：每个 Format 自声明 ONNX 导出行为。
- **QAT（训练感知量化）**：`OpQuantConfig` 的 `grad_*` 字段填充即启用 QAT backward；为空则退化为 STE（inference-only 语义）。
- **Unified quantize_model**：一键量化入口（Module 替换 + forward patching 自动注入 QuantizeContext）

**下一阶段（Phase 8 — 研究能力扩展）**：
- P1: Transform 体系（SmoothQuant / Bias correction / CLE / Hadamard rotation）
- P2: Calibration 管线（可插拔 scale 策略 + 校准数据集 + scale 持久化）
- 完整优先级清单见 auto memory: `format-research-roadmap.md`

**不在当前范围内**：
- ORT / TensorRT runtime 推理适配
- 模型压缩（剪枝/蒸馏）
- RNN 家族算子（`mx/rnn.py` 不在 `src/ops/` 重建）
- FlashAttention 量化

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
│   ├── ops/             # 量化算子（Linear/Conv/Norm/Softmax/Pool/Elemwise，Phase 3；RNN 不做）
│   ├── analysis/        # 层级误差分析，Observer 模式（Phase 3 起落骨架，Phase 4 落 AnalysisContext）
│   ├── mapping/         # 模型级量化算子替换入口（quantize_model unified）
│   ├── onnx/            # ONNX export（Phase 5）
│   └── context/         # QuantizeContext 上下文管理器（Phase 6）
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

### 3.2 算子级配置（OpQuantConfig，ADR-005）

- 量化算子的**完整配置**用 `OpQuantConfig`（见 `docs/architecture/005-op-quant-config.md`）
- 每个 tensor 角色是一个 `tuple[QuantScheme, ...]` pipeline（0/1/N 个 scheme）
  - forward 角色：`input` / `weight` / `bias` / `output`
  - backward 角色：`grad_output` / `grad_input` / `grad_weight` / `grad_bias`
  - backward 内部 gemm 复量化：`input_gw` / `grad_output_gw` / `weight_gi` / `grad_output_gi`
- 非 matmul 算子只填 `input` / `output`（+ `grad_input` / `grad_output` 可选），其余字段保持 `()`
- `cfg.is_training` 自动识别是否启用 QAT（任一 backward 字段非空）
- 算子 forward/backward 实现是**"for scheme in cfg.<role>: x = quantize(x, scheme)"**的机械消费，不含分支逻辑

**inner_scheme 模式（Activation/Softmax/Pool）**：这三类算子的每个量化步骤都使用同一个 scheme。  
模块类对外接口统一为 `cfg: OpQuantConfig`，内部从 `cfg.input[0]` 提取 `inner_scheme` 传入 Function。  
支持向下兼容参数 `inner_scheme: QuantScheme = None`，自动转换为 `OpQuantConfig(input=(inner_scheme,), grad_input=(inner_scheme,))`。  
详见 `docs/plans/2026-04-25-defect-fix-specs.md` M1 章节。

### 3.3 Observer 分析模式（ADR-002）

- 量化算子在关键点发事件（`ObservableMixin._emit(role, pipeline_index, stage, fp32, quant, scheme, group_map=None)`），不做 analysis 计算
- 事件 stage 枚举：`input_pre_quant` / `weight_pre_quant` / `output_post_quant` / `grad_output_pre_quant` / `grad_weight_post_quant` / `grad_input_post_quant`
- 粒度感知切分：Observer 通过 `src/analysis/slicing.py::iter_slices(fp32, quant, granularity, group_map)` 统一入口获取 per-tensor / per-channel / per-block / 未来动态分组切片；**`iter_slices` PER_BLOCK 模式沿 `GranularitySpec.block_axis` 切片，不写死 last dim**
- `SliceAwareObserver` 抽象：子类只需实现 `_measure(key, fp32_slice, quant_slice) -> metric_dict`，自动按 granularity 循环聚合
- 外部 Observer 通过上下文管理器挂载：`with AnalysisContext(model, observers=[...]) as ctx: ...`（Phase 4 落地）
- 量化代码和 analysis 代码严格解耦，关闭 analysis 时 `_emit` 直接 early return，零开销

**`emit_fn` 回调模式**：`_emit()` 只能在持有 `self` 的 `QuantizedXxx.forward()` 中调用。  
`XxxFunction.forward()` 末尾参数接收 `emit_fn=None`（可选回调）；`QuantizedXxx.forward()` 传入 `self._emit if self._observers else None`。  
Function 内每个量化关键点用 `if emit_fn: emit_fn(role, idx, stage, fp32, quant, scheme)` 触发事件。  
详见 `docs/plans/2026-04-25-defect-fix-specs.md` C1 章节。

### 3.4 ONNX Export

- 每个量化 `autograd.Function` 提供 `symbolic()` 方法
- `_emit_quantize_node(g, x, scheme)` 委托给 `scheme.format.export_onnx(g, x, scheme)` — Format 自声明 ONNX 行为（Strategy 模式，与 `quantize()` 对称）
- `IntFormat.export_onnx()` → 标准 QDQ（`QuantizeLinear`/`DequantizeLinear`），非 PER_BLOCK
- `FPFormat.export_onnx()` → QDQ for `fp8_e4m3`/`fp8_e5m2` + non-PER_BLOCK；其余走 `MxQuantize`
- `FormatBase.export_onnx()` 默认 → `com.microxscaling::MxQuantize`（自定义 domain）
- **JIT tracing guard**：`FormatBase._quantize_per_block()` 在 `torch.jit.is_tracing()` 时直接 `return x`，跳过 `_reshape_to_blocks`。symbolic() 负责生成真实的 ONNX 量化节点。两阶段分离：forward() for shape inference，symbolic() for ONNX graph
- 目标：`onnx.checker` 通过，`netron` 能可视化；不要求 ORT 可执行推理

---

## 4. 开发工作流

### 分支
- 主开发分支：`feature/refactor-src`（所有 src/ 重建工作的 long-lived branch）
- 单次 review / 多任务聚合分支：`claude/<short-desc>`，review 完成后 fast-forward 合入 `feature/refactor-src`
- 不得推送到 `master` 或 `main`

### Commit 规范
```
<type>(<scope>): <简短描述>

[可选正文：说明 why，不说 what]
```
type: `feat` / `fix` / `refactor` / `test` / `docs` / `chore`
scope: `scheme` / `formats` / `quantize` / `ops` / `analysis` / `mapping` / `onnx` / `docs`

示例：
```
feat(scheme): add TransformBase + IdentityTransform to QuantScheme
refactor(quantize): replace MxSpecs dispatch with Format.quantize() Strategy
```

### 子任务生命周期（必须严格遵守）

每个子任务从开始到结束遵循固定节奏，**不得跳过任何步骤**：

1. **实现** → TDD（先写失败测试 → 实现 → 测试通过）
2. **Review** → 派遣 review agent，修复 Critical/Major 问题
3. **Commit** → 小步提交，测试 + 实现在同一 commit
4. **总结与状态更新** → 立即更新 `docs/status/CURRENT.md`：打勾已完成子任务、更新下一步、更新断点续传必读文件、补充关键经验记录。**不积累，不延后**
5. **子任务结束信号** → 完成上述步骤后，Claude 应向用户说明：**"子任务 X 已完成并提交。下一步是 Y。请按需 `/clear` 或开始新对话，以保持 context 干净。"**
   - 用户决定何时执行 `/clear`（Claude Code 的 `/clear` 命令清除对话历史）
   - Claude 不应在同一 context 内连续推进多个子任务，除非用户明确要求
   - CURRENT.md 是唯一持久化状态，context 清空后依靠 CURRENT.md 断点续传

> **说明：context 控制由用户主导**。Claude Code 的 context 压缩是系统自动行为，Claude 无法主动"清理"已在 context 中的内容。"Clear Context"的实操含义是：子任务结束 → 更新 CURRENT.md → 提示用户 → 等待用户 `/clear` 或开新会话。不要在同一会话中"假装"context 已清空而继续工作。

### 测试门（每个 task 完成前必须通过）

| Phase | 必须通过的测试 | 门槛 |
|---|---|---|
| Phase 2 修正 | `pytest src/tests/ -x`（等价性全绿，无 regression）| bit-exact |
| Phase 3 算子 | 各子阶段 `pytest src/tests/test_ops_equiv_<family>.py -x` + 总门 `pytest src/tests/ -x` | **bit-exact**（`torch.equal`，dither 固定 seed，**不允许 atol/rtol**） |
| Phase 4 分析 | 所有已有测试 + `pytest src/tests/test_analysis.py` | 数值稳定（atol 文档明示） |
| Phase 5 ONNX | 所有已有测试 + `pytest src/tests/test_onnx_export.py` | 图结构正确 + `onnx.checker` 通过 |
| Phase 6+ 集成 | `pytest src/tests/ -q` 全量（当前 1068 tests） | 0 xfail，无 regression |

---

## 5. 开发准则

### 5.1 TDD 原则（测试驱动开发）

- **测试先于或同步于实现**：每个子任务的测试与实现代码在同一个 commit 中，绝不允许"先实现后补测试"
- **子任务节奏**：写失败测试 → 实现 → 测试通过 → commit → 派遣 review agent
- **等价性测试也遵循 TDD**：从 `mx/` 迁移的代码，先写 `assert torch.equal(mx_output, src_output)`，再实现 src 函数。**Phase 3 bit-exact，不允许 `allclose` 宽松匹配**
- **测试命名要表达行为**：`test_per_channel_rejects_string_axis` 而不是 `test_per_channel_error`
- **负面测试覆盖所有 raise 点**（本条在 2026-04-24 细化）：每一个 `__post_init__` 里的 `raise`、每一个工厂方法的类型/值守卫、每一个公共 API 的错误分支，**必须各自至少一条** `pytest.raises` 测试并断言 `match=` 关键字子串。新增任何 `raise ...` 必须伴随新增至少一条负面测试 —— 在 commit 里同步提交，不得拆到下一个 subtask

### 5.2 Review Agent 门（每个子任务完成后强制执行）

每个子任务完成、标记为 done 之前，**必须派遣 review agent** 检查以下清单：

| 检查项 | 说明 |
|---|---|
| 接口合规 | 实现是否符合 `docs/architecture/` 对应 ADR 的接口规范 |
| 测试覆盖 | 正向路径、错误路径、边界值是否均有测试 |
| 验证漏斗 | frozen dataclass 的每一层（构造期 `__post_init__` + 动态检查层如 `Format.quantize()`）是否都有对应测试，任何"静默通过 + 延迟崩溃"都是缺陷 |
| API 陷阱 | 有无静默类型错误、缺类型验证、破坏性签名变更 |
| 边界约束 | 是否违反 Section 2 的 `src/` ↔ `mx/` 隔离约束 |
| 可哈希性 | 作为 frozen dataclass 字段的对象是否实现 `__eq__`/`__hash__` |
| Observer 接入（Phase 3+ 算子）| 新算子是否在量化关键点通过 `emit_fn` 回调触发事件；`emit_fn` 由 `QuantizedXxx.forward()` 传入 Function，不能在 Function 静态方法内直接调用 `self._emit`（无 self）；`emit_fn = self._emit if self._observers else None` 保证零开销 |
| 接口一致性（算子族）| 所有 `QuantizedXxx` 模块类的构造参数必须有 `cfg: OpQuantConfig`；Activation/Softmax/Pool 可额外支持 `inner_scheme` 向下兼容参数，但须自动转换为 `cfg`，不得以 `inner_scheme` 作为唯一公共接口 |
| 分析层兼容（iter_slices）| 若新增 `GranularityMode` 或修改现有模式的 axis 语义，检查 `iter_slices` 是否需要同步更新；PER_BLOCK 必须沿 `granularity.block_axis` 切片，不得写死 last dim |

Review agent 发现的 **Critical / Major** 问题必须在当前子任务内修复，不得留到下一个子任务。

**派遣 review agent 的提示模板**：
```
对刚完成的 <子任务名> 做代码 review。
背景：<一句话描述该子任务做了什么>
检查文件：<列出修改的文件路径>
重点检查：<针对该子任务的具体风险点>
参照规范：docs/architecture/<对应 ADR>
输出：每个问题带文件:行号，最后给严重程度总结表格。
```

### 5.3 多 Agent 开发

根据任务类型选择合适的 agent，保持主 context 干净：

| 场景 | 使用 agent 类型 | 原因 |
|---|---|---|
| 探索代码库（找文件、理解现有实现） | `Explore` agent | 大量文件读取不污染主 context |
| 架构决策（新格式如何集成、权衡分析） | `Plan` agent | 系统性推理，产出结构化计划 |
| 子任务完成后的代码审查 | `general-purpose` agent | 深度多维度检查（见 5.2） |
| 研究外部 API（PyTorch 内部机制、ONNX spec） | `general-purpose` agent | Web 搜索 + 代码结合 |
| 独立的多文件实现 | 并行 agent | 无依赖的子任务并行派遣 |

**Context 卫生原则**：
- grep / diff 的长列表输出通过 agent 汇总为结论后返回主 context
- 主 context 只保留"当前子任务直接需要读"的文件内容
- 不要在主 context 里连续读取 5 个以上文件（用 Explore agent 代替）
- 每完成一个子任务就触发"子任务结束信号"（见 Section 4 生命周期步骤 5），不在同一 context 内连续推进多个子任务

### 5.4 API 设计约束

这些规则来自实战 review 中反复出现的问题，**每次新增公共 API 时必须对照检查**：

**可哈希抽象基类**：若 ABC 的实例会用作 frozen dataclass 的字段（如 `TransformBase` 用于 `QuantScheme`），必须在 ABC 中将 `__eq__` 和 `__hash__` 声明为 `@abstractmethod`，强制子类实现，防止 id-based hash 静默破坏值相等性。

**`__post_init__` 验证完整性（2026-04-24 加强）**：frozen dataclass 的 `__post_init__` 必须对**全部字段**做类型验证。新增字段时，同步更新 `__post_init__` 的字段校验白名单，并在 review agent 清单里显式对照字段逐项确认——漏掉一个字段就会在运行时产生难以定位的 AttributeError（典型案例：P2F-7 的 `granularity` 字段漏验证）。

**签名变更稳定性**：向已有工厂方法中间插入新位置参数时，必须用关键字专用参数（`def f(a, *, new_param=0, old_kw=...)`）或在函数开头对旧类型做守卫（`if isinstance(new_param, str): raise TypeError(...)`），防止旧调用方式静默变成错误语义。

**无静默默认值**：构造函数默认值不得隐藏重要语义（如 `format="int8"` 的隐式默认、`granularity=per_tensor` 的隐式默认）。有非显然默认值时，**docstring 必须在类级和字段级各写一次**"默认行为是 X"——类级说明便于概览，字段级注解便于 IDE tooltip。

**维度索引的负值**：所有 `axis` / `channel_axis` 参数必须在文档和验证中明确声明是否支持负值（PyTorch 风格的 -1 = last dim）。不支持则加 `axis >= 0` 校验。支持则需分两层保证：（1）文档层说明支持；（2）运行时层在可验证位置（持有张量形状的位置，如 `Format.quantize()`）做越界断言，**不能假设"只要用户知道就不会传错"**。

**跨对象一致性验证（2026-04-24 新增）**：某些约束涉及多个对象（如 `GranularitySpec.channel_axis` 是否有效依赖于 tensor shape），无法在单个对象 `__post_init__` 中验证。这类"延迟验证"必须：（1）文档说明"此字段的越界 / 一致性检查在 `<具体函数>` 中动态做"；（2）在该函数中有显式断言并带清晰错误信息；（3）配一条 `pytest.raises` 测试覆盖该动态路径。不允许"静默假设在另一处会检查"。

**Format ONNX 导出 Strategy 模式（2026-04-27 新增）**：新增 Format 子类时，如需不同于默认 `MxQuantize` 的 ONNX 行为，须覆写 `export_onnx(self, g, x, scheme)`。`_emit_quantize_node()` 已简化为一行委托，**不要在 helpers.py 中再添加硬编码 format name 分派**。

**JIT tracing 与量化路径分离（2026-04-27 新增）**：PyTorch old-style ONNX exporter 分两阶段：JIT tracing（调 `forward()` 做 shape 推断）→ ONNX 建图（调 `symbolic()` 生成节点）。`_quantize_per_block()` 已加 `torch.jit.is_tracing()` guard 跳过 `_reshape_to_blocks`。**新增量化路径时，若涉及 JIT-unfriendly 操作（Tensor.item()、动态 shape 分支），必须在入口加同样的 guard**。symbolic() 负责在 ONNX 图中表达量化语义，forward() 只负责 shape 推断。

---

## 6. 开发 Phase 计划（按顺序推进）

### Phase 2 修正（已完成 ✅）— 扶正三轴架构

子任务：
- [x] P2F-1 ~ P2F-7：GranularitySpec、TransformBase、FormatBase.quantize()、消除 MxSpecs、缺陷收口

### Phase 3（已完成 ✅）— 算子层（src/ops/）

- [x] P3.0 ~ P3.6：全算子族（Linear/Conv/Norm/Activation/Softmax/Pool/Elemwise/SIMD）+ quantize_model

### Phase 4（已完成 ✅）— 层级误差分析（src/analysis/）

- [x] AnalysisContext + QSNR/MSE/Histogram/Distribution Observer + 报告聚合

### Phase 5（已完成 ✅）— ONNX Export（src/onnx/）

- [x] 全算子 Function.symbolic() + QDQ + MxQuantize + Format.export_onnx() Strategy

### Phase 6（已完成 ✅）— QuantizeContext（src/context/）

- [x] torch/F 命名空间 patch + module-stack hooks + ctx.export_onnx()

### Phase 7（已完成 ✅）— Unified quantize_model

- [x] Module 替换 + forward patching + model.export_onnx() 便捷方法

### Phase 8（当前 — 研究能力扩展）

**P1: Transform 体系** — SmoothQuant、Bias correction、Cross-layer equalization、Hadamard rotation
**P2: Calibration 管线** — 可插拔 scale 策略（max/percentile/MSE/KL）+ 校准数据集 + scale 持久化

完整优先级清单见 auto memory: `format-research-roadmap.md`

---

## 7. TASK 协议（断点续传规范）

这是避免上下文爆炸、支持新终端快速续接的核心机制。

### 6.1 启动新 task 时

1. 在 `docs/plans/YYYY-MM-DD-<taskname>.md` 创建实现计划（≤200 行）：
   - 目标和验收标准
   - 涉及文件清单（带路径）
   - 子任务 checklist（有序）
2. 更新 `docs/status/CURRENT.md`（见 6.3 格式）

### 6.2 完成每个子任务后（立即更新，不批量）

严格遵循 Section 4 "子任务生命周期"的步骤 4-5：

**步骤 4 — 总结与状态更新**：在 `docs/status/CURRENT.md` 中：
- 打勾已完成子任务
- 更新"下一步"为具体的下一个子任务
- 更新"断点续传必读文件"（只列真正需要读的文件，≤5 个，带行号范围）
- 补充"关键经验记录"（跨任务复用的发现，如 _bp key 行为、axis 约定等）

每完成一个子任务就 commit（小步提交），不要积累大改动。

**步骤 5 — Clear Context**：状态更新完成后、进入下一个子任务之前：
- 将已发现的可复用知识写入 memory 文件（`~/.claude/projects/.../memory/`）
- 不在 context 中保留已完成子任务的中间文件内容
- 下一个子任务通过"断点续传必读文件"按需加载，而非依赖 context 残留

### 6.3 CURRENT.md 固定格式

```markdown
# Current Task

**Task ID**: <Phase>-<编号>（如 P2F-3、P3.1-d）
**Plan**: docs/plans/YYYY-MM-DD-<name>.md
**Branch**: feature/refactor-src

## Progress

- [x] 子任务 1（已完成）
- [x] 子任务 2（已完成）
- [ ] **子任务 3（进行中）**
- [ ] 子任务 4

## 待讨论设计决策（如有）

> 用于记录需要与用户确认再继续的设计问题。有未决策项时不得推进依赖它的子任务。

- [ ] 决策 A：<描述选项 A vs B，以及影响范围>

## 下一步（具体动作）

<一句话，精确到函数/文件/行号级别>

## 断点续传必读文件

1. `src/scheme/quant_scheme.py`（全文）
2. `src/formats/base.py`（全文）
3. `src/quantize/elemwise.py`（1-120 行）
4. `docs/plans/2026-04-24-phase2-fix.md`（全文）

## 关键经验记录

<跨任务复用的发现，每条一句话>
```

### 6.4 新终端续接流程（Claude 自动执行）

```
1. 读 CLAUDE.md（本文件）
2. 读 docs/status/CURRENT.md
3. 只读 CURRENT.md 里"断点续传必读文件"清单中的文件
4. 若 CURRENT.md 有"待讨论设计决策"，先与用户确认后再继续
5. 确认当前 task 和下一步后，继续工作
```

### 6.5 Context 管理实操（2026-04-25 补充）

Claude Code 的 context 是对话历史，系统会自动压缩，但**Claude 无法主动清除**。以下是可操作的 context 管理策略：

| 操作 | 执行者 | 时机 |
|---|---|---|
| 子任务完成信号 | Claude（文字说明） | 每个子任务 commit 后 |
| `/clear` 清空 context | **用户** | 在 Claude 发出信号后，用户决定是否清空 |
| 新对话开始（最彻底） | **用户** | 跨大子任务时推荐 |
| 在 context 内使用 Explore/general agent | Claude | 需要大量文件读取时，避免污染主 context |

**关键原则**：每次新对话/清空 context 后，Claude 必须从 CURRENT.md 的"断点续传必读文件"重新加载状态，不依赖 context 历史。**CURRENT.md 是唯一可信的持久化状态。**

---

## 8. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/architecture/001-three-axis-quant-scheme.md` | QuantScheme 三轴设计、Format/Granularity/Transform 接口规范 |
| `docs/architecture/002-observer-analysis.md` | 层级误差 analysis 的 Observer 模式 + SliceAwareObserver + iter_slices |
| `docs/architecture/003-onnx-export.md` | ONNX export 策略（混合 QDQ + 自定义 domain） |
| `docs/architecture/004-mxspecs-migration.md` | MxSpecs → QuantScheme 渐进式迁移计划 |
| `docs/architecture/005-op-quant-config.md` | OpQuantConfig：算子级 scheme pipeline 容器（forward + QAT backward）|
| `docs/plans/2026-04-24-phase2-fix.md` | Phase 2 三轴扶正实现计划（P2F-1 ~ P2F-6）|
| `docs/plans/2026-04-24-p2f7-findings.md` | P2F-7 Phase 2 review 后缺陷收口清单 |
| `docs/plans/2026-04-24-phase3.md` | Phase 3 算子层实现计划（P3.0 ~ P3.6，RNN 不做）|
| `docs/plans/2026-04-25-phase3-review.md` | Phase 3 代码 Review 报告（C1/M1-M5/m1-m6 缺陷清单，修复优先级）|
| `docs/plans/2026-04-25-defect-fix-specs.md` | C1/M1/M2 三项缺陷的详细修复规范（emit_fn 回调模式、OpQuantConfig 接口统一、block_axis 切片修复）|
| `docs/status/CURRENT.md` | 活跃 task 断点（新终端必读） |
| `docs/plans/` | 各 task 详细实现计划 |
| `~/.claude/projects/.../memory/format-research-roadmap.md` | Phase 8 研究能力缺口与优先级（auto memory） |
