# Review Gate

每个子任务完成、标记为 done 之前，必须派遣 review agent。

Review agent 发现的 **Critical / Major** 问题必须在当前子任务内修复，不得留到下一个子任务。

---

## Review 维度 (按优先级)

### P0 — 数学推导

量化公式的数学正确性。先推导，后验证。

- rounding 行为是否正确（nearest-even / truncation 等）
- scale 计算公式是否与 ADR / 文档一致
- overflow / underflow 行为是否符合预期
- 手工推导期望值 → `torch.equal` 比对代码输出

### P0 — 测试覆盖

- **正向路径**: 每种 granularity × format × transform 组合是否至少有一条用例
- **边界条件**: 零值、NaN、Inf、denormal、空组（tensor 小于 group）、非整除 block
- **跨粒度一致性**: 同一 format 在 PER_TENSOR → PER_CHANNEL → PER_BLOCK → BANK 四种粒度下误差是否合理递进
- **跨格式一致性**: 同一粒度下 int4/int8/fp8 退化是否符合预期
- **outlier_format 组合**: 每种 granularity mode 下 int4+int8 等组合是否有覆盖
- 不规则张量（奇数维度、质数维度、非对称形状）是否有覆盖
- 每个 `raise` 点是否都有 `pytest.raises` + `match=`

### P0 — Session API

用户通过高层 API 的完整链路是否正确：

```
QuantConfig → resolve_config() → to_op_config() → quantize_model()
→ CalibrationSession(sparse=True) → inference forward
```

- QuantConfig 参数是否正确解析为 OpQuantConfig
- outlier_ratio / outlier_format 是否正确传递给 QuantScheme
- a_outlier_format override 模式是否正确（None → 跟随 / 非 None → 覆盖）
- scale_storage (fp32 / pot) 切换是否只影响 scale、不影响数据路径
- 确定性: 相同输入 5x forward 是否得到完全相同输出

### P0 — Study API

多配置对比聚合的完整链路是否正确：

```
Study(configs=[QuantConfig...], model=fp32_model)
→ study.run(calib_data, eval_data=eval_fn)
→ StudyReport → summary() / save() / plot / reload
```

- Study.run() 对每个 config 是否正确深拷贝模型
- model_factory 参数是否正常工作
- StudyReport 输出: summary_dataframe / to_dataframe / save / from_file / plot / tables
- 结果 JSON 序列化往返一致（save → from_file → 数值不变）
- 多 config 比较时 per-layer / per-role 指标是否对齐

### P0 — E2E 回归

任何修改 transform / format / quantize / session / calibration 的 commit 必须通过：

| 脚本 | 模型 | 数据集 |
|------|------|--------|
| `scripts/mnist_hadamard_study.py` | 3-layer MLP | MNIST |
| `scripts/transformer_agnews_study.py` | 2-layer Transformer | AG News |
| `scripts/verify_batch_independence.py` | Tiny MLP (8→4→2) | Random (16 samples) |

合理性判据:
- FP32 accuracy 不得为 0（回归检测）
- int8-pc: |quant - fp32| < 0.02
- int4-pb32: |quant - fp32| < 0.05
- Hadamard/SmoothQuant 退化 ~1% 以内
- verify_batch_independence.py: 全部 6 项 PASS

> **E2E 回归模式库**: `docs/verification/e2e-regression-patterns.md` — 记录每个曾导致 E2E 回归的 bug 模式。
> 修改涉及 calibration / quantize 的代码前，对照该文档检查。

### P1 — 数值边界

- scale 极小或极大时是否产生 inf/nan
- scale_storage 精度不足时是否只影响 scale 精度、不破坏数据
- 混合精度场景（如 fp32 scale + int4 data）是否正确

### P1 — 代码架构

- 依赖方向: 工具层 → 驱动层 → 算子层 → 数学层，无反向依赖
- `src/` 不得 `import mx`
- 新功能通过注册/扩展抽象基类实现，不改核心量化函数
- 包边界: 单文件包是否属于错误分层，`utils/` / `common/` / `misc/` 是否被用作公共包名

### P1 — ADR 一致性

逐条对照对应 ADR 的每个 Decision:

- 函数签名、参数名、默认值是否与 ADR 一致
- 数据流路径是否与 ADR 描述完全一致
- ADR 明确标记为"不可用"的功能是否在代码中被正确阻塞
- ADR 中描述的限制/前提条件是否在代码中有对应校验

### P2 — Transform 正交

- 新功能在 Hadamard + SmoothQuant + 无 transform 三种场景下是否都正确
- transform.forward → quantize → transform.inverse 三步中新参数是否正确透传

### P2 — 确定性

- 同一输入多次运行是否 bit-exact 相同（无随机性、无非确定性算子）
- 跨平台/跨设备是否一致（至少同设备保证确定）

---

## 现有检查项 (保留)

| 检查项 | 说明 |
|--------|------|
| 接口合规 | 实现是否符合 `docs/architecture/` 对应 ADR 的接口规范 |
| 验证漏斗 | frozen dataclass 的每一层（构造期 `__post_init__` + 动态检查层如 `Format.quantize()`）是否都有对应测试 |
| API 陷阱 | 有无静默类型错误、缺类型验证、破坏性签名变更 |
| 边界约束 | 是否违反 `src/` ↔ `mx/` 隔离约束 |
| 可哈希性 | 作为 frozen dataclass 字段的对象是否实现 `__eq__`/`__hash__` |
| Observer 接入 | 新算子是否在量化关键点通过 `emit_fn` 回调触发事件 |
| 接口一致性 | 所有 `QuantizedXxx` 模块类的构造参数必须有 `cfg: OpQuantConfig` |
| 分析层兼容 | 若新增 `GranularityMode`，检查 `iter_slices` 是否需要同步更新 |

---

## 测试层级原则

最终提交的测试必须通过 Session / Study / QuantConfig 等高层用户接口调用。
底层 API（`FormatBase.quantize()`、`_quantize_elemwise_core` 等）仅限开发调试，不允许留在最终测试代码中。

详见 [`principles/testing-layer.md`](testing-layer.md) 和 [`standards/quantization-testing.md`](../standards/quantization-testing.md)。

---

## 派遣模板

```
对刚完成的 <子任务名> 做代码 review。
背景：<一句话描述该子任务做了什么>
检查文件：<列出修改的文件路径>
重点检查：
  - 数学推导: <具体关注点>
  - 测试覆盖: <具体关注点>
  - Session/Study API: <具体关注点>
  - E2E: <是否跑了 MNIST + Transformer>
  - ADR 一致性: <对应 ADR 编号>
参照规范：docs/architecture/<对应 ADR>
输出：每个问题带文件:行号，最后给严重程度总结表格。
```
