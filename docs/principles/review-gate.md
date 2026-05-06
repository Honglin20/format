# Review Agent 门

## 规则

**每个子任务完成、标记为 done 之前，必须派遣 review agent。**

Review agent 发现的 **Critical / Major** 问题必须在当前子任务内修复，不得留到下一个子任务。

## 检查清单

| 检查项 | 说明 |
|--------|------|
| 接口合规 | 实现是否符合 `docs/architecture/` 对应 ADR 的接口规范 |
| 测试覆盖 | 正向路径、错误路径、边界值是否均有测试 |
| 验证漏斗 | frozen dataclass 的每一层（构造期 `__post_init__` + 动态检查层如 `Format.quantize()`）是否都有对应测试 |
| API 陷阱 | 有无静默类型错误、缺类型验证、破坏性签名变更 |
| 边界约束 | 是否违反 `src/` ↔ `mx/` 隔离约束 |
| 可哈希性 | 作为 frozen dataclass 字段的对象是否实现 `__eq__`/`__hash__` |
| Observer 接入 | 新算子是否在量化关键点通过 `emit_fn` 回调触发事件 |
| 接口一致性 | 所有 `QuantizedXxx` 模块类的构造参数必须有 `cfg: OpQuantConfig` |
| 分析层兼容 | 若新增 `GranularityMode`，检查 `iter_slices` 是否需要同步更新 |

## 派遣模板

```
对刚完成的 <子任务名> 做代码 review。
背景：<一句话描述该子任务做了什么>
检查文件：<列出修改的文件路径>
重点检查：<针对该子任务的具体风险点>
参照规范：docs/architecture/<对应 ADR>
输出：每个问题带文件:行号，最后给严重程度总结表格。
```
