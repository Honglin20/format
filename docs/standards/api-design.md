# API 设计约束

> 这些规则来自实战 review 中反复出现的问题。**每次新增公共 API 时必须对照检查。**

## 可哈希抽象基类

若 ABC 的实例会用作 frozen dataclass 的字段（如 `TransformBase` 用于 `QuantScheme`），必须在 ABC 中将 `__eq__` 和 `__hash__` 声明为 `@abstractmethod`，强制子类实现，防止 id-based hash 静默破坏值相等性。

## `__post_init__` 验证完整性

frozen dataclass 的 `__post_init__` 必须对**全部字段**做类型验证。新增字段时，同步更新 `__post_init__` 的字段校验白名单，并在 review agent 清单里显式对照字段逐项确认。

漏掉一个字段就会在运行时产生难以定位的 AttributeError（典型案例：P2F-7 的 `granularity` 字段漏验证）。

## 签名变更稳定性

向已有工厂方法中间插入新位置参数时，必须用关键字专用参数（`def f(a, *, new_param=0, old_kw=...)`）或在函数开头对旧类型做守卫（`if isinstance(new_param, str): raise TypeError(...)`），防止旧调用方式静默变成错误语义。

## 无静默默认值

构造函数默认值不得隐藏重要语义（如 `format="int8"` 的隐式默认、`granularity=per_tensor` 的隐式默认）。

有非显然默认值时，**docstring 必须在类级和字段级各写一次**"默认行为是 X"——类级说明便于概览，字段级注解便于 IDE tooltip。

## 维度索引的负值

所有 `axis` / `channel_axis` 参数必须在文档和验证中明确声明是否支持负值（PyTorch 风格的 -1 = last dim）。

- 不支持则加 `axis >= 0` 校验
- 支持则需分两层保证：
  1. 文档层说明支持
  2. 运行时层在可验证位置（持有张量形状的位置，如 `Format.quantize()`）做越界断言

**不能假设"只要用户知道就不会传错"。**

## 跨对象一致性验证

某些约束涉及多个对象（如 `GranularitySpec.channel_axis` 是否有效依赖于 tensor shape），无法在单个对象 `__post_init__` 中验证。这类"延迟验证"必须：

1. 文档说明"此字段的越界 / 一致性检查在 `<具体函数>` 中动态做"
2. 在该函数中有显式断言并带清晰错误信息
3. 配一条 `pytest.raises` 测试覆盖该动态路径

不允许"静默假设在另一处会检查"。

## Format ONNX 导出 Strategy 模式

新增 Format 子类时，如需不同于默认 `MxQuantize` 的 ONNX 行为，须覆写 `export_onnx(self, g, x, scheme)`。不要在 helpers.py 中再添加硬编码 format name 分派。

## JIT tracing 与量化路径分离

PyTorch old-style ONNX exporter 分两阶段：
1. JIT tracing（调 `forward()` 做 shape 推断）
2. ONNX 建图（调 `symbolic()` 生成节点）

`_quantize_per_block()` 已加 `torch.jit.is_tracing()` guard 跳过 `_reshape_to_blocks`。新增量化路径时，若涉及 JIT-unfriendly 操作（Tensor.item()、动态 shape 分支），必须在入口加同样的 guard。symbolic() 负责在 ONNX 图中表达量化语义，forward() 只负责 shape 推断。
