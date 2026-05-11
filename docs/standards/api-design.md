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

## Result 类访问器模式

所有持有分析数据的 result dataclass 必须暴露 `.tables` 访问器属性（终端/格式化输出）。必要时也可暴露 `.plot` 访问器（可视化）。

**模式**：

```python
# 1. 访问器类：构造时接收父对象引用，方法操作父对象数据
class XxxTablesAccessor:
    def __init__(self, report: "XxxReport"):
        self._report = report

    def some_table(self, ...) -> str:
        ...  # 从 self._report 读取数据，返回格式化文本

# 2. 在父对象上以 @property 暴露，lazy import 避免循环依赖
@property
def tables(self) -> "XxxTablesAccessor":
    from src.xxx._tables import XxxTablesAccessor
    return XxxTablesAccessor(self)
```

**当前遵循此模式的类**：

| 父类 | 访问器 | `_report` 字段 |
|------|-------|---------------|
| `StudyReport` | `.tables` → `StudyTablesAccessor` | `self._report._results` |
| `StudyReport` | `.plot` → `StudyPlotAccessor` | `self._report._results` / `self._report.to_dataframe()` |
| `SessionResult` | `.tables` → `SessionTablesAccessor` | `self._result`（qsnr_per_layer / accum_qsnr_per_layer / qsnr_per_role） |

**规则**：

1. 访问器方法用公共名（不下划线），返回 `str`（tables）或 `plt.Figure`（plot）。
2. 访问器内部不做数据采集——只读取已有数据并格式化。数据采集在 Session 阶段完成。
3. 访问器构造函数只接受一个参数：父对象引用。
4. 跨 accessor 共享的计算逻辑放在父类，方法名不下划线表示公共 API。逻辑必须放在最小数据单元上（如 `SessionResult.correlate_hook_observer()`），聚合层通过迭代委托复用。
5. 数据缺失时抛出 `ValueError` 并提供操作指导（告知缺少哪些 `outputs` key）。

## 能力归属原则

分析能力必须归属到最小数据单元（`SessionResult`），聚合层（`StudyReport`）不做独立分析，只负责跨配置对比和格式化。

| 层级 | 角色 | 示例 |
|------|------|------|
| `SessionResult` | 拥有分析能力 | `correlate_hook_observer()`, `qsnr_per_role()`, `.report`, `.tables` |
| `StudyReport` | 跨配置委托聚合 | `correlate_hook_observer()` 聚合各 result；`.tables` / `.plot` 跨配置叠加 |

**规则**：
1. 新增分析能力时，先在 `SessionResult` 上实现，`StudyReport` 通过迭代委托复用。
2. `StudyReport` 的方法如果只做聚合（`for r in results: r.method()`），也必须暴露为公共 API，便于 `StudyTablesAccessor` 和 `StudyPlotAccessor` 调用。
3. 核心算法只存在于 `SessionResult`，不得在 `StudyReport` 中重复实现。
