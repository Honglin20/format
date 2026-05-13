# Session 概览

> 第 1 章 · [Session 文档索引](INDEX.md)

Session 是单次量化实验的生命周期管理。一个 `QuantConfig` → 一个 `Session` → 一个 `SessionResult`。

## 两种使用模式

### 一键模式

```python
from src.session import Session, QuantConfig

result = Session(model, QuantConfig(w_format="int8")).run(
    calib_data, eval_fn=eval_fn)
print(result.summary())
```

### 分步模式

方便检查和调试每一阶段：

```python
session = Session(model, cfg)
session.quantize(calib_data=calib_data)   # 1. 构建量化模型
# session.qmodel 此时可用，可以手动推理
session.calibrate(calib_data)             # 2. 计算 scale
session.analyze(calib_data)               # 3. 误差分析 (QSNR/MSE)
session.evaluate(eval_data, eval_fn)      # 4. 精度评估
session.cost()                            # 5. 性能估算
result = session.result                   # 6. 获取结果
```

所有步骤方法返回 `self`，可以链式调用：

```python
session.quantize(calib_data=calib_data).calibrate(calib_data).analyze(calib_data)
```

## 管道流程

```
quantize()  →  calibrate()  →  analyze()  →  evaluate()  →  cost()
  构建量化模型    计算 scale     误差分析      精度评估      性能估算
```

`Session.run()` 内部按顺序执行这五步（evaluate 和 cost 在指定 outputs/eval_fn 时触发）。

## 推理模式切换

```python
session = Session(model, cfg).quantize()
output = session(x)        # 量化模型推理

session.use_fp32()
output = session(x)        # 切换回 fp32

session.use_quant()
output = session(x)        # 再切回量化
```

**Session 不会修改原始模型**：`quantize()` 内部 deepcopy，原模型不变。

## 常用属性

| 属性 | 说明 |
|------|------|
| `session.qmodel` | 量化后的模型（quantize() 之后可用） |
| `session.fp32_model` | 原始模型副本（keep_fp32=True） |
| `session.mode` | 当前推理模式：`"fp32"` 或 `"quant"` |
| `session.result` | 构建 SessionResult |

## eval_fn 合约

`accuracy_table()`、`summary()` 的准确率列、`pareto_frontier(metric="accuracy")` 依赖 `eval_fn`。

```python
def eval_fn(model, data) -> dict[str, float]:
    """在 data 上运行 model，返回 {指标名: 浮点值}。

    model — Session 当前推理模式下的模型（evaluate 阶段会分别传 fp32 和 quant）。
    data — eval_data（未单独传入时用 calib_data）。
    """
    ...
    return {"loss": 0.1234}             # 单指标
    # 或
    return {"loss": 0.12, "acc": 0.95}   # 多指标
```

**不传 `eval_fn`** → `evaluate()` 阶段被跳过 → `accuracy_table()` 显示 `(no accuracy metrics — run with eval_fn)`。

## run() 参数

```python
Session(model, cfg).run(
    calib_data,              # 校准数据（必传）
    eval_data=None,          # 评估数据（不传则用 calib_data）
    eval_fn=None,            # (model, data) -> Dict[str, float]（见上方合约）
    outputs="default",       # "default" / "all" / ["qsnr", "mse", ...]
)
```

---
← [Session 文档索引](INDEX.md) | [下一章：QuantConfig 配置](config.md) →
