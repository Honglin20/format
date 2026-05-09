# Session API 参考

```python
from src.session import Session
```

## 构造

```python
session = Session(
    model,           # nn.Module
    config,          # QuantConfig
    keep_fp32=True,  # 是否保留 fp32 参考模型
)
```

## 生命周期方法（链式）

所有方法返回 `self`。

```python
session.quantize(*, calib_data=None)                          # 构建量化模型
session.calibrate(calib_data, *, eval_fn=None)                 # 计算 scale
session.analyze(calib_data, *, outputs="default", eval_fn=None) # 误差分析
session.evaluate(eval_data, eval_fn)                           # 精度评估
session.cost()                                                 # 性能估算
```

### quantize()

- 构建量化模型，之后 `session.qmodel` 可用
- `calib_data` 在 `transform="smoothquant"` 或 `"prescale"` 时必传
- MX per_block 格式不需要 calib_data

### calibrate()

- 计算量化 scale。MX per_block 自动跳过
- `transform="adaptive"` 时，此阶段做逐层 transform 选择

### analyze()

- `outputs`: `"default"` | `"all"` | `["qsnr", "mse", "histogram", "distribution", "fit"]`

### evaluate()

- `eval_fn(model, data) -> Dict[str, float]`：同时在 fp32 和 quant 模型上调用

## 一键方法

```python
result = session.run(
    calib_data,
    *,
    eval_data=None,
    eval_fn=None,
    outputs="default",
)
```

等价于 `quantize → calibrate → analyze → evaluate → cost → result`，evaluate/cost 根据 outputs 按需执行。

## 推理方法

```python
output = session(x)            # 量化模型推理
session.use_fp32()             # 切换 fp32 模式
session.use_quant()            # 切换量化模式
mode = session.mode            # "fp32" 或 "quant"
```

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `session.qmodel` | `nn.Module` | 量化后的模型（quantize() 之后） |
| `session.fp32_model` | `nn.Module\|None` | fp32 参考模型副本 |
| `session.result` | `SessionResult` | 构建结果对象 |
| `session.mode` | `str` | `"fp32"` 或 `"quant"` |

## 其他公开 API

```python
from src.session import (
    QuantConfig,       # 用户面量化配置
    Study,             # 多配置聚合对比
    quantize_model,    # 底层模块替换（model → qmodel）
    QuantizeContext,   # torch/F inline 拦截上下文
    per_layer_optimal, # 逐层最优配置搜索
)
```
