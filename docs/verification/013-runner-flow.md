# 013: ExperimentRunner.run() — 多配置实验管线合约

**对应测试函数**: `test_runner_flow()`, `test_runner_edge_cases()`
**验证层级**: Layer 4 — Pipeline Refactor

## 验证内容

`ExperimentRunner.run()` 的完整合约：输入类型、执行流程、输出结构、边界行为。该 API 是一个高层实验调度器，接收一个 fp32 模型和多组量化配置，对每组配置依次执行 deepcopy → QuantSession → calibrate → analyze → evaluate，汇总返回所有结果。

## 合约

### 签名

```python
class ExperimentRunner:
    def __init__(
        self,
        fp32_model: nn.Module,
        eval_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
        calib_data: Optional[Iterable[torch.Tensor]] = None,
        analyze_data: Optional[Iterable[torch.Tensor]] = None,
        eval_data: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
        observers: Optional[List[Observer]] = None,
        *,
        default_calibrator: ScaleStrategy = MSEScaleStrategy(),
    ):
        ...

    def run(
        self,
        configs: Dict[str, OpQuantConfig],
    ) -> Dict[str, Dict[str, Any]]:
        ...
```

### 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `fp32_model` | `nn.Module` | 是 | 原始 FP32 模型，作为精度上界参照。每个配置从头 deepcopy。 |
| `eval_fn` | `(logits, labels) -> Dict[str, float]` | 是 | 用户提供的评估函数，三个阶段均使用该回调。必须返回 flat dict（如 `{"accuracy": 0.85, "loss": 0.5}`）。 |
| `calib_data` | `Iterable[Tensor]` 或 None | 否 | 校准数据。每个元素是单张量输入（无 label）。`None` 跳过校准阶段。 |
| `analyze_data` | `Iterable[Tensor]` 或 None | 否 | 误差分析数据。每个元素是单张量输入。`None` 跳过分析阶段。 |
| `eval_data` | `Iterable[Tuple[Tensor, Tensor]]` 或 None | 否 | 评估数据。每个元素是 `(input, label)` 二元组。`None` 只跑校准和分析，不执行 E2E 评估。 |
| `observers` | `List[Observer]` 或 None | 否 | 挂载到 `AnalysisContext` 的 Observer 列表。默认 `[QSNRObserver()]`。 |
| `default_calibrator` | `ScaleStrategy` | 否 | 各配置的默认校准策略。可通过 study part 覆盖。默认 `MSEScaleStrategy()`。 |

### Config 格式

```python
configs = {
    "MXINT-8": OpQuantConfig(
        weight=QuantScheme(format=int8_fmt, granularity=PER_B32),
        input=QuantScheme(format=int8_fmt, granularity=PER_B32),
        output=QuantScheme(format=int8_fmt, granularity=PER_B32),
        storage=QuantScheme(format=bfloat16_fmt, granularity=PER_T),
    ),
    "INT8-PC": OpQuantConfig(
        weight=QuantScheme(format=int8_fmt, granularity=PER_C0),
        input=QuantScheme(format=int8_fmt, granularity=PER_Cm1),
        output=QuantScheme(format=int8_fmt, granularity=PER_Cm1),
        storage=QuantScheme(format=bfloat16_fmt, granularity=PER_T),
    ),
}
```

每对 `(config_name, OpQuantConfig)` 定义一个独立的量化实验。

### 执行流程

```
for each (name, cfg) in configs:
    1. deepcopy(fp32_model)                              # 避免 state 污染
    2. QuantSession(model_copy, cfg, calibrator=...)      # 构建 session
    3. if calib_data is not None:
         calibrate(session, calib_data)                   # eval_fn 副作用阶段
    4. if analyze_data is not None:
         report = analyze(session, analyze_data, observers)  # observer hooks 阶段
    5. if eval_data is not None:
         result = compare(session, eval_data, eval_fn)    # fp32 vs quant 指标对比
    6. aggregate result into output dict
```

### 各阶段详细说明

**阶段 1: calibrate**

```python
with session.calibrate():
    for input_tensor in calib_data:
        session(input_tensor)        # 前向传播 → forward hook 收集激活 amax
    # exit: 自动计算并回写 _output_scale 到各模块
```

校准完成后，`eval_fn` 此时并未被调用。校准的副作用（`_output_scale` 写入）已持久化到 `session.qmodel`。

**阶段 2: analyze**

```python
with session.analyze(observers=observers) as ctx:
    for input_tensor in analyze_data:
        session(input_tensor)        # 前向传播 → Observer 在各量化关键点记录 fp32/quant 对
report = ctx.report()                # 聚合所有 Observer 数据 → Report 对象
```

此阶段中 `eval_fn` 未被调用。仅 Observer 被动收集数据。

**阶段 3: evaluate**

```python
# fp32 baseline — eval_fn receives model + data, handles iteration internally
session.use_fp32()
fp32_metrics = eval_fn(session, eval_data)      # → {"accuracy": 0.85, "loss": 0.5}

# quant inference — same eval_fn, same eval_data
session.use_quant()
quant_metrics = eval_fn(session, eval_data)     # → {"accuracy": 0.82, "loss": 0.6}

# delta = quant - fp32 (key-wise subtraction)
delta = {k: quant_metrics[k] - fp32_metrics[k] for k in fp32_metrics}
```

### 返回值

```python
{
    "MXINT-8": {
        "fp32":  {"accuracy": 0.85, "loss": 0.5},          # eval_fn 在 fp32 模型上的结果
        "quant": {"accuracy": 0.82, "loss": 0.6},          # eval_fn 在量化模型上的结果
        "delta": {"accuracy": -0.03, "loss": 0.1},         # quant - fp32
        "report": Report(raw),                             # AnalysisContext.report() 的返回
        "qsnr_per_layer": {                                # 可选 — 从 report 提取的 QSNR 摘要（层名 → dB）；仅在 report 非空时存在
            "0.linear": 42.99,
            ...
        },
        "mse_per_layer": {                                 # 可选 — 从 report 提取的 MSE 摘要（层名 → 标量）；仅在 report 非空时存在
            "0.linear": 0.0012,
            ...
        },
    },
    "INT8-PC": {
        "fp32":  {"accuracy": 0.85, "loss": 0.5},
        "quant": {"accuracy": 0.83, "loss": 0.58},
        "delta": {"accuracy": -0.02, "loss": 0.08},
        "report": Report(raw),
        "qsnr_per_layer": {                                # 可选 — 仅在 report 非空时存在
            "0.linear": 44.57,
            ...
        },
        "mse_per_layer": {                                 # 可选 — 仅在 report 非空时存在
            "0.linear": 0.00053,
            ...
        },
    },
}
```

### 异常与守卫

| 场景 | 期望异常 | 说明 |
|------|---------|------|
| `eval_fn` 参数类型错误 | `TypeError` | 对非 callable 的 eval_fn 立即 raise |
| `eval_fn` 返回非 dict | `TypeError` | 任何非 dict 返回值视为违反合约，立即 raise |
| `configs` 为空字典 | 返回空字典 `{}` | 空输入无实验需运行，直接返回 |
| `calib_data=None` 且 `analyze_data=None` 且 `eval_data=None` | 返回含 `report=None` 的条目 | 仅构建 session，跳过所有阶段 |
| `eval_data=None` | `fp32` / `quant` / `delta` 字段为 `None` | 不影响 calibrate 和 analyze 阶段 |
| `analyze_data=None` | `report`、`qsnr_per_layer`、`mse_per_layer` 字段为 `None` | 不影响 calibrate 和 evaluate 阶段 |
| `calib_data=None` 但 cfg 依赖 output scales | 模型使用未校准的 `_output_scale` | 这是用户责任，runner 不隐式跳过 |
| 配置中的 `OpQuantConfig` 字段为空 (all None) | 模型保持不变（identity） | 全 None 的 cfg 相当于不量化 |
| `eval_fn` 内部对无效 label 抛出异常 | 向上传播 | runner 不做 eval_fn 的异常包装 |
| 多次调用 `run()` | 每次独立 deepcopy + 运行 | 不对内部 state 做缓存假设 |

## 期望行为

### 场景 1: 完整管线（3 配置 × 全数据）

```python
runner = ExperimentRunner(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    calib_data=calib_loader,
    analyze_data=analyze_loader,
    eval_data=eval_loader,
    observers=[QSNRObserver(), MSEObserver()],
)
results = runner.run({
    "MXINT-8": mxint8_cfg,
    "INT8-PC": int8_pc_cfg,
    "FP8-PB":  fp8_pb_cfg,
})
```

期望：
- `results` 有 3 个键：`"MXINT-8"`, `"INT8-PC"`, `"FP8-PB"`
- 每个条目含 `fp32`, `quant`, `delta`, `report`；当 `report` 非空时额外含 `qsnr_per_layer` 和 `mse_per_layer`
- 所有 `fp32` 子字段的度量值相同（deepcopy 后共用同一个原始模型权重的语义，评估指标聚合过程一致）
- 各 `quant` 子字段的度量值可能不同（不同量化配置导致不同精度损失）
- `report` 是 `Report` 实例，含每层每个量化点的 QSNR/MSE 数据
- `qsnr_per_layer` 和 `mse_per_layer` 是从 `report` 提取的摘要（可选 — 仅在 `report` 非空时存在）

### 场景 2: 仅校准 + 评估（无分析）

```python
runner = ExperimentRunner(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    calib_data=calib_loader,
    eval_data=eval_loader,
    # analyze_data=None (默认)
)
```

期望：
- 每个条目 `"report"` 为 `None`
- `"qsnr_per_layer"` 和 `"mse_per_layer"` 为 `None`
- `"fp32"`, `"quant"`, `"delta"` 正常计算

### 场景 3: 仅评估（无校准、无分析）

```python
runner = ExperimentRunner(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    eval_data=eval_loader,
)
```

期望：
- 跳过校准阶段 → session 使用的 scale 是构造时的初始值（如 1.0 或未赋值）
- `"report"` 为 `None`
- `"fp32"`, `"quant"`, `"delta"` 正常计算

### 场景 4: 仅校准（无评估、无分析）

```python
runner = ExperimentRunner(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    calib_data=calib_loader,
    # analyze_data=None (默认)
    # eval_data=None (默认)
)
```

期望：
- 每个条目 `"fp32"`, `"quant"`, `"delta"` 均为 `None`
- `"report"` 为 `None`
- 模型内部已写入 `_output_scale` 缓冲（校准的副作用生效）

### 场景 5: 空配置字典

```python
results = runner.run({})
```

期望：
- `results == {}`

### 场景 6: eval_fn 返回非 dict

```python
def bad_eval_fn(logits, labels):
    return 0.85  # 标量，不是 dict
```

期望：
- `runner.run(...)` 在 evaluate 阶段抛出 `TypeError`

### 场景 7: eval_fn 参数类型错误

```python
runner = ExperimentRunner(
    fp32_model=model,
    eval_fn="not_a_function",  # 字符串非 callable
    ...
)
```

期望：
- `ExperimentRunner.__init__()` 抛出 `TypeError`

### 场景 8: 多配置共享 fp32 deepcopy 隔离性

验证 after `run()`, `fp32_model` 状态未被修改（weight/bias 未变）、传入的 `configs` dict 未被 runner 修改。

## 验证结果

- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
- [ ] 说明
