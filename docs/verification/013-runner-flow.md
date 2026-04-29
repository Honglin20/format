# 013: ExperimentRunner.run() — 多配置实验管线合约

**对应测试函数**: `test_runner_flow()`, `test_runner_edge_cases()`
**验证层级**: Layer 4 — Pipeline Refactor

## 验证内容

`ExperimentRunner.run()` 的完整合约：输入类型、执行流程、输出结构、边界行为。该 API 是一个高层实验调度器，接收搜索空间（纯数据）和运行时参数（fp32 模型、eval_fn、各类数据），对搜索空间中的每组配置依次执行 deepcopy → QuantSession → calibrate → analyze → evaluate，汇总返回所有结果。

## 合约

### 签名

```python
class ExperimentRunner:
    def __init__(self, search_space: dict):
        ...

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable[[nn.Module, Any], Dict[str, float]],
        calib_data: Any = None,
        analyze_data: Any = None,
        eval_data: Any = None,
        observers: list | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        ...
```

> **设计原理**: 搜索空间（纯数据，定义"要跑哪些实验"）在构造时注入；运行时参数（模型、数据、回调）在 `run()` 时注入。这允许同一个 `ExperimentRunner` 实例对不同的模型/数据集重复调用 `run()`。

### 搜索空间格式

搜索空间是纯 Python dict，在 `ExperimentRunner.__init__` 时注入：

```python
search_space = {
    "<study_part>": {
        "description": "...",            # 可选，人类可读描述
        "calibrator": "mse",             # 可选，当前固定使用 MSEScaleStrategy
        "configs": {
            "<config_name>": {           # descriptor dict → resolve_config() 解析
                "format": "int8",
                "granularity": "per_channel",
                "axis": 0,
            },
            # 或直接传 OpQuantConfig 实例
            "<config_name>": OpQuantConfig(...),
        },
    },
}
```

每对 `(study_part/config_name)` 构成一个独立实验，结果键为 `"<study_part>/<config_name>"`。

### run() 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `fp32_model` | `nn.Module` | 是 | 原始 FP32 模型。每个 config 从头 deepcopy，原模型不被修改。 |
| `eval_fn` | `(nn.Module, Any) -> Dict[str, float]` | 是 | 用户提供的评估函数，三个阶段均使用该回调。接收 model/session 和数据，返回 flat dict（如 `{"accuracy": 0.85, "loss": 0.5}`）。 |
| `calib_data` | Any 或 None | 否 | 校准数据。传给 `eval_fn` 驱动前向传播触发 hooks。`None` 跳过校准阶段。 |
| `analyze_data` | Any 或 None | 否 | 误差分析数据。默认回退到 `calib_data`。`None`（且 `calib_data` 也为 None）跳过分析阶段。 |
| `eval_data` | Any 或 None | 否 | 评估数据。传给 `eval_fn` 计算 fp32 vs quant 指标。`None` 跳过评估阶段（`fp32`/`quant`/`delta` 字段均为 `None`）。 |
| `observers` | `List[Observer]` 或 None | 否 | 挂载到 `AnalysisContext` 的 Observer 列表。`None` 使用默认 `[QSNRObserver(), MSEObserver()]`；空列表 `[]` 表示不挂载任何 Observer。 |

> **eval_fn 调用的三个阶段**：
> - **校准**: `eval_fn(session, batch)` — 只利用 forward 副作用触发 hooks，忽略返回值
> - **分析**: `eval_fn(session, batch)` — 同上，Observer 在各量化关键点被动收集数据
> - **评估**: `fp32_metrics = eval_fn(deepcopy(fp32_model), eval_data)` 和 `quant_metrics = eval_fn(session, eval_data)` — 使用返回值计算 delta

### 执行流程

```
for each study_part, part_def in search_space:
    for each config_name, cfg_desc in part_def.configs:
        1. cfg = resolve_config(cfg_desc) 或直接使用 OpQuantConfig 实例
        2. session = QuantSession(deepcopy(fp32_model), cfg, calibrator=MSEScaleStrategy(), keep_fp32=True)
        3. if calib_data is not None:
             with session.calibrate():
                 for batch in calib_data:
                     eval_fn(session, batch)          # forward 副作用，返回值忽略
        4. report = None
           analyze_input = analyze_data or calib_data
           if analyze_input is not None:
               with session.analyze(observers=observers) as ctx:
                   for batch in analyze_input:
                       eval_fn(session, batch)        # forward 副作用，返回值忽略
               report = ctx.report()
        5. if eval_data is not None:
               fp32_metrics = eval_fn(deepcopy(fp32_model), eval_data)
               quant_metrics = eval_fn(session, eval_data)
               delta = {k: quant_metrics[k] - fp32_metrics[k] for k in fp32_metrics}
           else:
               fp32_metrics = quant_metrics = delta = None
        6. entry = {"fp32": fp32_metrics, "quant": quant_metrics, "delta": delta}
           if report is not None:
               entry["report"] = report
               entry["qsnr_per_layer"] = extract_metric_per_layer(report, "qsnr_db")
               entry["mse_per_layer"] = extract_metric_per_layer(report, "mse")
        7. results[f"{study_part}/{config_name}"] = entry
```

> **Note**: `MSEScaleStrategy()` 作为默认校准策略，区别于 `QuantSession` 的 `MaxScaleStrategy()` 默认。这是有意的：研究实验受益于 MSE 最优 scale；`QuantSession` 的 `MaxScaleStrategy` 是通用场景的保守默认。

### 返回值

```python
{
    "part_a_8bit/MXINT-8": {
        "fp32":  {"accuracy": 0.85, "loss": 0.5},          # eval_fn 在 fp32 模型上的结果（eval_data=None 时为 None）
        "quant": {"accuracy": 0.82, "loss": 0.6},          # eval_fn 在量化模型上的结果（eval_data=None 时为 None）
        "delta": {"accuracy": -0.03, "loss": 0.1},         # quant - fp32（eval_data=None 时为 None）
        "report": Report(...),                              # 仅在 analyze 执行且 report 非空时存在
        "qsnr_per_layer": {"0.linear": 42.99, ...},        # 可选 — 仅在 report 非空时存在
        "mse_per_layer": {"0.linear": 0.0012, ...},        # 可选 — 仅在 report 非空时存在
    },
    "part_a_8bit/MXFP-8": { ... },
    "part_b_4bit/MXINT-4": { ... },
}
```

**键命名规则**: `"{study_part}/{config_name}"`，其中 `study_part` 是搜索空间的一级键，`config_name` 是 `configs` 下的二级键。

**字段存在性规则**:
- `"fp32"`, `"quant"`, `"delta"`: 始终存在。`eval_data=None` 时值为 `None`。
- `"report"`: 仅在 analyze 阶段执行且 `ctx.report()` 非空时存在。否则 key 不存在于 entry 中。
- `"qsnr_per_layer"`, `"mse_per_layer"`: 仅在 `"report"` 存在时存在。

### 异常与守卫

| 场景 | 期望异常 | 说明 |
|------|---------|------|
| `eval_fn` 参数类型错误 | `TypeError` | 对非 callable 的 eval_fn 立即 raise |
| `eval_fn` 返回非 dict | `TypeError` | 任何非 dict 返回值视为违反合约，立即 raise |
| `search_space` 为空字典 | 返回空字典 `{}` | 空输入无实验需运行，直接返回 |
| `calib_data=None` 且 `analyze_data=None` 且 `eval_data=None` | 返回条目含 `"fp32": None, "quant": None, "delta": None`，不含 `"report"` / `"qsnr_per_layer"` / `"mse_per_layer"` | 仅构建 session，跳过所有阶段 |
| `eval_data=None` | `fp32` / `quant` / `delta` 字段为 `None` | 不影响 calibrate 和 analyze 阶段 |
| `analyze_data=None` 且 `calib_data=None` | `report`、`qsnr_per_layer`、`mse_per_layer` 字段不存在于返回值中 | 不影响 calibrate（calib_data=None 时也跳过）和 evaluate 阶段 |
| `calib_data=None` 但 cfg 依赖 output scales | 模型使用未校准的 `_output_scale` | 这是用户责任，runner 不隐式跳过 |
| 配置中的 `OpQuantConfig` 字段为空 (all None) | 模型保持不变（identity） | 全 None 的 cfg 相当于不量化 |
| `eval_fn` 内部对无效 label 抛出异常 | 向上传播 | runner 不做 eval_fn 的异常包装 |
| 多次调用 `run()` | 每次独立 deepcopy + 运行 | 不对内部 state 做缓存假设 |

## 期望行为

### 场景 1: 完整管线（3 配置 × 全数据）

```python
runner = ExperimentRunner(search_space)
results = runner.run(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    calib_data=calib_loader,
    analyze_data=analyze_loader,
    eval_data=eval_loader,
    observers=[QSNRObserver(), MSEObserver()],
)
```

期望：
- `results` 的键为 `"{study_part}/{config_name}"` 格式
- 每个条目含 `fp32`, `quant`, `delta`, `report`；当 `report` 非空时额外含 `qsnr_per_layer` 和 `mse_per_layer`
- 所有 `fp32` 子字段的度量值相同（deepcopy 后共用同一个原始模型权重的语义）
- 各 `quant` 子字段的度量值可能不同（不同量化配置导致不同精度损失）
- `report` 是 `Report` 实例，含每层每个量化点的 QSNR/MSE 数据

### 场景 2: 仅校准 + 评估（无分析）

```python
results = runner.run(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    calib_data=calib_loader,
    eval_data=eval_loader,
    # analyze_data=None (默认)
)
```

期望：
- 每个条目不包含 `"report"`、`"qsnr_per_layer"`、`"mse_per_layer"` 字段
- `"fp32"`, `"quant"`, `"delta"` 正常计算

### 场景 3: 仅评估（无校准、无分析）

```python
results = runner.run(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    eval_data=eval_loader,
)
```

期望：
- 跳过校准阶段 → session 使用的 scale 是构造时的初始值（如 1.0 或未赋值）
- 条目不包含 `"report"`、`"qsnr_per_layer"`、`"mse_per_layer"` 字段
- `"fp32"`, `"quant"`, `"delta"` 正常计算

### 场景 4: 仅校准（无评估、无分析）

```python
results = runner.run(
    fp32_model=model,
    eval_fn=my_accuracy_fn,
    calib_data=calib_loader,
)
```

期望：
- 每个条目 `"fp32"`, `"quant"`, `"delta"` 均为 `None`
- 条目不包含 `"report"`、`"qsnr_per_layer"`、`"mse_per_layer"` 字段
- 模型内部已写入 `_output_scale` 缓冲（校准的副作用生效）

### 场景 5: 空搜索空间

```python
runner = ExperimentRunner({})
results = runner.run(fp32_model=model, eval_fn=my_eval_fn)
```

期望：
- `results == {}`

### 场景 6: eval_fn 返回非 dict

```python
def bad_eval_fn(model, data):
    return 0.85  # 标量，不是 dict
```

期望：
- `runner.run(...)` 在 evaluate 阶段抛出 `TypeError`

### 场景 7: eval_fn 参数类型错误

```python
runner.run(fp32_model=model, eval_fn="not_a_function", ...)
```

期望：
- 调用时 `eval_fn(session, batch)` 抛出 `TypeError`

### 场景 8: 多配置共享 fp32 deepcopy 隔离性

验证 after `run()`, `fp32_model` 状态未被修改（weight/bias 未变）、传入的 `search_space` dict 未被 runner 修改。

## 验证结果

- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
- [ ] 说明
