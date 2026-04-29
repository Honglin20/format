# 012: Pipeline — QuantSession E2E 验证

**对应测试函数**: `test_pipeline_quant_session_e2e()`
**验证层级**: Layer 3 — 完整 Pipeline

## 验证内容

QuantSession 完整流程：quantize_model → CalibrationSession → AnalysisContext，验证：
1. 量化后的模型输出与手工计算一致
2. CalibrationSession 正确收集 output amax 并计算 scales
3. AnalysisContext 正确计算 QSNR

## 配置

```python
# Simple Linear(2, 3) model
model = nn.Linear(2, 3)
model.weight.data = W = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
model.bias.data   = b = [0.1, 0.2, 0.3]

# QuantConfig: int8 per_channel(axis=0) for all roles
scheme = QuantScheme(format=IntFormat(bits=8), granularity=GranularitySpec.per_channel(axis=0))
cfg = OpQuantConfig(storage=scheme, input=scheme, weight=scheme, output=scheme)

session = QuantSession(model, cfg, calibrator=MSEScaleStrategy(n_steps=5))
```

## 手工推导

### Step 1: quantize_model

`model` 被替换为 `QuantizedLinear`，weight/bias 被复制。forward 被 patch 为 QuantizeContext 包装。

### Step 2: calibrate — forward pass

输入 x = [[0.5, -0.25], [1.0, 0.75]]

CalibrationSession 注册 forward hook，在 forward 后捕获 output amax：
- output = QuantizedLinear.forward(x) = y_q（与 Test 007 相同）
- y_q = [[0.12151, 0.67727, 1.37813], [2.59219, 6.19219, 9.8]]
- amax per column (axis=0): max(|col0|)=2.59219, max(|col1|)=6.19219, max(|col2|)=9.8

### Step 3: calibrate — scale computation

MSEScaleStrategy(n_steps=5) 对每列做 grid search：

- col0 amax=2.59219 → best scale = 2.59219 * 1.625 = **4.21230...**
- col1 amax=6.19219 → best scale = 6.19219 * 1.625 = **10.06230...**
- col2 amax=9.8 → best scale = 9.8 * 1.625 = **15.925**

这些 scale 被 assign 为 `_output_scale` buffer。

### Step 4: analyze

AnalysisContext 挂载 QSNRObserver → forward pass → 在量化关键点记录 fp32/quant 对 → 计算 QSNR。

## 期望值

由于 MSEScaleStrategy grid search 和 forward hook 的交互涉及精确 float32 计算，期望值通过 torch 预计算确定：

```python
# calibrate 后的 _output_scale
expected_scales = {
    "fc": tensor([4.212304, 10.062304, 15.925])  # shape (3,)
}

# calibrate 后的模型输出（带 output scale 的完整量化）
# 与 Test 007 的输出一致
expected_y = torch.tensor([[0.12151..., 0.67727..., 1.37813...],
                           [2.59219..., 6.19219..., 9.8]])
```

## 验证结果

- [x] 运行日期: 2026-04-29
- [x] 结果: PASS
- [x] 说明: quantize_model forward 与 Test 007 bit-exact；CalibrationSession amax/scales 正确；AnalysisContext report 结构验证通过。
