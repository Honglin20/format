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
# Simple Linear(2, 3) model（须包在 nn.Sequential 中，quantize_model 替换 children）
model = nn.Sequential(nn.Linear(2, 3))
model[0].weight.data = W = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
model[0].bias.data   = b = [0.1, 0.2, 0.3]

# QuantConfig: int8 per_channel(axis=0) for compute roles（无 storage）
scheme = QuantScheme(format=IntFormat(bits=8), granularity=GranularitySpec.per_channel(axis=0))
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

session = QuantSession(model, cfg, calibrator=MSEScaleStrategy(n_steps=5), keep_fp32=False)
```

## 手工推导

### Step 1: quantize_model

`model` 被替换为 `QuantizedLinear`，weight/bias 被复制。forward 被 patch 为 QuantizeContext 包装。

### Step 2: calibrate — forward pass

输入 x = [[0.5, -0.25], [1.0, 0.75]]

CalibrationSession 注册 forward hook，在 forward 后捕获 output amax（按 axis=-1 即每行 keepdim=True → shape (2, 1)）：
- output = QuantizedLinear.forward(x) = y_q（与 Test 007 相同）
- y_q = [[0.12151, 0.67727, 1.37813], [2.59219, 6.19219, 9.8]]
- amax per row (axis=-1, keepdim=True):
  - row0: max(0.12151, 0.67727, 1.37813) = **1.37813**
  - row1: max(2.59219, 6.19219, 9.8) = **9.8**
- _running_amax["0"] shape (2, 1) = [[1.37813], [9.8]]

### Step 3: calibrate — scale computation

MSEScaleStrategy(n_steps=5) 对每行做 grid search：

- row0 amax=1.37813 → best factor=1.625 → scale = 1.37813 * 1.625 = **2.23945...**
- row1 amax=9.8 → best factor=1.625 → scale = 9.8 * 1.625 = **15.925**

scales["0"] shape (2, 1) = [[2.23945], [15.925]]（与 _running_amax 同 shape）

### Step 4: analyze

AnalysisContext 挂载 QSNRObserver → forward pass → 在量化关键点记录 fp32/quant 对 → 计算 QSNR。

## 期望值

由于 MSEScaleStrategy grid search 和 forward hook 的交互涉及精确 float32 计算，期望值通过 torch 预计算确定：

```python
# CalibrationSession 收集的 output amax（per-row, keepdim=True → shape (2, 1)）
expected_amax = {
    "0": torch.tensor([[1.37812507152557373047], [9.80000019073486328125]])  # shape (2, 1)
}

# CalibrationSession 计算的 scales（best_factor=1.625 → amax * 1.625）
expected_scales = {
    "0": torch.tensor([[2.23945331573486328125], [15.92500019073486328125]])  # shape (2, 1)
}

# quantize_model forward 输出与 Test 007 bit-exact 一致
expected_y = torch.tensor([[0.121508784592152, 0.677270472049713, 1.378125071525574],
                           [2.592187404632568, 6.192187309265137, 9.800000190734863]])
```

## 验证结果

- [x] 运行日期: 2026-04-29
- [x] 结果: PASS
- [x] 说明: quantize_model forward 与 Test 007 bit-exact；CalibrationSession amax/scales 正确；AnalysisContext report 结构验证通过。
