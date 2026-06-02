# GPTQ 幂等修复设计

## 问题

GPTQ 内部用 `_precompute_scale(W_f32)` 从原始 FP32 权重算出 per-channel amax，
基于这个固定 scale 逐列量化 + Hessian 补偿。但写入 `module.weight.data` 的是
FP32 值（非量化后的离散值），forward pass 中 `quantize(w, cfg.weight)` 不传
`scale`，重新从修改后的权重算 amax — 两个 amax 不同，re-quant 产出不同结果。

诊断数据（MNIST MLP, Linear 512x784, int4 per_channel）：

| scheme | naive MSE | GPTQ MSE | GPTQ re-quant MSE | idempotent |
|--------|-----------|----------|--------------------|------------|
| int4-pc | 0.000312 | 0.000397 | 0.000427 | False |
| int4-pc-sparse | 0.000045 | 0.000056 | 0.000089 | False |
| int4-pc-gsparse | 0.000184 | 0.000231 | 0.000246 | False |

GPTQ re-quant MSE 始终大于 GPTQ MSE，导致 GPTQ 不仅不提升精度，反而降低。

## 方案

GPTQ 写入 `_weight_scale` buffer，forward 读 buffer 而非重算 amax。

### 改动清单

1. **`GPTQOptimizer.optimize()`** — 量化后把 `full_scale` 注册为
   `module._weight_scale` buffer（`register_buffer`）
2. **`CalibrationBuffers`** — 加 `weight_scale: Optional[torch.Tensor] = None`
3. **`QuantizedLinear.forward()`** — 从 buffer 读取 `_weight_scale`，传入
   `CalibrationBuffers`
4. **`LinearFunction.forward()`** — `quantize(w, cfg.weight,
   scale=buffers.weight_scale, importance=...)` 传入 scale
5. **`QuantizedConv2d.forward()`** + `ConvFunction.forward()` — 同样读
   `_weight_scale` buffer
6. **`QuantizedConvTranspose2d`** + `ConvTransposeFunction` — 同上

### 向后兼容

`_weight_scale` 只在 GPTQ（或未来其他 weight calibration）设置时存在。
没有该 buffer 时 `quantize()` 的 `scale=None` 行为不变（从权重重算 amax）。

### 不改动的部分

- `quantize()` 函数本身 — 已经支持 `scale` 参数
- `_precompute_scale()` — 逻辑正确，只是结果没被保存
- sparse/group_sparse 的 mask 传递 — 属于后续独立问题
