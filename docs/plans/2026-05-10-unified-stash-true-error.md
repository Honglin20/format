# True Error: Simplified Hook-Based Accumulation

> 以 fp32 原始模型 + forward hook 获取逐层累积 QSNR，修复 `_is_passthrough` 覆盖不全和多 batch 覆盖问题。不引入 stash、不建 passthrough Quantized* 树。

**状态**: 已实施（2026-05-10）

---

## 1. 问题诊断

当前 `Session.analyze(true_error=True)` 的实现有两个问题：

1. **`_is_passthrough` 覆盖不全**：只有 `QuantizedLinear` 设置了 `_is_passthrough = (cfg == OpQuantConfig())`。其他 21 个 Quantized* 类型没设置，`getattr(mod, "_is_passthrough", False)` 回退到 `False`，导致空 cfg 的非 Linear 模块被错误纳入 true-error 对比，产出 QSNR = ∞ 的无意义结果。

2. **多 batch 只保留最后一个**：`_run_model` 对 list 数据逐 batch 调用 `model(batch)`，hook 每次都覆盖 `fp32_refs[name]`，只保留最后一个 batch 的值。

根因不是"缺少 stash 机制"，而是过滤条件不准确 + 缺少多 batch 累加。

---

## 2. 最终方案（已实施）

### 核心原则

- **累积误差天然定义在模块边界**。只需要「Layer N 在 fp32 和量化模型分别输出什么」，不需要模块内部中间值。forward hook 恰好捕获模块输出。
- **fp32 参考用原始 nn.Module（deep copy）**，不是 Quantized* passthrough 树。它天然是 golden reference，零验证成本。
- **局部误差用现有 observer 系统**，保持不变。

### 改动清单

| 文件 | 改动 |
|------|------|
| `src/scheme/op_config.py` | 新增 `cfg_causes_quantization(cfg)` — 判断 cfg 是否真的会触发量化 |
| `src/session/_session.py` | `analyze()` 重写：用 `cfg_causes_quantization` 过滤、多 batch 累加 signal/error/count、eval_fn 优先 |
| `src/session/_model.py` | `_get_quantized_modules()` 用 `cfg_causes_quantization` 替代 `_is_passthrough` |
| `src/tests/test_session.py` | 新增 7 个测试：`cfg_causes_quantization` 边界条件、true_error 单/多 batch、eval_fn、observer 组合 |

### `cfg_causes_quantization` 逻辑

```python
def cfg_causes_quantization(cfg) -> bool:
    if cfg is None:
        return False
    if cfg == OpQuantConfig():
        return False
    for f in fields(cfg):
        if getattr(cfg, f.name) is not None:
            return True
    return False
```

遍历 OpQuantConfig 的所有 dataclass 字段（storage / input / weight / output / bias / grad_*），只要有一个非 None 就认为会产生量化。

### `analyze(true_error=True)` 新流程

```
1. 确定 quant_names = 所有 cfg_causes_quantization 为 True 的模块名
2. 判断 batch 模式：
   - eval_fn 为 None 且 calib_data 是 list → 逐 batch 循环（fp32 → quant → 累加）
   - 否则 → 单次 forward（eval_fn 优先）
3. 每个 batch：
   a. fp32_model(batch) with hooks → fp32_refs
   b. qmodel(batch) with hooks → quant_outs
   c. accum_signal += ||fp32||², accum_error += ||fp32 - quant||², accum_count += numel
4. QSNR = 10 * log10(Σsignal/Σcount / Σerror/Σcount)
```

`eval_fn` 始终是第一优先级：如果提供，所有 forward pass 都通过 `eval_fn(model, data)` 调用，不会直接调 `model(batch)`。

### 与 observer 的共存

`true_error=True` + observers 同时使用时，observers 包裹整个 quant forward 循环（一个 AnalysisContext 覆盖所有 batch），observers 数据跨 batch 累积，true error 也跨 batch 累积。两者独立产出结果。

---

## 3. 为什么不做原方案

| 原方案 | 问题 | 当前做法 |
|--------|------|---------|
| 双 Quantized* 树（passthrough + active） | 需在 22 个模块实现 passthrough forward + bit-exact 验证矩阵 | 用原始 nn.Module deep copy 作为 fp32 参考（已有，0 改动） |
| 统一 stash `{input, weight, output}` | input QSNR 与上一层 output QSNR 冗余；weight QSNR 是局部的（非累积），可直接公式算 | hook 只抓 output，信息量等价于三 role 但无冗余 |
| Phase 3 内联 op stash | 命名依赖 forward 调用顺序，data-dependent 控制流下断裂 | observer 系统已覆盖内联 op 的局部误差 |
| 模块级 stash 填充 | 需改动 22 个 forward() 方法 | 0 个模块 forward 改动 |

### input/weight QSNR 为什么不抓

- **input QSNR(Layer N)** ≈ **output QSNR(Layer N-1)**。因为 Layer N 的输入就是 Layer N-1 的输出（经过 N-1 的 output 量化）。差别只是 Layer N 的 cfg.input 量化增益，通常很小。所以 input QSNR 是重复信息。
- **weight QSNR**：`||W||² / ||W - quantize(W)||²`，纯局部计算，不依赖 forward pass。可以直接从模块 weight 和 cfg.weight 算出来。

### 为什么不做 passthrough Quantized* 树

原始 nn.Module（`copy.deepcopy(model)`）就是完美的 passthrough 参考：
- 输出与用户原始模型 bit-exact（它是同一个模型）
- 不需要为 22 个 Quantized* 类型实现 passthrough 路径
- 不需要验证「passthrough 模式是否等价于原始 nn.Module」
- 已在 `_QuantSession.__init__` 中创建（`keep_fp32=True`）

---

## 4. 已识别限制

1. **eval_fn 多 batch**：当 eval_fn 提供时，eval_fn 内部可能多次调用 model。当前只做单次 eval_fn 调用，hook 捕获最后一次 model() 的输出。如果 eval_fn 内部有随机性（dropout、数据 shuffle），两次 eval_fn 调用（fp32 vs quant）可能处理不同数据。这是 eval_fn 黑盒的固有限制。

2. **data-dependent normalization**：LayerNorm/RMSNorm 的统计量（mean, var）在 fp32 和量化模型中不同（因为输入不同）。这是正确的端到端行为，但意味着归一化层的 QSNR 同时反映了输入量化和统计量漂移，无法解耦。

3. **weight QSNR 未采集**：如果需要逐层 weight QSNR，可以单独实现为 `QSNR(W, quantize(W, cfg.weight))`，不走 forward。

---

## 5. 测试覆盖

- `test_cfg_causes_quantization_empty` — None / OpQuantConfig() 返回 False
- `test_cfg_causes_quantization_with_scheme` — input/storage/grad_input 非 None 返回 True
- `test_true_error_returns_qsnr_per_layer` — 基本 true_error 产出 QSNR 和 MSE
- `test_true_error_multi_batch_accumulation` — 多 batch 累加，层名一致
- `test_true_error_with_eval_fn` — eval_fn 被调用且优先于 model()
- `test_true_error_with_observers_combined` — true_error + observer 共存
- `test_true_error_excludes_non_quantizing_modules` — 空 cfg 模块不出现
