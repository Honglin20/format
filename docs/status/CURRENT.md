# Current Task

**Task ID**: P3.3（Norm 家族）— 全部完成
**Plan**: `docs/plans/2026-04-24-phase3.md`
**Branch**: `feature/refactor-src`

---

## 本次会话完成的工作

### P3.3 Norm 家族（已完成）
- 实现 `src/ops/vec_ops.py`：量化向量原语（vec_add/sub/mul/div/recip/sqrt/reduce_sum/reduce_mean），scheme=None 时 passthrough
- 实现 `src/ops/norm.py`：BatchNorm/LayerNorm/GroupNorm/RMSNorm 四个 autograd.Function + 模块
- 新增 `norm_config_from_mx_specs()` 适配器（返回 `(OpQuantConfig, inner_scheme)` 元组）
- 32 bit-exact 等价性测试全绿
- 关键设计决策：
  - Norm 算子只有 elemwise 量化（无 MX block），需要 `inner_scheme` 驱动中间步骤的 vec_ops
  - `OpQuantConfig` 仅用于入口/出口量化（input/weight/bias/grad_output），出口量化为空
  - RMSNorm backward 的最后一步 `vec_sub(dx1, dx_norm3)` 在 mx 中没有量化（无 mx_specs 参数）

### P3.3 之前的会话完成的工作
- P3.0: P2F-7 defect closure
- P3.1-a~e: OpQuantConfig + ObservableMixin + Linear/MatMul/BMM
- P3.2: Conv + ConvTranspose（44 tests）

---

## 未完成的工作

### P3.4：激活 / Softmax / AdaptiveAvgPool
- mx/activations.py, mx/softmax.py, mx/adaptive_avg_pooling.py

### P3.5：Elementwise / SIMD / Vector ops
- mx/elemwise_ops.py, mx/simd_ops.py, mx/vector_ops.py

### P3.6：mapping + e2e test
- src/mapping/quantize_model.py + 端到端 small model bit-exact 验证

---

## 下一步（具体动作）

读 `mx/activations.py`、`mx/softmax.py`、`mx/adaptive_avg_pooling.py`，了解激活/Softmax/Pool 家族的 forward/backward 量化模式，然后实现 P3.4

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-phase3.md`（全文）— Phase 3 子阶段定义与验收门
2. `docs/architecture/005-op-quant-config.md`（全文）— OpQuantConfig 设计
3. `src/ops/norm.py`（全文）— P3.3 Norm 实现（inner_scheme 模式参考）
4. `src/ops/vec_ops.py`（全文）— 量化向量原语（Norm/Activation/Softmax 共用）
5. `src/tests/_compat.py`（全文）— op_config_from_mx_specs + norm_config_from_mx_specs 适配器

---

## 关键经验记录

1. **Norm 算子的 inner_scheme 设计**：Norm 算子（BN/LN/GN/RMSNorm）的每个中间步骤都量化（via vec_ops），但 mx 所有 vec_ops 都使用相同的 elemwise scheme（`mx_specs['round']`）。因此设计为 `OpQuantConfig`（入口/出口量化）+ `inner_scheme`（中间步骤量化）的二元组。`norm_config_from_mx_specs()` 返回 `(OpQuantConfig, inner_scheme)`。
2. **Norm 的 OpQuantConfig 结构**：入口用 inner_scheme（input/weight/bias/grad_output 各一条 pipeline），出口为空（output/grad_input/grad_weight/grad_bias = ()）。这与 matmul 家族的 OpQuantConfig 用法不同。
3. **vec_ops scheme=None 必须支持**：当 `quantize_backprop=False` 时，mx 的 backward vec_ops 变成 passthrough（bfloat=0, fp=0），对应 `inner_scheme=None`。`vec_ops.py` 的 `_q()` 函数处理此情况。
4. **RMSNorm backward 最后一步不量化**：mx 的 `RMSNormFunction.backward` 中 `vec_sub(dx1, dx_norm3)` 没有传 `mx_specs`，所以最终减法是 fp32。这与 LayerNorm/BatchNorm/GroupNorm 的行为不同。
5. **Norm forward 保存中间张量**：与 Conv/Linear 不同，Norm 必须保存中间计算结果（x_shift, x_norm, x_std_inv/x_vare），因为 backward 需要它们。`cfg.is_training` 只决定保存量化权重还是原始权重。

---

## 验收门

```bash
pytest src/tests/ -x -q    # 607 tests 全绿
grep -rn "from.*mx" src/ops/ src/analysis/ src/mapping/   # 应无命中
```
