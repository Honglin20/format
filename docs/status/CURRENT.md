# Current Task

**Task ID**: P3.2（Conv 家族）— P3.2 ConvFunction 已完成，TransposeConv 未做
**Plan**: `docs/plans/2026-04-24-phase3.md`
**Branch**: `feature/refactor-src`（ahead of origin by 13 commits，待推送）

---

## 本次会话完成的工作

### P2F-7（已完成）
- commit `bcf4031` + `04fb902`：granularity 类型 guard、channel_axis 越界断言、负面测试

### P3.1-a（已完成）
- commit `e1e6800` + `012eea5`：OpQuantConfig 冻结数据类 + 95 测试 + review 修复

### P3.1-b（已完成）
- commit `8c02240` + `7661dc8`：ObservableMixin + QuantEvent + ObserverBase + SliceAwareObserver + iter_slices + review 修复（C1 _observers 实例隔离、C2 QuantEvent 验证、M1 iter_slices 越界、M3 DYNAMIC_GROUP block_size）

### P3.1-c（已完成）
- commit `87db188` + `73feb9f`：op_config_from_mx_specs Linear 适配器 + block_axis/output pipeline/_bp key 修复 + 9 测试

### P3.1-d（已完成）
- commit `945d817`：QuantizedLinear + LinearFunction，42 equivalence tests bit-exact
- 关键发现：forward 保存 post-elemwise 张量（非 post-MX），backward gemm pipeline 只含 MX scheme（无 double elemwise）
- _compat 修复：_bp keys 在 apply_mx_specs 后为 None，不是 fallback 到 forward format

### P3.1-e（已完成）
- commit `ac6e02f`：quantized_matmul + quantized_bmm，68 total equivalence tests bit-exact
- _compat 新增 op_type="matmul"，forward in2 MX axis=-2，backward axes 不同于 linear

### P3.2 ConvFunction（已完成）
- commit `e1af119`：ConvFunction + QuantizedConv{1,2,3}d，22 equivalence tests bit-exact
- _compat 新增 op_type="conv"，forward MX axis=1（channel dim），backward 使用 a_elem_format/w_elem_format（非 _bp）
- 关键发现：mx/conv 的 backward 直接用 a_elem_format（不是 _bp 变体），与 linear 行为不同

---

## 未完成的工作

### P3.2 剩余：TransposeConv
- mx/transpose_convolution.py 的 ConvTranspose2dFunction 需要单独实现
- forward: input MX axis=1, weight MX axis=0（注意：与 Conv 的 weight axis=1 不同）
- backward: 使用 torch.conv2d (不是 conv_transpose) 计算 grad_input
- 需要 op_type="conv_transpose" 适配器路径

### P3.3：Norm 家族
- BatchNorm / LayerNorm / GroupNorm
- mx/batchnorm.py, mx/layernorm.py, mx/groupnorm.py

### P3.4：激活 / Softmax / AdaptiveAvgPool
- mx/activations.py, mx/softmax.py, mx/adaptive_avg_pooling.py

### P3.5：Elementwise / SIMD / Vector ops
- mx/elemwise_ops.py, mx/simd_ops.py, mx/vector_ops.py

### P3.6：mapping + e2e test
- src/mapping/quantize_model.py + 端到端 small model bit-exact 验证

---

## 新终端续接流程

1. 读 `CLAUDE.md`
2. 读本文件（`docs/status/CURRENT.md`）
3. 读下方的"断点续传必读文件"
4. 从 P3.2 TransposeConv 或 P3.3 Norm 继续

---

## 下一步（具体动作）

**选项 A**：继续 P3.2 TransposeConv — 读 `mx/transpose_convolution.py`，实现 ConvTransposeFunction + op_type="conv_transpose" 适配器

**选项 B**：跳过 TransposeConv 先做 P3.3 Norm 家族 — TransposeConv 使用较少，可以后补

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-phase3.md`（全文）— Phase 3 子阶段定义与验收门
2. `docs/architecture/005-op-quant-config.md`（全文）— OpQuantConfig 设计
3. `src/ops/conv.py`（全文）— P3.2 Conv 实现（forward 保存 post-elemwise 模式参考）
4. `src/tests/_compat.py`（全文）— op_config_from_mx_specs 适配器（含 linear/matmul/conv 路径）
5. `src/scheme/op_config.py`（全文）— OpQuantConfig 数据类

---

## 关键经验记录

1. **_bp key 行为**：`apply_mx_specs` 不会把 a_elem_format 填到 a_elem_format_bp。Linear 的 backward 用 _bp keys（`finalize_mx_specs` 才会填充），但 Conv 的 backward 直接用 a_elem_format/w_elem_format。适配器需按 op_type 区分。
2. **Forward 保存策略**：所有算子 forward 都保存 post-elemwise（pre-MX）张量用于 backward，匹配 mx 的 bf_in/bf_weight 约定。
3. **Conv 的 MX axis**：Conv forward 两个输入都用 axis=1（channel dim），与 Linear/MatMul 的 axis=-1/-2 不同。
4. **Review agent 限制**：subagent 会触发 API rate limit，inline self-review 是可靠的后备方案。

---

## 验收门

```bash
pytest src/tests/ -x -q    # 553 tests 全绿
grep -rn "from.*mx" src/ops/ src/analysis/ src/mapping/   # 应无命中
```
