# Current Task

**Task ID**: P3.2（Conv 家族）— 全部完成
**Plan**: `docs/plans/2026-04-24-phase3.md`
**Branch**: `feature/refactor-src`

---

## 本次会话完成的工作

### P3.2 TransposeConv（已完成）
- 实现 `ConvTransposeFunction` + `QuantizedConvTranspose{1,2,3}d`
- 新增 `op_type="conv_transpose"` 适配器路径（`_compat.py`）
- 22 bit-exact 等价性测试全绿
- 关键差异（vs Conv）：
  - Forward: weight MX axis=0（Conv 用 axis=1）
  - Backward grad_weight: `_conv_weight(grad_output, weight.shape, input)` — 参数顺序交换
  - Backward grad_input: 用 `F.conv2d`（不是 conv_transpose），weight MX axis=1（Conv 用 axis=0）

### P3.2 全部子任务汇总
- ConvFunction + QuantizedConv{1,2,3}d：22 tests
- ConvTransposeFunction + QuantizedConvTranspose{1,2,3}d：22 tests
- **P3.2 合计 44 tests，全部 bit-exact**

---

## 未完成的工作

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
4. 从 P3.3 Norm 家族 继续

---

## 下一步（具体动作）

读 `mx/batchnorm.py`、`mx/layernorm.py`、`mx/groupnorm.py`，了解 Norm 家族的 forward/backward 量化模式，然后实现 P3.3

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-phase3.md`（全文）— Phase 3 子阶段定义与验收门
2. `docs/architecture/005-op-quant-config.md`（全文）— OpQuantConfig 设计
3. `src/ops/conv.py`（全文）— P3.2 Conv + ConvTranspose 实现（forward 保存 post-elemwise 模式参考）
4. `src/tests/_compat.py`（全文）— op_config_from_mx_specs 适配器（含 linear/matmul/conv/conv_transpose 路径）
5. `src/scheme/op_config.py`（全文）— OpQuantConfig 数据类

---

## 关键经验记录

1. **Conv vs ConvTranspose MX axis 差异**：Conv forward weight MX axis=1（out_channels），ConvTranspose forward weight MX axis=0（in_channels）。这是因为两者的 weight 布局不同：Conv 是 (out_c, in_c, kH, kW)，ConvTranspose 是 (in_c, out_c, kH, kW)。
2. **ConvTranspose backward grad_input 用 F.conv2d**：mx/transpose_convolution.py 的 grad_input 计算使用 `torch.conv2d`（常规卷积），不是 conv_transpose。这是数学上正确的：transpose conv 的反向传播等价于常规 conv。
3. **_conv_weight 参数顺序**：Conv 的 grad_weight 用 `_conv_weight(input, weight.shape, grad_output)`，ConvTranspose 用 `_conv_weight(grad_output, weight.shape, input)` — 参数 1 和 3 交换。
4. **ConvTranspose backward weight MX axis=1**：grad_input 中 weight MX 沿 axis=1（out_channels/groups），而 Conv 用 axis=0。这是因为 conv_transpose 的 grad_input 需要沿 output channel 维度归约。

---

## 验收门

```bash
pytest src/tests/ -x -q    # 575 tests 全绿
grep -rn "from.*mx" src/ops/ src/analysis/ src/mapping/   # 应无命中
```
