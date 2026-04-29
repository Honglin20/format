# Format Study Verification — 理论推导与实验验证

## 方法论：先推导，后验证

每个测试用例遵循严格的 **"先理论推导、后代码验证"** 流程：

1. **推导文档**（`docs/verification/NNN-name.md`）：在纸上手工推算所有中间值和最终期望值
2. **验证代码**（`examples/test_format_study_verification.py`）：运行实验，`torch.equal` bit-exact 比对
3. **结果记录**：推导文档末尾记录验证结果（PASS/FAIL + 实际输出）

禁止"先跑实验再看结果是否合理"——必须先有期望值再跑代码。

## 固定数据

所有测试用例共享同一组固定数据，确保可复现：

```python
# Linear(2, 3): in_features=2, out_features=3
W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)
b = torch.tensor([0.1, 0.2, 0.3])                          # shape (3,)
x = torch.tensor([[0.5, -0.25], [1.0, 0.75]])              # shape (2, 2)

# fp32 reference output: y = x @ W^T + b
# y[0] = [0.5*1 + (-0.25)*2, 0.5*3 + (-0.25)*4, 0.5*5 + (-0.25)*6] + [0.1,0.2,0.3]
#      = [0.0, 0.5, 1.0] + [0.1, 0.2, 0.3]
#      = [0.1, 0.7, 1.3]
# y[1] = [1.0*1 + 0.75*2, 1.0*3 + 0.75*4, 1.0*5 + 0.75*6] + [0.1,0.2,0.3]
#      = [2.5, 6.0, 9.5] + [0.1, 0.2, 0.3]
#      = [2.6, 6.2, 9.8]
fp32_y = torch.tensor([[0.1, 0.7, 1.3], [2.6, 6.2, 9.8]])
```

## 推导文档命名规范

```
docs/verification/NNN-<short-name>.md
```

- `NNN`：三位数字，按验证层级递增
- `<short-name>`：简短描述，如 `int8-per-tensor`、`linear-smoothquant`

## 推导文档模板

```markdown
# NNN: <测试名称>

**对应测试函数**: `test_<name>()`
**验证层级**: Layer 1/2/3（核心量化 / 算子+Transform / Pipeline）

## 给定数据

（列出本测试使用的张量及其值）

## 手工推导

（逐步展示计算过程，包含中间值）

### 步骤 1: ...

### 步骤 2: ...

## 期望值

```python
expected = torch.tensor([...])
```

## 验证结果

- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
- [ ] 实际输出与期望一致 / 差异说明
```

## 验证层级

| 层级 | 范围 | 推导文档编号 |
|------|------|------------|
| Layer 1 | 核心量化：Format.quantize() 逐元素正确性 | 001-006 |
| Layer 2 | 算子 + Transform：Linear 配合 SmoothQuant/Hadamard | 007-009 |
| Layer 3 | 完整 Pipeline：QuantSession calibrate → analyze → compare | 010-012 |

## 已知限制

- 微型张量 (W 3×2, x 2×2) 无法测试 per_block(32) —— block_size 超过张量维度
- fp4 / nf4 的 per_channel 和 per_block 变体推迟到中等张量扩展
- 当前仅覆盖 Linear 算子，Conv2d / Norm / Activation 等后续扩展

## 运行全部验证

```bash
PYTHONPATH=. python -m pytest examples/test_format_study_verification.py -v
```
