# Format Study 验证脚本 — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个"先理论推导、后实验验证"的测试脚本，验证 format study 实验框架（核心量化 → 算子+Transform → 完整 Pipeline）的正确性。

**Architecture:** 每个测试用例配一个独立的推导文档（`docs/verification/NNN-name.md`），记录手工推算过程和期望值。验证脚本 (`examples/test_format_study_verification.py`) 引用这些推导结果，用 `torch.equal` 做 bit-exact 比对。固定权重和输入数据（微型张量 W(3×2), x(2×2)），所有值可在纸上手工验算。

**Tech Stack:** PyTorch, src/ 量化框架（FormatBase, QuantScheme, OpQuantConfig, QuantSession）, pytest

**设计文档:** `docs/plans/2026-04-28-format-study-design.md`（format study 实验矩阵）

---

## 固定数据

```python
# Linear(2, 3): in_features=2, out_features=3
W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
b = torch.tensor([0.1, 0.2, 0.3])                          # (3,)
x = torch.tensor([[0.5, -0.25], [1.0, 0.75]])              # (2, 2)

# fp32 reference output:
# y = x @ W^T + b
# y[0] = [0.5*1 + (-0.25)*2, 0.5*3 + (-0.25)*4, 0.5*5 + (-0.25)*6] + [0.1,0.2,0.3]
#      = [0.0, 0.5, 1.0] + [0.1, 0.2, 0.3]
#      = [0.1, 0.7, 1.3]
# y[1] = [1.0*1 + 0.75*2, 1.0*3 + 0.75*4, 1.0*5 + 0.75*6] + [0.1,0.2,0.3]
#      = [2.5, 6.0, 9.5] + [0.1, 0.2, 0.3]
#      = [2.6, 6.2, 9.8]
fp32_y = torch.tensor([[0.1, 0.7, 1.3], [2.6, 6.2, 9.8]])
```

## 已知限制

- per_block(32) 和 MX 格式需要至少 32 个元素沿 block_axis，微型张量不满足
- fp4/nf4 的 per_channel 变体推迟到中等张量扩展

---

### Task 0: 创建目录结构和验证方法说明

**Files:**
- Create: `docs/verification/README.md`

**Step 1: 创建 verification 目录和总览文档**

`docs/verification/README.md`:
- 说明"先推导后验证"方法论
- 固定数据定义
- 推导文档命名规范（`NNN-<test-name>.md`）
- 每个推导文档的结构模板：给定数据 → 手工计算过程 → 期望值 → 代码验证结果

**Step 2: Commit**

```bash
git add docs/verification/README.md
git commit -m "docs(verification): add verification methodology and directory structure"
```

---

### Task 1: int8 per_tensor 推导 + 验证

**Files:**
- Create: `docs/verification/001-int8-per-tensor.md`
- Create: `examples/test_format_study_verification.py`（骨架 + 第一个测试）

**Step 1: 写推导文档**

推导 int8(固定范围 [-127/64, 127/64]) 对 x 和 W 的逐元素量化结果。

```
int8 per_tensor 量化公式:
  val_scaled = x * 64
  val_rounded = sign * floor(|val_scaled| + 0.5)
  val_q = clamp(val_rounded / 64, -1.984375, 1.984375)

对 x = [[0.5, -0.25], [1.0, 0.75]]:
  x[0,0]=0.5:  0.5*64=32 → round(32)=32 → 32/64=0.5 → clamp → 0.5
  x[0,1]=-0.25: -0.25*64=-16 → round(-16)=-16 → -16/64=-0.25 → clamp → -0.25
  x[1,0]=1.0:  1.0*64=64 → round(64)=64 → 64/64=1.0 → clamp → 1.0
  x[1,1]=0.75: 0.75*64=48 → round(48)=48 → 48/64=0.75 → clamp → 0.75
  expected = [[0.5, -0.25], [1.0, 0.75]]

对 W: 同理逐元素计算... (每个元素值恰好都是 1/64 的整数倍，量化无损)
  expected_W = W (无损)
```

**Step 2: 写验证脚本骨架 + test_int8_per_tensor()**

```python
#!/usr/bin/env python3
"""
Format Study Verification Script
==================================
每个测试用例对应 docs/verification/NNN-name.md 中的推导。
先读推导文档理解期望值来源，再运行验证。
"""
import torch
import pytest
from src.formats.int_formats import IntFormat
from src.scheme.granularity import GranularitySpec

# ---- Shared fixtures ----
W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
b = torch.tensor([0.1, 0.2, 0.3])
x = torch.tensor([[0.5, -0.25], [1.0, 0.75]])

def test_int8_per_tensor():
    """验证 int8 per_tensor 量化。推导见 docs/verification/001-int8-per-tensor.md"""
    fmt = IntFormat(bits=8)
    gran = GranularitySpec.per_tensor()

    # x 量化（期望值来自手工推导）
    expected_x = torch.tensor([[0.5, -0.25], [1.0, 0.75]])
    actual_x = fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), f"x mismatch:\n{actual_x}\nvs\n{expected_x}"

    # W 量化
    expected_w = W.clone()
    actual_w = fmt.quantize(W, gran)
    assert torch.equal(actual_w, expected_w), f"W mismatch:\n{actual_w}\nvs\n{expected_w}"
```

**Step 3: 运行验证**

```bash
PYTHONPATH=. python -m pytest examples/test_format_study_verification.py::test_int8_per_tensor -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add docs/verification/001-int8-per-tensor.md examples/test_format_study_verification.py
git commit -m "test(verification): add int8 per_tensor derivation and verification"
```

---

### Task 2: int8 per_channel 推导 + 验证

**Files:**
- Create: `docs/verification/002-int8-per-channel.md`
- Modify: `examples/test_format_study_verification.py`

**Step 1: 写推导文档**

int8 per_channel(axis=0) 对 x 量化：
```
axis=0 → 每列独立计算 amax
col0: max(|0.5|, |1.0|) = 1.0
col1: max(|-0.25|, |0.75|) = 0.75

归一化: x_norm = x / amax
col0: [0.5/1.0, 1.0/1.0] = [0.5, 1.0]
col1: [-0.25/0.75, 0.75/0.75] = [-0.333..., 1.0]

逐元素 int8 量化: *64 → round → /64 → clamp(±1.984375)
col0: [0.5*64=32→32/64=0.5, 1.0*64=64→64/64=1.0]
col1: [-0.333*64=-21.333→round(-21.333)=-21→-21/64=-0.328125,
        1.0*64=64→64/64=1.0]

rescale: x_q * amax
col0: [0.5*1.0, 1.0*1.0] = [0.5, 1.0]
col1: [-0.328125*0.75, 1.0*0.75] = [-0.24609375, 0.75]

expected_x = [[0.5, -0.24609375], [1.0, 0.75]]
```

int8 per_channel(axis=0) 对 W(3×2) 量化...（逐行计算）

**Step 2: 写 test_int8_per_channel()**

**Step 3: 运行验证 → PASS**

**Step 4: Commit**

---

### Task 3: int4 per_tensor 推导 + 验证

**Files:**
- Create: `docs/verification/003-int4-per-tensor.md`
- Modify: `examples/test_format_study_verification.py`

int4 固定范围 [-7/4, 7/4] = [-1.75, 1.75], 步长 1/4 = 0.25.
```
公式: x*4 → round → /4 → clamp(±1.75)
```

---

### Task 4: int4 per_channel 推导 + 验证

**Files:**
- Create: `docs/verification/004-int4-per-channel.md`
- Modify: `examples/test_format_study_verification.py`

---

### Task 5: fp8_e4m3 per_tensor 推导 + 验证

**Files:**
- Create: `docs/verification/005-fp8-e4m3-per-tensor.md`
- Modify: `examples/test_format_study_verification.py`

fp8_e4m3 格式: ebits=4, mbits=3, 浮点表示。需手工推 e4m3 的量化映射。

---

### Task 6: nf4 per_channel 推导 + 验证

**Files:**
- Create: `docs/verification/006-nf4-per-channel.md`
- Modify: `examples/test_format_study_verification.py`

NF4 查表量化：16 个量化等级是非均匀分布的。手工查表。

---

### Task 7: Linear + int8 per_channel 算子验证

**Files:**
- Create: `docs/verification/007-linear-int8-pc-forward.md`
- Modify: `examples/test_format_study_verification.py`

手工推演 `LinearFunction.forward()` 全过程：
```
1. quantize input(x, int8_pc) → 已由 Task 2 推导
2. quantize weight(W, int8_pc) → 已由 Task 2 推导
3. y = F.linear(qx, qw) + b  → 手算矩阵乘法
4. quantize output(y, int8_pc) → 手算
expected_y = ...
```

---

### Task 8: SmoothQuant 等价性验证

**Files:**
- Create: `docs/verification/008-linear-smoothquant.md`
- Modify: `examples/test_format_study_verification.py`

手工计算 SmoothQuant scale，验证 `(x/s) @ (W*s) = x @ W`（数学等价性），再叠加量化验证。

```
s_j = max(|X_j|)^0.5 / max(|W_j|)^0.5  (alpha=0.5)
对 channel 0: max(|0.5|,|1.0|)=1.0, max(|1.0|,|3.0|,|5.0|)=5.0
  s_0 = 1.0^0.5 / 5.0^0.5 = 1.0 / 2.236 = 0.4472...
对 channel 1: ...
```

---

### Task 9: Hadamard 自逆性验证

**Files:**
- Create: `docs/verification/009-linear-hadamard.md`
- Modify: `examples/test_format_study_verification.py`

FWHT(2D, last_dim=2): pad to next_pow2 → 2 已对齐。
```
H = 1/sqrt(2) * [[1, 1], [1, -1]]
x_2d(2,2): H(x) = x @ H (沿 last dim)
手工计算 H(x)，验证 H(H(x)) == x
```

---

### Task 10: Pipeline 校准验证

**Files:**
- Create: `docs/verification/010-pipeline-calibration.md`
- Modify: `examples/test_format_study_verification.py`

手工推演 `QuantSession.calibrate()` 流程：
```
1. quantize_model() 替换 Linear → QuantizedLinear
2. 前向传播: output = QuantizedLinear(x)
3. Hook 捕获 output amax
4. MSEScaleStrategy 网格搜索最优 scale
5. _output_scale buffer 赋值
→ 验证 _output_scale 与手工计算一致
```

---

### Task 11: Pipeline 分析验证

**Files:**
- Create: `docs/verification/011-pipeline-analyze.md`
- Modify: `examples/test_format_study_verification.py`

手工计算 QSNR 和 MSE：
```
QSNR = 10 * log10(var(fp32_output) / var(error))
MSE = mean((fp32_output - quant_output)^2)
→ 与 AnalysisContext.report() 比对
```

---

### Task 12: Pipeline E2E 验证

**Files:**
- Create: `docs/verification/012-pipeline-e2e.md`
- Modify: `examples/test_format_study_verification.py`

手工计算 `session.compare()` 的 E2E 精度差异。

---

### Task 13: 更新 CURRENT.md

**Files:**
- Modify: `docs/status/CURRENT.md`

更新断点状态，记录验证脚本的存在和"先推导后验证"方法论。

---

### Task 14: 全量运行 + 最终检查

**Step 1: 运行全部验证测试**

```bash
PYTHONPATH=. python -m pytest examples/test_format_study_verification.py -v
```

Expected: ALL PASS

**Step 2: 确保现有测试无 regression**

```bash
PYTHONPATH=. python -m pytest src/tests/ -x -q
```

Expected: ALL 1316 tests pass.

**Step 3: Commit**
