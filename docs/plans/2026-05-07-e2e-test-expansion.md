# E2E 模型比对测试扩展计划

**日期**: 2026-05-07
**状态**: 待确认

---

## 1. 审视结论

### 1.1 已有测试

| 层 | 文件 | 覆盖 | 格式 | 问题 |
|---|---|---|---|---|
| 全模型 E2E | `verify_layer_equiv.py` | 21 模块 + 10 inline | 仅 `int4` | 脚本非 pytest，单格式 |
| 全模型 E2E | `test_e2e_small_model.py` | Linear, Conv2d | 仅 `bfloat=16` | 算子少，无 MX 格式 |
| 单算子 equiv | `test_ops_equiv_matmul.py` | Linear/MatMul/BMM | bf16 + 5 MX 组合 | 格式最多但不全 |
| 单算子 equiv | `test_ops_equiv_conv.py` | Conv2d | 仅 bf16 | **无 MX 格式** |
| 单算子 equiv | `test_ops_equiv_conv_transpose.py` | ConvTranspose2d | 仅 bf16 | **无 MX 格式** |
| 单算子 equiv | `test_ops_equiv_norm.py` | BN/LN/GN/RMSNorm | bf16/bf10/ste | elemwise only，合理 |
| 单算子 equiv | `test_ops_equiv_activations.py` | 7 激活 | bf16/bf10/ste | elemwise only，合理 |
| 单算子 equiv | `test_ops_equiv_elemwise.py` | 12 simd | bf16/bf10/ste | elemwise only，合理 |
| 单算子 equiv | `test_ops_equiv_softmax.py` | Softmax | bf16/bf10/exp2/ste | elemwise only，合理 |
| 单算子 equiv | `test_ops_equiv_pooling.py` | AdaptiveAvgPool2d | bf16/bf10/ste | elemwise only，合理 |

### 1.2 算子分类

| 类别 | 算子 | MX block 格式适用？ | 需扩展 |
|---|---|---|---|
| **A: GEMM** | Linear, MatMul, BMM | ✅ | 补充缺失的 MX 格式 |
| **B: 卷积** | Conv1d/2d/3d, ConvTranspose1d/2d/3d | ✅ | **从零建立 MX 格式覆盖** |
| **C: Norm** | BN1d/2d/3d, LN, GN, RMSNorm | ❌ | 当前基本充分 |
| **D: 激活** | Sigmoid/Tanh/ReLU/ReLU6/LeakyReLU/SiLU/GELU | ❌ | 当前基本充分 |
| **E: Softmax** | Softmax | ❌ | 当前基本充分 |
| **F: Pool** | AdaptiveAvgPool2d | ❌ | 当前基本充分 |
| **G: SIMD** | add/sub/mul/div/exp/log/matmul/mm/bmm/linear | ❌ | 当前基本充分 |

### 1.3 mx 可用的全部格式

**MX block 格式**（`w_elem_format` / `a_elem_format`）：`fp8_e5m2`, `fp8_e4m3`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1`, `int8`, `int4`, `int2`（8 种）

**Storage 格式**（`bfloat`）：`bfloat=16`（标准）、`bfloat=10`（极端）、无 storage

**量化反向传播**：`quantize_backprop=True`（默认）、`False`（STE）

**Block size**：`32`（标准）、其他值

---

## 2. 测试扩展任务

### 2.1 Task 1: 统一格式参数化矩阵（`src/tests/_formats.py`）

**目的**：消除各测试文件重复定义 MX_SPECS_CONFIGS，提供一个可复用的格式参数化工厂。

```python
# 新文件 src/tests/_formats.py

# MX 全部 8 种格式
ALL_MX_ELEM_FORMATS = [
    "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3",
    "fp4_e2m1", "int8", "int4", "int2",
]

# 为 GEMM/Conv 类算子生成参数化：storage × MX format × QBP
def build_mx_specs_params(
    include_bfloat: bool = True,
    include_no_storage: bool = True,
    include_no_qbp: bool = False,
    formats: list = ALL_MX_ELEM_FORMATS,
) -> list:
    """生成 pytest.param 列表，覆盖所有 MX 格式组合。"""
    ...

# 为 elemwise-only 类算子生成参数化
def build_elemwise_specs_params() -> list:
    """bf16, bf10, fp16, ste 变体。"""
    ...

# 轻量 Smoke 子集（快速迭代用）
SMOKE_MX_FORMATS = ["fp8_e4m3", "int8", "int4"]
```

### 2.2 Task 2: 扩展 Conv 类算子的 MX 格式测试

修改 `test_ops_equiv_conv.py` 和 `test_ops_equiv_conv_transpose.py`：
- 添加完整 MX_SPECS_CONFIGS（与 matmul 对齐）
- 覆盖 Conv1d, Conv2d, Conv3d fwd + bwd
- 覆盖 ConvTranspose1d, ConvTranspose2d, ConvTranspose3d fwd + bwd

### 2.3 Task 3: 补齐 MatMul 类算子的缺失 MX 格式

`test_ops_equiv_matmul.py` 当前有 6 种配置，补齐到全部 8 种 MX 格式 × 关键组合。
添加 smoke 标记区分全量测试和快速验证。

### 2.4 Task 4: 全模型 E2E 参数化测试

创建 `src/tests/test_e2e_all_ops.py`：
- 复用 `AllOpsModel`（从 `verify_layer_equiv.py` 提取为共享 fixture）
- 参数化驱动：`@pytest.mark.parametrize("mx_specs", ALL_MX_SPECS_CONFIGS)`
- 对比 `quantize_model(model, cfg)` vs MX reference chain
- 包含 fwd-only（快速）和 fwd+bwd（完整）两种模式
- Smoke 标记：默认 `-m "not slow"` 跑快速子集

### 2.5 Task 5: verify_layer_equiv.py → pytest 迁移

将 `verify_layer_equiv.py` 的验证逻辑吸收到 `test_e2e_all_ops.py`：
- 保留 layer-by-layer 断言
- 保留 config match 验证
- 删除手工脚本（或降级为 `tools/` 下的可选调试工具）

### 2.6 Task 6: 更新 CI/快速测试门

- 确认 `pytest src/tests/ -q -m "not slow"` 覆盖 smoke 子集
- 全量 MX 格式测试通过 `-m "slow"` 标记按需运行
- 更新 `CLAUDE.md` 中的测试命令引用

---

## 3. 测试参数矩阵

### 3.1 GEMM 类（Linear/MatMul/BMM）— 预期 ~40 配置

| storage | MX format | QBP | 备注 |
|---|---|---|---|
| bf16 | 8 种 MX 全部 | True | 主力配置 |
| bf16 | smoke 3 种 | False | STE 变体 |
| none | 8 种 MX 全部 | True | 纯 MX 无 elemwise |
| bf10 | smoke 3 种 | True | 极端 bfloat |
| none | (none) | False | passthrough |

### 3.2 卷积类（Conv/ConvTranspose × 3D）— 预期 ~25 配置

| storage | MX format | QBP | 备注 |
|---|---|---|---|
| bf16 | smoke 3 种 | True | 主力配置（覆盖 1d/2d/3d） |
| bf16 | 全部 8 种 | True | 全量（仅 2d） |
| none | smoke 3 种 | True | 纯 MX |
| none | (none) | False | passthrough |

### 3.3 全模型 E2E — 预期 ~15 配置

| storage | MX format | QBP | 模式 |
|---|---|---|---|
| bf16 | smoke 3 种 | False | fwd-only（快速） |
| bf16 | 全部 8 种 | False | fwd-only（全量） |
| bf16 | smoke 3 种 | True | fwd+bwd（慢） |
| none | smoke 3 种 | False | 纯 MX fwd-only |

---

## 4. 执行顺序

1. **Task 1** — `_formats.py` 参数化工厂（基础设施）
2. **Task 2** — Conv 类 MX 格式测试（最大缺口）
3. **Task 3** — MatMul 类补齐缺失格式
4. **Task 4** — 全模型 E2E 参数化测试
5. **Task 5** — verify_layer_equiv.py 迁移/删除
6. **Task 6** — CI/文档更新

每个 Task 完成后：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q`

---

## 5. 风险与注意事项

- **Conv 算子的 MX block_axis 不同**：input block_axis=1, weight block_axis=1（Conv）或 0（ConvTranspose），已在 `_compat.py` 中处理
- **ConvTranspose1d/3d 无 MX reference**：mx 只有 `ConvTranspose2d` 模块，1d/3d 只能用 elemwise 量化
- **测试时间**：全量 MX 格式 × 全部 Conv 维度 × bwd 可能很慢，用 smoke 标记区分
- **BF16 tensor core 路径**：部分低精度 MX 格式（int4/int2/fp6/fp4）在 mx 内部触发 BF16 matmul，src 需对齐此行为
