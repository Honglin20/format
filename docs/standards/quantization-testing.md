# 量化测试用例编写规范

## 核心原则

### 1. 数学正确性优先

等价性测试必须 bit-exact（`torch.equal`），不允许 `allclose`。每个测试用例必须先有手工推导的期望值，再写代码验证。

### 2. 推导前置

写测试用例前：
1. **说明格式的量化流程和原理** — 公式、scale 计算方式、round_mode 行为、block 分组逻辑
2. **手工推导** — 逐步展示所有中间值和最终期望值
3. **代码验证** — `torch.equal` 比对推导结果

推导文档写入 `docs/verification/NNN-*.md`。流程细节见 `principles/math-verification.md`。

## 形状覆盖要求

每个量化测试至少覆盖：

| 维度 | 示例形状 | 验证点 |
|------|---------|--------|
| 1D | `(N,)` | 向量量化，边界元素 |
| 2D | `(M, N)` | 矩阵，常见 Linear weight/input |
| 3D+ | `(B, C, H, W)` 或 `(B, S, E)` | batch + channel + spatial |

## 不规则张量覆盖

| 场景 | 示例 | 验证点 |
|------|------|--------|
| 奇数维度 | `(7, 13)` | 非 2 的幂不引入 bias |
| 质数维度 | `(17, 23)` | 最极端的非整除情况 |
| 非对称形状 | `(3, 128)` | 一维规则、一维不规则 |

## Block 格式专项

block 格式（如 `per_block(32)`）需要额外覆盖：

| 场景 | 示例 | 验证点 |
|------|------|--------|
| 整除 | `tensor(8, 64)`, `block_size=32`, `block_axis=-1` | 每 block 独立 scale |
| 不整除 | `tensor(8, 70)`, `block_size=32`, `block_axis=-1` | 最后 block 只有 6 个元素 |
| 退化（block > 维度） | `tensor(3, 5)`, `block_size=32`, `block_axis=-1` | 整行作为一个 block |
| 不同 block_axis | `tensor(4, 8, 16)`, `block_axis=1` | 沿非最后维度切片 |

## 测试用例模板

```markdown
## 格式原理
（量化公式、scale 计算方式、round_mode 行为、block 分组逻辑）

## 给定数据
W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x = torch.tensor([[0.5, -0.25]])

## 手工推导

### per_tensor, int8
scale = max(|W|) / 127 = 4.0 / 127 ≈ 0.031496
W_q = round(W / scale) = round([[31.75, 63.5], [95.25, 127.0]]) = [[32, 63], [95, 127]]

## 期望值
expected = torch.tensor([[32, 63], [95, 127]], dtype=torch.int8)

## 验证
assert torch.equal(result, expected)
```

## 测试命名

用名字表达行为和场景，不是表达错误类型：

```python
# ✓ 好的命名
def test_int8_per_block_32_divisible():
def test_int8_per_block_32_partial_last_block():
def test_int8_per_block_1d_degenerate():

# ✗ 差的命名
def test_per_block_edge():
def test_quantize_2():
```

## 负面测试

每个量化函数的每个 `raise` 点至少一条 `pytest.raises` + `match=`。

## 测试层级：回归高层 API

量化测试同样受 **测试层级原则** 约束：最终提交的测试必须通过 `Session` / `Study` / `QuantConfig` / `resolve_config` 等高层用户接口调用。底层 API（`FormatBase.quantize()`、`_quantize_elemwise_core` 等）仅限开发调试，不允许留在最终测试代码中。

详见 [`principles/testing-layer.md`](../principles/testing-layer.md)。

## 端到端回归覆盖（新增功能必执行）

新增任何涉及 `quantize()`、`FormatBase`、`QuantScheme`、`Session` 的功能（如新 granularity mode、新 format、group_sparse、outlier_sparse、transform），必须通过**端到端集成测试**验证，不允许仅依赖单元测试。

### 覆盖维度清单

| 维度 | 覆盖点 | 最小要求 |
|------|--------|---------|
| **Granularity** | PER_TENSOR, PER_CHANNEL, PER_BLOCK, BANK | 每种 mode 至少 1 条 Session/Study API 测试 |
| **Ratio 边界** | 0.01（最小比）、0.5（中等）、1.0（全 H） | 至少 3 个边界值 |
| **形状多样性** | 1D（bias/norm weight）、2D（Linear weight）、4D（Conv weight） | 每种至少 1 条 |
| **最小通道数** | C=2 或 num_banks=1（退化场景） | 至少 1 条 |
| **Format 组合** | int+int、int+float、float+int | 至少 3 种组合 |
| **Transform** | hadamard、smoothquant、prescale | 每种与 group_sparse 交互至少 1 条 |
| **高层 API** | Session.run()、Study.run() | 两种 API 各至少 1 条 |
| **全流程** | eval_fn + eval_data + observer outputs (qsnr/distribution/histogram) | 至少 1 条完整流程 |
| **Scale 安全** | 传递 shape 不匹配的 scale（原始 bug 触发场景） | 直接 quantize() 安全网测试 |
| **Weight-only** | weight_only=True | 至少 1 条 |
| **quantize_nonlinear** | True / False | 两种各至少 1 条 |

### 测试文件命名

```
src/tests/test_<feature>_e2e.py  — 端到端集成测试（Session / Study API）
src/tests/test_<feature>.py      — 单元测试（QuantScheme / FormatBase 直接调用）
```

### 回归门（bug 修复后必须执行）

修复涉及量化路径的 bug 后，必须：
1. 编写直接触发该 bug 的失败测试（使用高层 API）
2. 确认修复后测试通过
3. 在 E2E 测试文件中添加覆盖该 bug 场景的测试用例
4. 运行完整 E2E 回归门：`PYTHONPATH=. python scripts/mnist_hadamard_study.py` + `python scripts/transformer_agnews_study.py`
