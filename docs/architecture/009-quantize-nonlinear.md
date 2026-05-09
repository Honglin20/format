# ADR-009: `quantize_nonlinear` — 非线性算子统一量化策略

**状态**: 已决策（2026-05-08）
**日期**: 2026-05-08

---

## 背景与问题

`OpQuantConfig`（ADR-005）定义了两级量化模型：

| 级别 | 类型 | 粒度 | 作用域 |
|------|------|------|--------|
| **storage** | elemwise 存储精度 | per-tensor | 所有 tensor 统一 |
| **compute** | MX 计算量化 | per-block | matmul-family operands |

当前 `quantize_model()` 通过 `_non_matmul_cfg()` / `_activation_cfg()` 对非线性算子**保护性剥离** compute 级别：

```python
# src/session/_model.py:61-82
def _non_matmul_cfg(cfg):
    if cfg.storage is not None:
        return OpQuantConfig(storage=cfg.storage, ...)  # 剥离 input/weight per_block
    if _is_mx_compute(cfg.input) or _is_mx_compute(cfg.weight):
        return OpQuantConfig()  # 空 config, 完全不量化
    return cfg
```

这导致 `quantize_nonlinear` flag 的 `True` / `False` 在行为上完全相同——两个模式对 nonlinear ops 都只施加 storage 级别量化，per-block MX compute 被丢弃。

用户需要一个「激进量化」模式：**所有算子都遵循相同的两级量化规则，matmul 和 nonlinear 没有特殊性**。

---

## 决策：入口 operand 对齐，内部不变

### 核心原则

Matmul 的 forward 是单次原子操作（fp32 accumulation），中间没有任何 re-quantize：

```
matmul:  input  → storage → per_block ─┐
         weight → storage → per_block ─┤
                                       ├─ matmul (fp32 accumulate) → output → storage
```

以 **LayerNorm** 为 nonlinear 代表，对 `quantize_nonlinear=True` 的行为规范：

```
norm:    input  → storage → per_block ─┐
         weight → storage → per_block ─┤
         bias   → storage → per_block ─┤
                                       ├─ vec_ops chain (每步过 storage) → output → storage
```

**差异只在一处：operand 入口从 storage-only 升级为 storage + per_block。内部 vec_ops 和 backward 保持 False 的逻辑不变。**

### 完整行为矩阵

| Stage | `False`（当前 = MX 行为） | `True`（对齐 matmul） |
|-------|--------------------------|----------------------|
| **Norm input** | `storage` | `storage` → `input`（per_block） |
| **Norm weight** | `storage` | `storage` → `weight`（per_block） |
| **Norm bias** | `storage` | `storage` → `bias`（per_block） |
| **Norm 中间 vec_ops** | `inner_scheme = storage` | `inner_scheme = storage`（**与 False 相同**） |
| **Norm output** | `storage` | `storage`（**与 False 相同**） |
| **Norm backward** | `storage` only | `storage` only（**与 False 相同**） |
| **Activation input** | `storage` | `storage` → `input`（per_block） |
| **Activation 中间 vec_ops** | `inner_scheme = storage` | `inner_scheme = storage`（**与 False 相同**） |
| **Activation output** | `storage` | `storage`（**与 False 相同**） |
| **Activation backward** | `storage` only | `storage` only（**与 False 相同**） |

### 为什么中间 vec_ops 不升级到 per_block

1. **与 matmul 对齐**：Matmul 的内部累加是 fp32，没有 re-quantize；Norm 的 vec_ops chain 相当于 matmul 的内部计算，应保持一致的精度语义。
2. **硬件现实**：Vector 引擎（CUDA Core / Scalar Unit）处理 pointwise 操作，没有 block-wise scale 硬体支援。Per-block 量化对无 reduce 维的 pointwise 操作无 outlier 隔离收益。
3. **累积效应可控**：如果每个 vec_op 都做 per_block re-quantize，6-8 步串聯的累积误差会远超 matmul 的单步量化，且这不是真实硬体行为。

### 为什么 backward 不升級

Backward 的中间梯度通常分佈更均匀，per-block 的收益不明顯。Matmul 的 backward 也只对 input/weight operand 做 per_block re-quantize，中间梯度保持 storage 级别。统一策略保持一致。

---

## 实现路径

### 变更点

| 函数 | 变更 |
|------|------|
| `_replace_module()` | 接收 `quantize_nonlinear`，传给 `_make_*` 函数 |
| `_make_ln/gn/bn/rms_norm` | `True` 时不同：`cfg` 保留原始 `OpQuantConfig`（含 `input`/`weight` per_block），不再通过 `_non_matmul_cfg` 剥离 |
| `_make_gelu/sigmoid/relu/...` | `True` 时不同：`cfg` 保留原始 `OpQuantConfig`（含 `input` per_block），不再通过 `_activation_cfg` 剥离 |
| `_norm_inner_scheme` | `True` 时：返回 `cfg.storage`（与 `False` 相同）；不返回 `cfg.input` per_block |
| `QuantizedNorm.forward()` | 入口对 input/weight/bias 顺序施加 `cfg.storage` → `cfg.input`/`cfg.weight`/`cfg.bias`（per_block） |
| `QuantizedActivation.forward()` | 入口对 input 施加 `cfg.storage` → `cfg.input`（per_block） |

### 不變更

- Forward 中间 vec_ops 的 `inner_scheme`：始終 `storage`（有 storage）或 `None`（无 storage）
- Backward 的 `quantize_backprop`：始終由 storage 决定（`norm_cfg.is_training`）
- Output exit 量化：始終 `storage` only
- Matmul-family：行為完全不變

### 与 `False` 等效的场景

当 config 没有 per-block MX compute（纯 storage、或纯 per-tensor compat），`True` 和 `False` 在 operand 入口产生相同结果——因为 `cfg.input`/`cfg.weight` 为 `None`，额外的 `quantize(x, cfg.input)` 是 passthrough。两者在位精确上等价。

---

## 验证策略

`quantize_nonlinear=True` 下 MX 库不作为对照标准（MX 不对 nonlinear 施加 per-block 量化）：

| 层 | 方法 | 说明 |
|----|------|------|
| **数学正确性** | `torch.autograd.gradcheck` | 对每个 autograd Function 验证 backward 公式是 forward 的正确导数 |
| **属性测试** | 幂等性 / 有界性 / 无 NaN | 独立于参考实现的不变式验证 |
| **内部一致性** | `True` vs `False` 对比 | 无 per_block 时二者必须 bit-exact 等价；有 per_block 时 SNR 在预期范围内 |
| **fp32 基线** | quant vs fp32 SNR | storage=0 时量化输出应接近 fp32 |

---

## 判断标准

- [ ] `_replace_module()` 根据 `quantize_nonlinear` 控制是否剥离 compute 字段
- [ ] Norm/Activation 的 `forward()` 入口对 operands 施加 `storage → compute` 两级量化
- [ ] 中间 vec_ops 和 backward 的 `inner_scheme`/`quantize_backprop` 与 `False` 行为一致
- [ ] 无 per_block compute 时 `True` ≡ `False`（bit-exact）
- [ ] 全量测试通过
