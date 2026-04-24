# ADR-004: MxSpecs → QuantScheme 渐进式迁移

**状态**: 已决策（进行中）  
**日期**: 2026-04-24

---

## 背景

`mx/` 库使用 39-key `MxSpecs` dict 作为量化配置。Phase 2 初版的 `src/` 代码虽然引入了 `QuantScheme`，但量化函数（`elemwise.py`、`mx_quantize.py`）仍然依赖 `MxSpecs`，导致两套配置并存。

## 决策

**方向**：完全迁移到 `QuantScheme` 驱动（ADR-001 三轴设计），废弃 `MxSpecs`。

**策略**：渐进式，分 phase 推进，不一次性大爆炸。MxSpecs 在 `mx/` 中作为 legacy 保留，`src/` 代码在 Phase 2 修正后完全不依赖 `MxSpecs`。

---

## MxSpecs 现状分析

`MxSpecs` 是 `dict`，39 个 key 按功能分组：

| 分组 | 代表 key | QuantScheme 对应位置 |
|---|---|---|
| 格式选择 | `bfloat`, `fp`, `bfloat_subnorms` | `format` 字段（FormatBase 对象） |
| 块大小 | `block_size` | `granularity.block_size` |
| 舍入模式 | `round`, `rne`, `sr` | `round_mode` 字段 |
| 输入/权重/梯度控制 | `a_elem_format`, `w_elem_format`, `g_elem_format` | 算子层 `QuantizedLinear.input_scheme`, `weight_scheme` |
| 辅助/调试 | `mx_flush_fp32_subnorms`, `quantize_backprop` | 算子层参数或废弃 |

完整 MxSpecs schema 见 `mx/specs.py:22-61`。

---

## 迁移路线图

### Phase 2 修正（P2F，当前）

**目标**：`src/` 内所有代码不再 import MxSpecs；量化函数签名改为 `QuantScheme` 驱动。

**具体行动**：
- `src/quantize/elemwise.py`：函数签名从 `(x, mx_specs, round_mode)` 改为 `(x, scheme: QuantScheme)`；if-elif 链替换为 `scheme.format.quantize(x, scheme.granularity, scheme.round_mode)`
- `src/quantize/mx_quantize.py`：同上，消除 `mx_specs['block_size']` 等直接访问
- `src/quantize/bfloat_quantize.py`：同上
- `src/quantize/vector.py`：同上
- `src/specs/specs.py`（整个文件）：**移入 `mx/` 兼容层**（见下），或直接删除，`src/` 不再提供此文件

**验证**：
```bash
# 验证 src/ 不依赖 MxSpecs 的 grep 检查（CI 门）
grep -r "MxSpecs\|mx_specs\|from.*specs import" src/ && echo "FAIL: MxSpecs leaked" || echo "OK"
```

### Phase 3 算子层（P3）

**目标**：`src/ops/` 中的算子完全基于 `QuantScheme`，不接触 MxSpecs。

`QuantizedLinear` 示例：
```python
class QuantizedLinear(ObservableMixin, nn.Linear):
    def __init__(self, ..., input_scheme: QuantScheme, weight_scheme: QuantScheme):
        super().__init__(...)
        self.input_scheme = input_scheme
        self.weight_scheme = weight_scheme
    # 不存在任何 MxSpecs 引用
```

等价性对标通过以下适配器完成（仅用于测试，不进主代码）：
```python
# src/tests/_compat.py（仅测试内部使用）
def scheme_from_mx_specs(mx_specs: dict) -> tuple[QuantScheme, QuantScheme]:
    """从旧 MxSpecs dict 构造 (input_scheme, weight_scheme)，仅用于等价性测试。"""
```

### 最终（mx/ 删除时）

当 Phase 3 等价性测试全部通过、`src/ops/` 功能完整后，执行：
```bash
git rm -r mx/
```
`MxSpecs` 随 `mx/` 自然消失，无需专门清理代码。

---

## 等价性保障

迁移过程中，等价性通过已有测试框架持续验证（见 `src/tests/`）：

```
src/tests/
  test_elemwise_equiv.py      # elemwise.py 新旧等价
  test_mx_quantize_equiv.py   # mx_quantize.py 新旧等价
  test_bfloat_quantize_equiv.py
  test_vector_equiv.py
  test_golden_equiv.py        # 黄金参考文件对比
```

迁移后每个文件的测试必须继续全绿，以 `mx.` 公共 API 为 reference（不改 `mx/` 源码）。

---

## 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| 新 QuantScheme API 破坏等价性 | 等价性测试在每个 P2F 子任务后必须全绿 |
| MxSpecs 中某些 key 语义无对应 | 先梳理（见上表），无对应的 key 记录在此文档并决定废弃或保留 |
| 格式分派性能下降（对象 vs dict） | Phase 2 修正后跑 benchmark 对比，diff > 5% 则优化 |

---

## 当前状态

- [x] Phase 1：`FormatBase` + Registry 雏形（`src/formats/`）
- [x] Phase 2 初版：`QuantScheme` 门面（仍依赖 MxSpecs）
- [ ] **Phase 2 修正（当前）**：量化函数改为 QuantScheme 驱动，src/ 零 MxSpecs 依赖
- [ ] Phase 3：算子层，完全基于 QuantScheme
- [ ] 最终：`mx/` 删除
