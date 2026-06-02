# P2: compute_sparse_mask() 独立函数 — 实施计划

**依赖**: ADR-012 Section 2
**前置**: P1 BANK 粒度（已完成）

---

## 目标

封装 `compute_sparse_mask()` 独立函数：输入校准数据 + format + granularity + outlier_ratio → 输出固定 mask（per-element boolean tensor）。

## 算法

```
Step 1 — Per-sample top-k (在 granularity group 内部):
  For each sample s in x_calib (shape (S, D1, D2, ...)):
    For each granularity group g in x_s:
      k_g = max(1, outlier_ratio * group_size)
      _, top_indices = topk(abs(x_s[g]).flatten(), k_g)
      mask_s[g] = scatter_(top_indices, True)
  → mask_all shape: (S, D1, D2, ...)

Step 2 — Cross-sample voting:
  mask_avg = mask_all.float().mean(dim=0)  # shape (D1, D2, ...), values ∈ [0, 1]

Step 3 — Final mask selection:
  k_total = max(1, outlier_ratio * total_elements_per_sample)
  _, top_positions = topk(mask_avg.flatten(), k_total)
  final_mask = scatter_(top_positions, True)  # shape (D1, D2, ...)
```

## 函数签名

```python
# src/quantize/_sparse_mask.py

def compute_sparse_mask(
    x_calib: torch.Tensor,
    fmt: FormatBase,
    granularity: GranularitySpec,
    outlier_ratio: float,
) -> torch.Tensor:
```

## Tasks

| Task | 内容 | 文件 |
|------|------|------|
| 1 | 数学推导文档 | `docs/verification/021-sparse-mask-voting.md` |
| 2 | 写失败测试 — mask shape + voting correctness | `src/tests/test_sparse_mask.py` |
| 3 | 实现 `compute_sparse_mask()` | `src/quantize/_sparse_mask.py` |
| 4 | 通过测试 + 验证 | 同上 |
| 5 | Review + commit | — |

## 验证

- Unit tests: mask shape matches tensor shape, voting correctness with known data, edge cases (zero input, all-same, single sample)
- Integration: calibrate with Multiple Samples → verify mask positions are high-variance elements
