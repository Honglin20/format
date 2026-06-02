# 021: Sparse Mask Cross-Sample Voting 数学推导

**对应测试**: `test_sparse_mask.py::TestSparseMaskVoting`
**验证层级**: Layer 2（校准期 mask 构造）

## 算法原理

给定校准集 `x_calib ∈ R^(S×D1×D2×...)`（S 个 sample），目标找一个固定 mask `M ∈ {0,1}^(D1×D2×...)` 使得该 mask 对所有 sample 的平均量化效果最优。

### Step 1 — Per-sample group-level top-k

对每个 sample s，在每个 granularity group g 内部独立取 top-k：

```
mask_s[g] = topk(|x_s[g]|, k_g)   where k_g = max(1, |g| * outlier_ratio)
```

结果: `mask_all ∈ {0,1}^(S×D1×D2×...)`，每个 sample 有自己的 mask。

### Step 2 — Cross-sample voting

```
mask_avg = (1/S) Σ_{s=1}^{S} mask_s    ∈ [0,1]^(D1×D2×...)
```

某个位置 p 的 `mask_avg[p]` 表示 "在多大比例 sample 中位置 p 被选为 outlier"。

### Step 3 — Final mask

```
k_total = max(1, |D1×D2×...| * outlier_ratio)
M = topk(mask_avg.flatten(), k_total)  → reshape to (D1×D2×...)
```

## 手工推导: 3 samples, 2×3 tensor, PER_TENSOR, outlier_ratio=1/3

```
x_0 = [[1, 5, 2],    x_1 = [[1, 8, 3],    x_2 = [[9, 1, 2],
       [3, 0, 4]]           [5, 2, 4]]           [3, 4, 1]]
```

### Per-sample top-2 (k = 6 * 1/3 = 2):

```
Sample 0: top-2 positions by abs value → (0,1)=5, (1,2)=4
  mask_0 = [[0,1,0], [0,0,1]]

Sample 1: top-2 → (0,1)=8, (1,1)=5? wait...
  abs: [1,8,3,5,2,4] → top-2: (0,1)=8, (1,0)=5
  mask_1 = [[0,1,0], [1,0,0]]

Sample 2: top-2 → (0,0)=9, (1,1)=4
  mask_2 = [[1,0,0], [0,1,0]]
```

### mask_avg:

```
mask_avg = (mask_0 + mask_1 + mask_2) / 3
         = [[1/3, 2/3, 0  ],
            [1/3, 1/3, 1/3]]
```

### Final mask (top-2):

Positions sorted by avg score: (0,1)=0.667, (0,0)=0.333, (1,0)=0.333, (1,1)=0.333, (1,2)=0.333, (0,2)=0

Top-2: (0,1) and one of the 0.333 positions (implementation defines tie-breaking, typically first encountered).

```
expected_mask = [[0, 1, 0],
                 [1, 0, 0]]  (假定期望 tie-break 选 (1,0))
```

> 注: `torch.topk` 的 tie-breaking 不是规范行为，测试中用严格控制的数据避免 tie。

## PER_CHANNEL 推导: 2 samples, 2×4 tensor, channel_axis=0, ratio=0.5

```
x_0 = [[1, 8, 3, 2],    x_1 = [[4, 2, 1, 5],
       [9, 1, 4, 5]]           [3, 7, 2, 1]]

Channel 0 (row 0): k = max(1, 4*0.5) = 2
  Sample 0, ch0: abs [1,8,3,2] → top-2: cols 1,2 → mask=[0,1,1,0]
  Sample 1, ch0: abs [4,2,1,5] → top-2: cols 0,3 → mask=[1,0,0,1]
  avg ch0: [0.5, 0.5, 0.5, 0.5]

Channel 1 (row 1): k = 2
  Sample 0, ch1: abs [9,1,4,5] → top-2: cols 0,3 → mask=[1,0,0,1]
  Sample 1, ch1: abs [3,7,2,1] → top-2: cols 1,2 → mask=[0,1,1,0]
  avg ch1: [0.5, 0.5, 0.5, 0.5]

per-channel top-2: tie across all positions — implementation defined.
```

> 测试中 PER_CHANNEL 使用设计好的非对称数据以确保唯一确定的结果。

## 边界情况

### Single sample (S=1)

mask_avg = mask_0 → final_mask = topk(mask_0, k_total)。退化为单 sample top-k（因为 mask_0 只有 0/1，topk 选 k_total 个 1）。

### All-identical samples

所有 mask_s 相同 → mask_avg = mask_0 → final_mask = topk(mask_0, k_total)。结果同单 sample。

### Zero tensor

`torch.topk` 在 zeros 上的行为: 返回前 k 个索引（通常是索引最小的 k 个 position）。此时所有位置 magnitude 相同，mask 由 position order 决定——非量化相关输入。

## 验证结果

- [ ] 运行日期: 
- [ ] 结果: 
- [ ] 实际输出与期望值完全一致
