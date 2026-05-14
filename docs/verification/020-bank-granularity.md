# 020: BANK Granularity 量化正确性

**对应测试**: `test_bank_quantize_pot_bit_exact` / `test_bank_quantize_fp32_bit_exact`
**验证层级**: Layer 1（核心量化）

## 格式原理

BANK: 沿 bank_axis 将 tensor 切分为 bank_size 大小的连续组。每个 bank 覆盖该轴段内的所有元素（跨所有非 bank 维度），共享一个 amax。

- format: int4
- granularity: bank, bank_axis=-1, bank_size=2
- scale_storage: pot / fp32

## 给定数据

```python
x = [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]  # shape (2, 4)
```

bank_axis=-1, bank_size=2 → num_banks = 4/2 = 2

## 手工推导 (POT)

reshape: (2, 4) → (2, 2, 2)
Bank dim at axis=1, inner dim at axis=2

Bank 0 (cols 0-1): values [1,2,5,6]
  amax = 6.0 → pot = 8.0
  normalized: [1,2,5,6]/8 = [0.125, 0.25, 0.625, 0.75]

  int4 elemwise (mbits=4, max_norm=1.75):
    x_q_elem = [0.25, 0.25, 0.75, 0.75]
    x_q = x_q_elem * 8.0 = [2.0, 2.0, 6.0, 6.0]

Bank 1 (cols 2-3): values [3,4,7,8]
  amax = 8.0 → pot = 8.0
  normalized: [3,4,7,8]/8 = [0.375, 0.5, 0.875, 1.0]

  int4 elemwise:
    x_q_elem = [0.5, 0.5, 1.0, 1.0]
    x_q = x_q_elem * 8.0 = [4.0, 4.0, 8.0, 8.0]

**期望值 (POT): `[[2.0, 2.0, 4.0, 4.0], [6.0, 6.0, 8.0, 8.0]]`**

## 手工推导 (FP32)

Bank 0: amax=6.0 (no pot rounding)
  [1,2,5,6]/6 = [0.1667, 0.333, 0.833, 1.0]
  int4: [0.25, 0.25, 0.75, 1.0]
  *6 = [1.5, 1.5, 4.5, 6.0]

Bank 1: amax=8.0 (no pot rounding)
  [3,4,7,8]/8 = [0.375, 0.5, 0.875, 1.0]
  int4: [0.5, 0.5, 1.0, 1.0]
  *8 = [4.0, 4.0, 8.0, 8.0]

**期望值 (FP32): `[[1.5, 1.5, 4.0, 4.0], [4.5, 6.0, 8.0, 8.0]]`**

## BANK vs PER_BLOCK 对比

同输入用 PER_BLOCK(axis=-1, block_size=2, no sparse):
- PER_BLOCK: M×(N/2)=4 个 block，每个 block 内 2 个元素共享 shared_exp
- BANK: N/2=2 个 bank，每个 bank 覆盖 M×2=4 个元素共享 amax

BANK 的 scale 更粗粒度——同一个 bank 跨所有 M 行。

## 验证结果

- [x] 运行日期: 2026-05-15
- [x] 结果: 23 tests PASS
- [x] 实际输出与手工推导期望值完全一致（torch.equal）
