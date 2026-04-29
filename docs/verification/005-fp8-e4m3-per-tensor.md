# 005: fp8_e4m3 per_tensor 量化验证

**对应测试函数**: `test_fp8_e4m3_per_tensor()`
**验证层级**: Layer 1 — 核心量化

## 量化公式（fp8_e4m3 per_tensor）

fp8_e4m3 格式参数（来自 registry）：
- ebits=4, mbits=5（sign(1)+implicit(1)+actual_mantissa(3)=5）
- max_norm_override=448.0
- emax = 2^(4-1) = 8
- saturate_normals = False (ebits != 0)

```
def _quantize_elemwise_core(A, bits=5, exp_bits=4, max_norm=448.0, saturate_normals=False):
    # exp_bits != 0 → 提取指数
    private_exp = floor(log2(|A|))
    min_exp = -(2^3) + 2 = -6
    private_exp = clip(private_exp, min=-6)

    # scale up: (A / 2^privExp) * 2^(5-2) = A * 8 / 2^privExp
    out = A / 2^private_exp * 8

    # round mantissa (3-bit actual mantissa: 2^3=8 levels)
    out = sign * floor(|out| + 0.5)

    # scale down
    out = out / 8 * 2^private_exp

    # saturate: values > 448.0 → Inf
    out = where(|out| > 448.0, sign*Inf, out)
```

## 给定数据

```python
x = [[0.5, -0.25], [1.0, 0.75]]
W = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
```

## 手工推导

### x 量化

**0.5**: log2(0.5) = -1 → privExp=-1 → 0.5/0.5*8=8 → round=8 → 8/8*0.5=**0.5**
**-0.25**: log2(0.25) = -2 → privExp=-2 → 0.25/0.25*8=8 → round=8 → 8/8*0.25=**-0.25**
**1.0**: log2(1.0) = 0 → privExp=0 → 1.0/1.0*8=8 → round=8 → 8/8*1.0=**1.0**
**0.75**: log2(0.75) ≈ -0.415 → privExp=-1 → 0.75/0.5*8=12 → round=12 → 12/8*0.5=**0.75**

所有值在 e4m3 格式中均可精确表示（3-bit 尾数 × exponent range）。max_norm=448.0 远超这些值，无截断。

`expected_x = [[0.5, -0.25], [1.0, 0.75]]`

### W 量化

W 值更大但全为整数，在 e4m3 的表示范围内：

**1.0**: privExp=0, 1.0/1.0*8=8, round=8, 8/8*1.0=**1.0**
**2.0**: privExp=1, 2.0/2.0*8=8, round=8, 8/8*2.0=**2.0**
**3.0**: privExp=1, 3.0/2.0*8=12, round=12, 12/8*2.0=**3.0**
**4.0**: privExp=2, 4.0/4.0*8=8, round=8, 8/8*4.0=**4.0**
**5.0**: privExp=2, 5.0/4.0*8=10, round=10, 10/8*4.0=**5.0**
**6.0**: privExp=2, 6.0/4.0*8=12, round=12, 12/8*4.0=**6.0**

全精确。与 int8 的 max_norm=1.984375 不同，e4m3 的 max_norm=448.0 不会截断这些值。

`expected_W = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]`

## 期望值

```python
expected_x = torch.tensor([[0.5, -0.25], [1.0, 0.75]])
expected_W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
```

## 验证结果

- [x] 运行日期: 2026-04-29
- [x] 结果: PASS
- [x] 说明: x/W 全部在 e4m3 精确可表示范围内，max_norm=448.0 无截断，bit-exact 通过
