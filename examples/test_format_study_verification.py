#!/usr/bin/env python3
"""
Format Study Verification Script
==================================

每个测试用例对应 ``docs/verification/NNN-name.md`` 中的手工推导。
验证方法：先读推导文档理解期望值来源，再运行此脚本做 bit-exact 比对。

三层递进验证：
  Layer 1 (tests 001-006): 核心量化 — Format.quantize() 逐元素正确性
  Layer 2 (tests 007-009): 算子 + Transform — Linear + SmoothQuant/Hadamard
  Layer 3 (tests 010-012): 完整 Pipeline — QuantSession calibrate→analyze→compare

所有测试共享同一组固定数据，确保可复现。
"""
import torch
from src.formats.int_formats import IntFormat
from src.formats.fp_formats import FPFormat
from src.formats.lookup_formats import NF4Format
from src.scheme.granularity import GranularitySpec
from src.scheme.quant_scheme import QuantScheme
from src.scheme.op_config import OpQuantConfig
from src.scheme.transform import IdentityTransform

# ---------------------------------------------------------------------------
# 固定数据（所有测试共享）
# ---------------------------------------------------------------------------

# Linear(2, 3): in=2, out=3
W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
x = torch.tensor([[0.5, -0.25], [1.0, 0.75]], dtype=torch.float32)

# fp32 reference output: y = x @ W^T + b
# y[0] = [0.5*1+(-0.25)*2, 0.5*3+(-0.25)*4, 0.5*5+(-0.25)*6] + [0.1,0.2,0.3]
#      = [0.0, 0.5, 1.0] + [0.1,0.2,0.3] = [0.1, 0.7, 1.3]
# y[1] = [1.0*1+0.75*2, 1.0*3+0.75*4, 1.0*5+0.75*6] + [0.1,0.2,0.3]
#      = [2.5, 6.0, 9.5] + [0.1,0.2,0.3] = [2.6, 6.2, 9.8]
fp32_y = torch.tensor([[0.1, 0.7, 1.3], [2.6, 6.2, 9.8]], dtype=torch.float32)

# 预定义的 format 实例
int8_fmt = IntFormat(bits=8)
int4_fmt = IntFormat(bits=4)

# 预定义的 granularity spec
PER_T = GranularitySpec.per_tensor()
PER_C0 = GranularitySpec.per_channel(axis=0)
PER_Cm1 = GranularitySpec.per_channel(axis=-1)

# ---------------------------------------------------------------------------
# Layer 1: 核心量化验证
# ---------------------------------------------------------------------------


def test_int8_per_tensor():
    """验证 int8 per_tensor 量化。

    推导: docs/verification/001-int8-per-tensor.md

    量化公式: x*64 → round → /64 → clamp(±127/64≈±1.984375)
    步长 1/64=0.015625, 共 127 个离散等级。
    """
    gran = PER_T

    # --- x 量化 ---
    # 所有输入值恰好是 1/64 的整数倍 → 量化无损
    expected_x = torch.tensor([[0.5, -0.25], [1.0, 0.75]], dtype=torch.float32)
    actual_x = int8_fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), (
        f"x mismatch:\n  actual: {actual_x}\n  expected: {expected_x}"
    )

    # --- W 量化 ---
    # 1.0 → OK; 2.0, 3.0, 4.0, 5.0, 6.0 全部超出 max_norm=1.984375 → 截断
    expected_w = torch.tensor(
        [[1.0, 1.984375], [1.984375, 1.984375], [1.984375, 1.984375]],
        dtype=torch.float32,
    )
    actual_w = int8_fmt.quantize(W, gran)
    assert torch.equal(actual_w, expected_w), (
        f"W mismatch:\n  actual:\n{actual_w}\n  expected:\n{expected_w}"
    )


def test_int8_per_channel():
    """验证 int8 per_channel(axis=0) 量化。

    推导: docs/verification/002-int8-per-channel.md

    每列独立: amax→norm→quantize_elemwise→rescale。
    """
    gran = PER_C0

    # --- x 量化 ---
    # col0 amax=1.0, col1 amax=0.75
    expected_x = torch.tensor(
        [[0.5, -0.24609375], [1.0, 0.75]], dtype=torch.float32,
    )
    actual_x = int8_fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), (
        f"x mismatch:\n  actual:\n{actual_x}\n  expected:\n{expected_x}"
    )

    # --- W 量化 ---
    # col0 amax=5.0, col1 amax=6.0
    expected_w = torch.tensor(
        [[1.015625, 1.96875], [2.96875, 4.03125], [5.0, 6.0]],
        dtype=torch.float32,
    )
    actual_w = int8_fmt.quantize(W, gran)
    assert torch.equal(actual_w, expected_w), (
        f"W mismatch:\n  actual:\n{actual_w}\n  expected:\n{expected_w}"
    )


def test_int4_per_tensor():
    """验证 int4 per_tensor 量化。

    推导: docs/verification/003-int4-per-tensor.md

    量化公式: x*4 → round → /4 → clamp(±1.75)
    步长 0.25, 共 15 个离散等级。
    """
    gran = PER_T

    # --- x 量化 ---
    # 所有输入值恰好是 0.25 的整数倍 → 无损
    expected_x = torch.tensor([[0.5, -0.25], [1.0, 0.75]], dtype=torch.float32)
    actual_x = int4_fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), (
        f"x mismatch:\n  actual: {actual_x}\n  expected: {expected_x}"
    )

    # --- W 量化 ---
    # 1.0→OK; 2.0,3.0,4.0,5.0,6.0 全部超出 max_norm=1.75 → 截断为 1.75
    expected_w = torch.tensor(
        [[1.0, 1.75], [1.75, 1.75], [1.75, 1.75]], dtype=torch.float32,
    )
    actual_w = int4_fmt.quantize(W, gran)
    assert torch.equal(actual_w, expected_w), (
        f"W mismatch:\n  actual:\n{actual_w}\n  expected:\n{expected_w}"
    )


def test_int4_per_channel():
    """验证 int4 per_channel(axis=0) 量化。

    推导: docs/verification/004-int4-per-channel.md

    每列独立 scale → int4 elemwise → rescale。
    """
    gran = PER_C0

    # --- x 量化 ---
    # col0 amax=1.0, col1 amax=0.75
    expected_x = torch.tensor(
        [[0.5, -0.1875], [1.0, 0.75]], dtype=torch.float32,
    )
    actual_x = int4_fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), (
        f"x mismatch:\n  actual:\n{actual_x}\n  expected:\n{expected_x}"
    )

    # --- W 量化 ---
    # col0 amax=5.0, col1 amax=6.0
    expected_w = torch.tensor(
        [[1.25, 1.5], [2.5, 4.5], [5.0, 6.0]], dtype=torch.float32,
    )
    actual_w = int4_fmt.quantize(W, gran)
    assert torch.equal(actual_w, expected_w), (
        f"W mismatch:\n  actual:\n{actual_w}\n  expected:\n{expected_w}"
    )


def test_fp8_e4m3_per_tensor():
    """验证 fp8_e4m3 per_tensor 量化。

    推导: docs/verification/005-fp8-e4m3-per-tensor.md

    ebits=4, mbits=5, max_norm=448.0. 浮点格式: 提取指数→scale→round→rescale.
    """
    gran = PER_T
    fp8_fmt = FPFormat(name="fp8_e4m3", ebits=4, mbits=5, max_norm_override=448.0)

    # --- x 量化 ---
    # 所有值在 e4m3 下精确可表示 (3-bit mantissa, exponent range [-6, 8])
    expected_x = torch.tensor([[0.5, -0.25], [1.0, 0.75]], dtype=torch.float32)
    actual_x = fp8_fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), (
        f"x mismatch:\n  actual: {actual_x}\n  expected: {expected_x}"
    )

    # --- W 量化 ---
    # 全部在 max_norm=448.0 以内，且精确可表示
    expected_w = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32,
    )
    actual_w = fp8_fmt.quantize(W, gran)
    assert torch.equal(actual_w, expected_w), (
        f"W mismatch:\n  actual:\n{actual_w}\n  expected:\n{expected_w}"
    )


def test_nf4_per_channel():
    """验证 nf4 per_channel(axis=0) 量化。

    推导: docs/verification/006-nf4-per-channel.md

    NF4 16-level LUT, per_channel: normalize → nearest-neighbor → rescale.
    """
    from src.formats.lookup_formats import NF4Format
    gran = PER_C0
    nf4_fmt = NF4Format()

    # --- x 量化 ---
    # col0 amax=1.0, col1 amax=0.75
    # normalized: [0.5, 1.0] → nf4 LUT → [0.44071, 1.0] → * amax
    # normalized: [-0.3333, 1.0] → nf4 LUT → [-0.28444, 1.0] → * amax
    expected_x = torch.tensor(
        [[0.44070982933044434, -0.21333104372024536], [1.0, 0.75]],
        dtype=torch.float32,
    )
    actual_x = nf4_fmt.quantize(x, gran)
    assert torch.equal(actual_x, expected_x), (
        f"x mismatch:\n  actual: {actual_x}\n  expected: {expected_x}"
    )
