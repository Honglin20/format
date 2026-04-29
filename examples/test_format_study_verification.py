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

# 预定义的 granularity spec
PER_T = GranularitySpec.per_tensor()
PER_C0 = GranularitySpec.per_channel(axis=0)
PER_Cm1 = GranularitySpec.per_channel(axis=-1)

# 预定义的 format 实例
int8_fmt = IntFormat(bits=8)
int4_fmt = IntFormat(bits=4)

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
