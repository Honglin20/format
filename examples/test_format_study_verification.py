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


# ---------------------------------------------------------------------------
# Layer 2: 算子 + Transform 验证
# ---------------------------------------------------------------------------


def test_linear_int8_per_channel_forward():
    """验证 QuantizedLinear 前向传播（int8 per_channel 全量化）。

    推导: docs/verification/007-linear-int8-pc-forward.md

    手工推演 LinearFunction.forward: 量化输入→量化权重→matmul→加bias→量化输出。
    """
    from src.ops.linear import LinearFunction

    scheme = QuantScheme(format=int8_fmt, granularity=PER_C0, transform=IdentityTransform())
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

    actual_y = LinearFunction.apply(x, W, b, cfg)

    expected_y = torch.tensor(
        [[0.121508784592152, 0.677270472049713, 1.378125071525574],
         [2.592187404632568, 6.192187309265137, 9.800000190734863]],
        dtype=torch.float32,
    )
    assert torch.equal(actual_y, expected_y), (
        f"output mismatch:\n  actual:\n{actual_y}\n  expected:\n{expected_y}"
    )


def test_linear_smoothquant_equivalence():
    """验证 SmoothQuant 数学等价性: (x/s) @ (W*s) = x @ W。

    推导: docs/verification/008-linear-smoothquant.md

    SmoothQuant (Xiao et al., 2023): per-channel scaling 保持 matmul 不变。
    """
    from src.transform.smooth_quant import SmoothQuantTransform, compute_smoothquant_scale

    # Compute scale
    s = compute_smoothquant_scale(x, W, alpha=0.5, act_channel_axis=-1, w_channel_axis=1)

    # Verify scale values
    # s[0] = sqrt(1.0)/sqrt(5.0) = 1/2.23607 = 0.44721...
    # s[1] = sqrt(0.75)/sqrt(6.0) = 0.86603/2.44949 = 0.35355...
    expected_s = torch.tensor([0.44721359014511108398, 0.35355338454246520996], dtype=torch.float32)
    assert torch.equal(s, expected_s), (
        f"scale mismatch: {s} vs {expected_s}"
    )

    # Activation smoothing
    sq_t = SmoothQuantTransform(s, channel_axis=-1)
    x_smooth = sq_t.forward(x)
    expected_x_smooth = torch.tensor(
        [[1.11803400516510009766, -0.70710676908493041992],
         [2.23606801033020019531, 2.12132048606872558594]],
        dtype=torch.float32,
    )
    assert torch.equal(x_smooth, expected_x_smooth), (
        f"x/s mismatch:\n  actual:\n{x_smooth}\n  expected:\n{expected_x_smooth}"
    )

    # Weight fusion (W * s, broadcast along input-channel dim=1)
    w_fused = W * s.view(1, -1)
    expected_w_fused = torch.tensor(
        [[0.44721359014511108398, 0.70710676908493041992],
         [1.34164071083068847656, 1.41421353816986083984],
         [2.23606801033020019531, 2.12132024765014648438]],
        dtype=torch.float32,
    )
    assert torch.equal(w_fused, expected_w_fused), (
        f"W*s mismatch:\n  actual:\n{w_fused}\n  expected:\n{expected_w_fused}"
    )

    # Verify mathematical equivalence: (x/s) @ (W*s)^T == x @ W^T
    y_orig = x @ W.T
    y_sq = x_smooth @ w_fused.T
    assert torch.allclose(y_orig, y_sq, atol=1e-7), (
        f"SmoothQuant equivalence broken:\n  x@W^T:\n{y_orig}\n  (x/s)@(W*s)^T:\n{y_sq}"
    )

    # Also verify inverse: x_q * s restores original
    x_restored = sq_t.inverse(x_smooth)
    assert torch.allclose(x_restored, x, atol=1e-7), (
        f"inverse mismatch:\n  restored:\n{x_restored}\n  original:\n{x}"
    )


def test_hadamard_self_inverse():
    """验证 Hadamard 变换的自逆性质: H(H(x)) == x。

    推导: docs/verification/009-linear-hadamard.md

    FWHT with 1/sqrt(d) normalization is self-inverse.
    Last dim = 2 (already power of 2, no padding).
    """
    from src.transform.hadamard import hadamard

    # FWHT along last dim
    hx = hadamard(x)

    # Verify H(x) values
    # Row 0: [0.5, -0.25] → sum=0.25, diff=0.75 → /sqrt(2) → [0.1768, 0.5303]
    # Row 1: [1.0, 0.75] → sum=1.75, diff=0.25 → /sqrt(2) → [1.2374, 0.1768]
    expected_hx = torch.tensor(
        [[0.17677669227123260498, 0.53033012151718139648],
         [1.23743689060211181641, 0.17677669227123260498]],
        dtype=torch.float32,
    )
    assert torch.equal(hx, expected_hx), (
        f"H(x) mismatch:\n  actual:\n{hx}\n  expected:\n{expected_hx}"
    )

    # Verify self-inverse: H(H(x)) == x (allclose: ~6e-8 float32 rounding from double 1/sqrt(2))
    hhx = hadamard(hx)
    assert torch.allclose(hhx, x, atol=1e-7), (
        f"Self-inverse broken: H(H(x)) != x\n  H(H(x)):\n{hhx}\n  x:\n{x}"
    )


# ---------------------------------------------------------------------------
# Layer 3: 完整 Pipeline 验证
# ---------------------------------------------------------------------------


def test_pipeline_mse_scale_strategy():
    """验证 MSEScaleStrategy 的 grid search 算法。

    推导: docs/verification/010-pipeline-calibration.md

    MSEScaleStrategy(n_steps=5): amax → linspace(0.5, 2.0, 5) → 对每个
    candidate 做 _simple_quantize 并计算 per-slice MSE → 选 MSE 最低的 scale。
    """
    from src.calibration.strategies import MSEScaleStrategy

    strategy = MSEScaleStrategy(n_steps=5)
    scale = strategy.compute(x, axis=0)

    # 手工 grid search 结果（见推导文档）：
    # col0 amax=1.0 → best factor=1.625 → scale=1.625
    # col1 amax=0.75 → best factor=1.625 → scale=1.21875
    # axis=0: keepdim along dim 0 → shape (1, 2)
    expected_scale = torch.tensor([[1.625, 1.21875]], dtype=torch.float32)
    assert torch.equal(scale, expected_scale), (
        f"MSEScaleStrategy mismatch:\n  actual: {scale}\n  expected: {expected_scale}"
    )


def test_pipeline_analysis_metrics():
    """验证 QSNR 和 MSE 公式的正确性。

    推导: docs/verification/011-pipeline-analyze.md

    QSNR = 10 * log10(||fp32||² / ||fp32 - quant||²)
    MSE  = mean((fp32 - quant)²)

    使用 Test 002 (int8 per_channel) 和 Test 007 (Linear forward) 的张量对。
    """
    import math

    def compute_qsnr(fp32, quant):
        signal = (fp32 ** 2).sum().item()
        noise = ((fp32 - quant) ** 2).sum().item()
        return 10.0 * math.log10(signal / noise)

    def compute_mse(fp32, quant):
        return ((fp32 - quant) ** 2).mean().item()

    # --- x pair (from Test 002) ---
    x_q = torch.tensor([[0.5, -0.24609375], [1.0, 0.75]], dtype=torch.float32)
    qsnr_x = compute_qsnr(x, x_q)
    mse_x = compute_mse(x, x_q)
    assert qsnr_x == 50.89481202687437, f"QSNR x: {qsnr_x}"
    assert mse_x == 3.814697265625e-06, f"MSE x: {mse_x}"

    # --- W pair (from Test 002) ---
    w_q = torch.tensor(
        [[1.015625, 1.96875], [2.96875, 4.03125], [5.0, 6.0]],
        dtype=torch.float32,
    )
    qsnr_w = compute_qsnr(W, w_q)
    mse_w = compute_mse(W, w_q)
    assert qsnr_w == 44.57457987982031, f"QSNR W: {qsnr_w}"
    assert mse_w == 0.0005289713735692203, f"MSE W: {mse_w}"

    # --- y pair (from Test 007) ---
    y_q = torch.tensor(
        [[0.121508784592152, 0.677270472049713, 1.378125071525574],
         [2.592187404632568, 6.192187309265137, 9.800000190734863]],
        dtype=torch.float32,
    )
    qsnr_y = compute_qsnr(fp32_y, y_q)
    mse_y = compute_mse(fp32_y, y_q)
    assert qsnr_y == 42.99014234455242, f"QSNR y: {qsnr_y}"
    assert mse_y == 0.0012008105404675007, f"MSE y: {mse_y}"


def test_pipeline_quant_session_e2e():
    """验证 QuantSession 端到端流程: quantize_model → calibrate → analyze。

    推导: docs/verification/012-pipeline-e2e.md

    用 nn.Sequential(nn.Linear(2, 3)) 作为容器模型，
    配置 int8 per_channel(axis=0)，验证：
    1. 量化后 forward 输出与 Test 007 bit-exact 一致
    2. CalibrationSession 正确收集 output amax
    3. MSEScaleStrategy 产生正确 scales
    """
    import torch.nn as nn
    from src.session import QuantSession
    from src.calibration.strategies import MSEScaleStrategy
    from src.analysis.observers import QSNRObserver, MSEObserver
    from src.analysis.context import AnalysisContext

    # 构建容器模型（quantize_model 替换 children，根模块必须是容器）
    model = nn.Sequential(nn.Linear(2, 3))
    model[0].weight.data.copy_(W)
    model[0].bias.data.copy_(b)

    scheme = QuantScheme(
        format=int8_fmt, granularity=PER_C0, transform=IdentityTransform(),
    )
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

    session = QuantSession(
        model, cfg, calibrator=MSEScaleStrategy(n_steps=5), keep_fp32=False,
    )

    # --- 1. 量化 forward 与 Test 007 一致 ---
    with torch.no_grad():
        y_q = session.qmodel(x)
    expected_y = torch.tensor(
        [[0.121508784592152, 0.677270472049713, 1.378125071525574],
         [2.592187404632568, 6.192187309265137, 9.800000190734863]],
        dtype=torch.float32,
    )
    assert torch.equal(y_q, expected_y), (
        f"quantized forward mismatch:\n  actual:\n{y_q}\n  expected:\n{expected_y}"
    )

    # --- 2. CalibrationSession: 收集 amax + 计算 scales ---
    with session.calibrate() as calib:
        session(x)

    # amax 沿 axis=-1（per-row）：row0 max=1.3781..., row1 max=9.8
    name = "0"  # QuantizedLinear 在 Sequential 中的 key
    expected_amax = torch.tensor(
        [[1.37812507152557373047], [9.80000019073486328125]], dtype=torch.float32,
    )
    actual_amax = calib._running_amax[name]
    assert torch.equal(actual_amax, expected_amax), (
        f"running_amax mismatch:\n  actual: {actual_amax}\n  expected: {expected_amax}"
    )

    # MSEScaleStrategy(n_steps=5) → best factor=1.625 → scale = amax * 1.625
    expected_scales = torch.tensor(
        [[2.23945331573486328125], [15.92500019073486328125]], dtype=torch.float32,
    )
    actual_scales = calib.scales()[name]
    assert torch.equal(actual_scales, expected_scales), (
        f"scales mismatch:\n  actual: {actual_scales}\n  expected: {expected_scales}"
    )

    # --- 3. AnalysisContext: QSNR/MSE observer ---
    qsnr_obs = QSNRObserver()
    mse_obs = MSEObserver()
    with AnalysisContext(session.qmodel, [qsnr_obs, mse_obs]) as ctx:
        session(x)
    report = ctx.report()
    df = report.to_dataframe()

    # 验证 report 结构：共 7 条记录，覆盖 input/weight/output 三个角色
    assert len(df) == 7, f"expected 7 analysis entries, got {len(df)}"
    roles = set(df["role"])
    assert roles == {"input", "weight", "output"}, f"unexpected roles: {roles}"

    # 验证每条记录都有有效的 QSNR（正数，有限值）
    for _, row in df.iterrows():
        assert row["qsnr_db"] > 0, f"non-positive QSNR at {row['role']}/{row['slice']}"
        assert row["mse"] >= 0, f"negative MSE at {row['role']}/{row['slice']}"

    # 验证 QSNR 和 MSE 的一致性：QSNR 高则 MSE 低
    # output channel 1 有最大 amax，量化相对误差最小 → QSNR 最高
    output_rows = df[df["role"] == "output"]
    best_idx = output_rows["qsnr_db"].idxmax()
    assert output_rows.loc[best_idx, "slice"] == "('channel', 1)", (
        f"expected output channel 1 to have best QSNR, got {output_rows.loc[best_idx, 'slice']}"
    )
