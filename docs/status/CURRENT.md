# Current Task

**Task ID**: P8.1 — SmoothQuant Configurable Channel Axis
**Plan**: docs/plans/2026-04-28-smoothquant-channel-axis.md
**Branch**: worktrees/format-study
**Tests baseline**: 1316 passed, 0 failures

## Progress

- [x] **Task 1: Add `channel_axis` to `SmoothQuantTransform`** ✅
- [x] **Task 2: Add `SmoothQuantWeightTransform`** ✅ (later removed in cleanup)
- [x] **Task 3: Add axis params to `compute_smoothquant_scale` and `from_calibration`** ✅
- [x] **Task 4: Fix experiment to pass correct axis for Conv2d** ✅
- [x] **Task 5: Cross-check all callers + full test suite** ✅
- [x] **Cleanup: Remove `SmoothQuantWeightTransform`, fuse W*s at calibration** ✅

## Key Design Decision

SmoothQuant weight compensation `W * s` is a ONE-TIME calibration fusion, not a per-forward transform. Weight fusion into model parameters at calibration time matches the original paper (Xiao et al., 2023) exactly. `SmoothQuantWeightTransform` was removed — the weight side uses `IdentityTransform` since the scale is permanently baked in.

## 下一步

Merge `worktrees/format-study` into `feature/refactor-src` and push.

## 断点续传必读文件

1. `src/transform/smooth_quant.py`（SmoothQuantTransform + compute_smoothquant_scale）
2. `src/tests/test_transform_smooth_quant.py`（40 tests, TestSmoothQuantTransform + TestComputeSmoothQuantScale + TestSmoothQuantQuantScheme）
3. `examples/experiment_format_study.py`（_make_smoothquant_transforms with weight fusion）
4. `docs/plans/2026-04-28-smoothquant-channel-axis.md`（实现计划）

## 关键经验记录

1. **SmoothQuant 原论文做法**：calibration 阶段一次性 `W = W * s`，推理时只需 `x / s`。不要用 per-forward weight transform
2. **`d != -1` Python 陷阱**：`0 != -1` 是 True，所以用负轴过滤 reduce_dims 时不会排除任何维度。必须用 `axis if axis >= 0 else ndim + axis` 规范化
3. **Conv2d channel axis**：NCHW 中 channel 是 dim=1，不是 dim=-1。Linear (N, C) 中 channel 是 dim=-1
