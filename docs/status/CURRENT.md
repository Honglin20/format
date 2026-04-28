# Current Task

**Task ID**: P8 — Format Precision Study (Review Fix Pass)
**Plan**: docs/plans/2026-04-28-format-study-plan.md
**Design**: docs/plans/2026-04-28-format-study-design.md
**Branch**: worktrees/format-study
**Tests baseline**: 1305 passed, 0 xfail

## Progress

- [x] **Task 1: Experiment script skeleton and user-facing API** ✅
- [x] **Task 2: Implement run_experiment() — single config runner** ✅
- [x] **Task 3: Part A — 8-bit format comparison** ✅
- [x] **Task 4: Part B — 4-bit format comparison** ✅
- [x] **Task 5: Part C — FP32 vs PoT scaling comparison** ✅
- [x] **Task 6: Part D — Transform analysis (SmoothQuant + Hadamard)** ✅
- [x] **Task 7: Block size sensitivity sweep** ✅
- [x] **Task 8: Table generation (6 tables)** ✅
- [x] **Task 9: Figure generation (11 figures)** ✅
- [x] **Task 10: Cleanup, defaults, and documentation** ✅
- [x] **Review fix pass — 18 issues (C1-C3, I1-I5, M1-M7, S1-S4)** ✅ (commit 92f870c)

## 下一步（具体动作）

Experiment script is fully functional. User should verify end-to-end by running:
`PYTHONPATH=. python examples/experiment_format_study.py`

## 断点续传必读文件

1. `examples/experiment_format_study.py`（全文，含所有 review 修复）
2. `src/transform/smooth_quant.py`（SmoothQuantWeightTransform companion class）
3. `src/transform/__init__.py`（SmoothQuantWeightTransform export）
4. `docs/plans/2026-04-28-format-study-plan.md`（全文）

## 关键经验记录

1. **SmoothQuant weight compensation 必须匹配原文**：activation 用 x/s（dim=-1），weight 用 W*s（dim=1），两个 Transform 独立。`_make_sq_op_cfg` 中 weight 角色必须用 SmoothQuantWeightTransform
2. **eval_fn 静默丢失**：所有 Part runner 和 `run_experiment` 必须逐层转发 `eval_fn` 参数，缺少即静默退回默认 accuracy
3. **Per-layer config fallback**：`sq_per_layer_cfg` 只包含 Linear/Conv 的条目，其他模块（LayerNorm）会 fallback 到 `_EMPTY_CFG`（不量化），造成 QSNR 人为膨胀
4. **HistogramObserver key 名**：返回 `fp32_hist`/`quant_hist`/`err_hist`（torch.histc counts），不是 `hist_bins`/`hist_counts`
5. **Format family 颜色一致性**：MXINT-8/MXINT-4 共享蓝色系，MXFP-8/MXFP-4 共享暖色系；Wong 2011 色盲友好调色板
6. **Fig 9 子图分离**：格式间 layer 数量差异大时（ToyMLP vs transformer），单 chart 堆叠 bars 会重叠；按格式拆分子图解决
7. **CLI argparse**：支持 `--skip-part-{a,b,c,d}`、`--plot-from results.json`（redraw 免重跑）
