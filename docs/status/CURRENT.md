# Current Task

**当前任务**: Example 19 Phase C 完成 — 准备进入 Phase D (Integration Test)
**Branch**: `feature/refactor-src`

---

## 已完成：Phase A — 4 个新 bitx API

| API | 文件 | 功能 |
|-----|------|------|
| `PerBlockQSNRObserver` | `src/analysis/observers.py` | 逐 block/channel 记录 QSNR（vectorized） |
| `block_error_analysis()` | `src/api/block_error_analysis.py` | 从 SessionResult 提取 per-block QSNR 排名 |
| `CrossConfigLayerRanking` | `src/analysis/cross_config_ranking.py` | 跨 config 层排序 |
| `TransformEffectReport` | `src/analysis/transform_effect.py` | 自动检测 ±SQ/±HD config 对，量化恢复效果 |

## 已完成：Phase B — Block Error Visualization

| 可视化 | 文件 | 功能 |
|--------|------|------|
| `block_error_heatmap()` | `src/viz/block_error_heatmap.py` | 1D/2D 热力图（block × channel, 颜色=QSNR） |
| `channel_error_bar()` | 同上 | per-channel QSNR 柱状图，outlier 标红 |
| `multi_config_block_comparison()` | 同上 | 跨 config (W8A8 vs W4A4) block 误差分组对比 |

## 已完成：Phase C — Agent Pipeline

| 交付物 | 文件 |
|--------|------|
| Workflow 定义 + 8 个 Pydantic result_type | `AgentHarness/examples/19_mxint_diagnostic.py` |
| workflow.json (自动生成) | `AgentHarness/workflows/mxint-diagnostic/workflow.json` |
| 8 个 agent md 指令文件 | `AgentHarness/workflows/mxint-diagnostic/agents/*.md` |
| 4 个 agent API 接口文档 | `docs/harness/api-*.md` |

### DAG 结构

```
adapter → study_runner ─┬→ gap_analyzer ───────────────────┐
                        └→ layer_attribution ─┬→ dist_prof ─┤
                                              ├→ block_an ──┤
                                              └→ intervent ─┘
                                                              ↓
                                                       synthesis
```

---

## 下一步：Phase D — Integration Test

1. 用 bitx MLP (MNIST) 做 end-to-end test
2. 验证所有 charts 正确渲染
3. Review + Commit

---

## 断点续传必读文件

1. `docs/harness/example19-spec.md` — 完整设计文档
2. `AgentHarness/examples/19_mxint_diagnostic.py` — Workflow 定义
3. `src/viz/block_error_heatmap.py` — 可视化函数
4. `src/analysis/cross_config_ranking.py` — 跨 config 层排序

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,700 passed, 48 failed (pre-existing)
