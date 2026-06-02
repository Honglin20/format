# Current Task

**当前任务**: Example 19 Phase A+B 完成 — 准备进入 Phase C (Agent Pipeline)
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

### Session 集成修改

- `src/session/_helpers.py` — `_OBSERVER_MAP` 新增 `"per_block_qsnr"`
- `src/session/_session.py` — `run_quantization` / `_run_hook_analysis` 合并自定义 observers

### 测试

- `src/tests/test_example19_apis.py` — 23 tests
- `src/tests/test_block_viz.py` — 6 tests
- E2E: MLP 模型生成 18 张可视化图，全部正确渲染
- 回归：2,700 passed, 48 failed (pre-existing)

---

## 下一步：Phase C — Agent Pipeline

按 spec §5 实现 8-agent 串行流水线：

1. `examples/19_mxint_diagnostic.py` — Pydantic result types + Workflow 定义
2. `workflows/mxint-diagnostic/agents/*.md` — 8 个 agent md 文件
3. `docs/harness/api-*.md` — agent 接口文档

---

## 断点续传必读文件

1. `docs/harness/example19-spec.md` — 完整设计文档
2. `src/viz/block_error_heatmap.py` — 可视化函数
3. `src/session/_session.py` — observer 合并逻辑

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,700 passed, 48 failed (pre-existing)
