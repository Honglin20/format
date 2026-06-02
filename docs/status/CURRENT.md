# Current Task

**当前任务**: 空闲 — bitx API + AgentHarness 集成已完成
**Branch**: `feature/refactor-src`

---

## 已完成：bitx API + AgentHarness 集成

### bitx 分析 API（`src/api/mxint_error_analysis.py`）

独立运行 MXInt8 误差分析脚本，支持 10 类图表（基础分析 ①–⑦ + 误差归因 ⑧ + 精度恢复 ⑨⑩）。
通过 adapter 三函数合约（get_model / get_eval_fn / get_data）与目标项目解耦。

```bash
python -m src.api.mxint_error_analysis --adapter /path/to/_adapter.py
```

### AgentHarness 工作流（`AgentHarness/workflows/mxint-analysis/`）

三 agent 串行流水线：analyzer → configurator → runner。
- workflow.json：UI 启动用（默认 AgentResult）
- examples/18_bitx_mxint_analysis.py：Python API 用（自定义 result_type，结构化输出）

### 集成文档（`docs/harness/`）

| 文件 | 内容 |
|------|------|
| [INDEX.md](../harness/INDEX.md) | 总索引 |
| [adapter-guide.md](../harness/adapter-guide.md) | adapter 三函数合约 |
| [workflow-patterns.md](../harness/workflow-patterns.md) | 工作流模式 + 反模式 |
| [result-types.md](../harness/result-types.md) | 自定义 Pydantic result_type |
| [charts.md](../harness/charts.md) | render_chart 图表清单 |

---

## 待讨论：GPTQ + Sparse/Group-Sparse

GPTQ + sparse 精度仍为负增益。根因：sparse 打破 Hessian 补偿的等精度假设。
待讨论方案：GPTQ 用纯 int4 做 Hessian 补偿，sparse 在 forward 独立处理。

---

## 断点续传必读文件

1. `src/api/mxint_error_analysis.py` — bitx 分析 API 主脚本
2. `docs/harness/INDEX.md` — AgentHarness 集成文档索引

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,627 passed, 40 failed
