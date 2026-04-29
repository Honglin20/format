# Current Task

**Task ID**: P8.R1 — Pipeline Refactor
**Plan**: docs/plans/2026-04-29-pipeline-refactor.md
**Branch**: claude/pipeline-refactor (worktree)

## Progress

- [x] Task 1-4: Derivation docs (013-016)
- [x] Task 5: protocol.py + config.py + tests (13 tests)
- [x] Task 6: runner.py + tests (3 tests)
- [x] Task 7: studies/format_study.py (pure data)
- [x] Task 8: theme.py + save.py + tests (2 tests)
- [x] Task 9: tables.py + tests (1 test)
- [x] Task 10: figures.py + tests (23 tests)
- [x] Task 11: Refactor experiment_format_study.py (696 lines removed)
- [x] Task 12: Integration tests (3 tests)
- [x] Task 13: Full suite + status update

## 下一步

Merge claude/pipeline-refactor into feature/refactor-src, then return to Phase 8 P1 mainline (Transform体系: SmoothQuant/Bias correction/CLE/Hadamard).

## 断点续传必读文件

1. `src/pipeline/runner.py`（全文）
2. `src/pipeline/config.py`（全文）
3. `src/viz/figures.py`（全文）
4. `examples/experiment_format_study.py`（全文，重构后 ~1300 行）
5. `docs/plans/2026-04-29-pipeline-refactor.md`（全文）

## 已知预存在测试失败

`pytest src/tests/` 有 26 个预存在失败（非本分支引入）：
- `test_golden_equiv.py` — 26 tests FileNotFoundError（golden data `.pt` 文件未 staging）
- 排除 golden 测试后全部通过：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q`

## 关键经验记录

1. **EvalFn IoC 模式验证通过**：单回调驱动 calibrate/analyze/evaluate 三阶段，模型交互完全由用户控制
2. **resolve_config descriptor 模式**：19 个 descriptor 全部解析成功，纯数据定义搜索空间可行
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）
4. **Figure 参数化**：11 个图表函数从硬编码 title/colors 改为参数，保持向后兼容
