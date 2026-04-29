# Current Task

**Task ID**: P8.V2 — Pipeline Refactor
**Plan**: (inline in task description)
**Branch**: claude/pipeline-refactor
**Tests baseline**: 1322 passed, 0 failures

## Progress

- [x] Task 0: Create docs/verification/ directory + README (previous phase)
- [x] Task 1-14: Format Study Verification (previous phase, all complete)
- [x] Task 5: Implement src/pipeline/config.py resolve_config descriptor parser
- [x] Task 6: Implement src/pipeline/runner.py ExperimentRunner
- [ ] Task 7: Implement src/pipeline/comparison.py (multi-config delta comparison)
- [ ] Task 8: Implement src/pipeline/search.py (auto-sweep over search space)
- [ ] Task 9: Implement src/pipeline/visualization.py (study results dashboard)
- [ ] Task 10: Update docs/ with pipeline examples

## Next Step (specific action)

Implement `src/pipeline/comparison.py` — multi-config delta comparison table generator.

## 断点续传必读文件

1. `src/pipeline/runner.py`（全文）
2. `src/pipeline/config.py`（全文）
3. `src/pipeline/__init__.py`（全文）
4. `src/session.py`（1-80 行）
5. `docs/status/CURRENT.md`（全文）

## 关键经验记录

1. **QuantSession deepcopy isolation**: QuantSession(keep_fp32=True) internally deep-copies the model, so the runner does NOT need to manage fp32_model separately during calibration/analysis — only during evaluation (where an independent fp32 run is needed for comparison).
2. **analyze_data fallback**: When analyze_data is None but calib_data is provided, the runner automatically reuses calib_data for analysis. This is the common case where calibration forward passes are also used for observer data collection.
3. **eval_fn IoC pattern**: The eval_fn receives the QuantSession itself (not just qmodel), so calibrate/analyze contexts that wrap session() calls work transparently through the same function.
