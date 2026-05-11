# Current Task

**当前任务**: ADR-009 实施 — `quantize_nonlinear` 非线性算子 operand 入口两级量化
**下一任务**: P7 自动格式搜索
**Branch**: `feature/refactor-src`

---

## ADR-009 实施进度

- [ ] `quantize_nonlinear=True` 时，非线性算子入口 operand 施加 storage + per_block compute 两级量化
- [ ] 中间 vec_ops 和 backward 保持 storage-only（与 `False` 一致）
- [ ] 测试：norm/activation/pool/softmax 的 quantize_nonlinear=True 行为验证

详见 `docs/architecture/009-quantize-nonlinear.md`

---

## 断点续传必读文件

1. `docs/architecture/009-quantize-nonlinear.md`（ADR-009：当前实施任务）
2. `docs/architecture/005-op-quant-config.md`（OpQuantConfig 两阶段模型，ADR-009 依赖）
3. `docs/architecture/008-session-refactor.md`（Session API 参考）
4. `docs/workflow/phase-plan.md`（整体进度）

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 2,496 passed
`test_golden_equiv.py` 有 26 个预存在失败（golden data `.pt` 文件未 staging）
