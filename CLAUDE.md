# CLAUDE.md — 项目入口

## 启动流程（每次必执行）

1. 读本文件
2. 读 `docs/status/CURRENT.md`
3. 读 CURRENT.md 中"断点续传必读文件"清单的文件（≤5 个）

**不要**在没有读 CURRENT.md 的情况下直接开始工作。

---

## 项目目标

基于 microsoft/microxcaling（`mx/`）做增量式量化库重建，新代码在 `src/`。
核心：三轴量化方案（format + granularity + transform）+ 算子级配置 + 层级误差分析 + ONNX export + QAT。

---

## 不可违反的边界规则

1. `src/` 不得 `import mx`，不得依赖 `MxSpecs` dict
2. `mx/` 只读，仅通过公共 API 调用做等价性测试
3. 新增格式/granularity/transform 通过注册/实现抽象基类，不改量化核心函数
4. 测试先于实现：写失败测试 → 实现 → 通过 → commit → review agent
5. 子任务完成后立即更新 `docs/status/CURRENT.md`，不积累；CURRENT.md 是唯一持久化状态

---

## 场景 → 读什么

| 我要做什么 | 读哪个文件 |
|------------|-----------|
| 新终端续接上次工作 | `docs/status/CURRENT.md` |
| 开始一个全新功能 | `docs/workflow/feature-lifecycle.md` |
| 新增量化格式 | `docs/standards/adding-format.md` → `docs/architecture/001-*.md` |
| 新增 Observer | `docs/standards/adding-observer.md` → `docs/architecture/002-*.md` |
| 新增 Transform | `docs/standards/adding-transform.md` → `docs/architecture/001-*.md` |
| 写量化测试用例 | `docs/standards/quantization-testing.md` → `docs/verification/README.md` |
| 涉及数学正确性 | `docs/principles/math-verification.md` — 先推导，后验证 |
| 新增公共 API | `docs/standards/api-design.md` |
| 子任务做完了，准备收尾 | `docs/principles/review-gate.md` → `docs/workflow/subtask-lifecycle.md` |
| 提交代码 | `docs/workflow/branching-commits.md` |
| 排查历史缺陷 | `docs/reviews/INDEX.md` |
| 查公式定义 | `docs/reference/INDEX.md` |
| 了解整体进度 | `docs/workflow/phase-plan.md` |
| 文档读写规范 | `docs/principles/documentation-rules.md` |

---

## 快速参考

- **Branch**: `feature/refactor-src`（主开发），`claude/<desc>`（单任务），不推 master/main
- **Commit**: `<type>(<scope>): <描述>` — type: feat/fix/refactor/test/docs/chore
- **测试门**: `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q`（当前 1,416 passed）

---

## 完整文档索引

→ **[docs/INDEX.md](docs/INDEX.md)**
