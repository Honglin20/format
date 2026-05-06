# 功能点开发全生命周期

## 总流程

```
ADR（如需新架构决策）
  → 实现计划（docs/plans/YYYY-MM-DD-<name>.md）
  → 数学推导（如需，docs/verification/NNN-<name>.md）
  → 测试用例（先写，确保失败）
  → 实现代码
  → 自测通过（pytest -x）
  → Review agent（见 principles/review-gate.md）
  → 修复 Critical/Major 问题
  → Commit
  → 更新 CURRENT.md
  → 子任务结束信号
```

每一步不可跳过，不可逆序。

## 各阶段详解

### 阶段 0：ADR（如需）

- 涉及新的架构决策（新增模块、改变接口模式、引入新概念）时，先在 `docs/architecture/` 写 ADR
- ADR 编号递增：`NNN-<title>.md`
- 只写不可逆的决策，不写实现细节

### 阶段 1：实现计划

- 在 `docs/plans/YYYY-MM-DD-<taskname>.md` 创建计划（≤200 行）
- 内容：目标和验收标准、涉及文件清单（带路径）、子任务 checklist（有序）
- 计划**先于代码**

### 阶段 2：数学推导（如需）

- 涉及量化数值正确性时，先在 `docs/verification/` 写推导文档
- 格式原理 → 手工推导中间值 → 写出期望值
- 参考 `principles/math-verification.md`

### 阶段 3：测试用例

- **先于实现代码**编写
- 确保运行失败（red phase）
- 覆盖：正向路径 + 负面测试 + 边界值
- 量化测试参考 `standards/quantization-testing.md`

### 阶段 4：实现代码

- 使测试通过（green phase）
- 遵循 `standards/api-design.md` 中的 API 约束
- 遵循对应模块的 `standards/adding-*.md` 规范

### 阶段 5：自测

- 运行 `pytest src/tests/ -x`
- 当前 Phase 的所有测试必须通过
- 测试门标准见 `phase-plan.md`

### 阶段 6：Review

- 派遣 review agent（模板见 `principles/review-gate.md`）
- Critical / Major 问题在当前子任务内修复
- Minor 问题可记录为已知限制

### 阶段 7：Commit

- 测试 + 实现在同一 commit
- 格式见 `branching-commits.md`

### 阶段 8：状态更新

- 立即更新 `docs/status/CURRENT.md`
- 打勾已完成、更新下一步、更新必读文件清单
- 发出子任务结束信号

## 输入/输出

| 阶段 | 输入 | 输出 |
|------|------|------|
| ADR | 设计问题 | `docs/architecture/NNN-*.md` |
| 计划 | 需求描述 | `docs/plans/YYYY-MM-DD-*.md` |
| 推导 | 格式/算法原理 | `docs/verification/NNN-*.md` |
| 测试 | 推导的期望值 | `src/tests/test_*.py`（失败） |
| 实现 | 测试文件 | 代码（测试通过） |
| Review | 修改的文件 | Review 报告 |
| Commit | Review 通过 | Git commit |
| 状态 | Commit 完成 | 更新的 CURRENT.md |
