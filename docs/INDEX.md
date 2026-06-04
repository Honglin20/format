# 文档总索引

## 目录

| 目录 | 内容 | 何时读 |
|------|------|--------|
| [principles/](principles/INDEX.md) | 开发准则 — 怎么做事（TDD、Review、文档规则） | 首次参与项目必读 |
| [standards/](standards/INDEX.md) | 开发规范 — 怎么做对（API设计、新增模块、量化测试） | 新增模块/写测试前 |
| [workflow/](workflow/INDEX.md) | 开发流程 — 按什么顺序做（功能生命周期、分支、Commit） | 开始新功能前 |
| [architecture/](architecture/INDEX.md) | 架构决策（ADR） | 理解设计意图时 |
| [plans/](plans/INDEX.md) | 实现计划 | 执行具体 task 前 |
| [reviews/](reviews/INDEX.md) | Review 报告 + Defect 修复记录 | 排查历史缺陷时 |
| [reference/](reference/INDEX.md) | 公式 / 外部参考 | 需要权威公式时 |
| [verification/](verification/README.md) | 数学验证文档 | 写量化测试推导前 |
| [harness/](harness/INDEX.md) | AgentHarness 工作流集成 — adapter、工作流模式、图表、result_type | bitx 与 AgentHarness 联合开发时 |
| [status/CURRENT.md](status/CURRENT.md) | 当前 task 断点 | **每次新终端启动必读** |

## 快速导航

| 我要做什么 | 读哪个 |
|------------|--------|
| 新终端启动，续接上次工作 | `status/CURRENT.md` |
| 开始一个全新功能 | `workflow/feature-lifecycle.md` |
| 新增一种量化格式 | `standards/adding-format.md` → `architecture/001-*.md` |
| 新增一个 Observer | `standards/adding-observer.md` → `architecture/002-*.md` |
| 新增一种 Transform | `standards/adding-transform.md` → `architecture/001-*.md` |
| 写量化测试用例 | `standards/quantization-testing.md` → `verification/README.md` |
| 提交代码前跑 E2E | `standards/e2e-testing.md`（三层 E2E 门，每次 commit 必跑） |
| 排查 E2E 回归 | `verification/e2e-regression-patterns.md`（回归模式库，对照全部 §） |
| 子任务做完了，准备收尾 | `principles/review-gate.md` → `workflow/subtask-lifecycle.md` |
| 提交代码 | `workflow/branching-commits.md` |
| 排查一个历史 bug / 回顾之前怎么修的 | `reviews/INDEX.md` |
| 查某个公式的定义 | `reference/INDEX.md` |
