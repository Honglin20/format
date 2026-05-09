# Docs Restructure — 文档体系重构设计

**日期**: 2026-05-06
**状态**: 已确认

## 目标

将 CLAUDE.md（410行）拆解为渐进式披露的文档体系：
- CLAUDE.md ~50行，只保留启动流程 + 项目目标 + 不可违反边界 + 文档导航表
- docs/ 下按内容类型分 9 个目录，各自有 INDEX.md
- 所有开发准则、规范、流程、架构细节、Phase 计划从 CLAUDE.md 移出

## 新目录结构

```
docs/
├── INDEX.md                         # 总索引
├── principles/                      # 开发准则 — 怎么做事
│   ├── INDEX.md
│   ├── tdd.md                       # TDD 流程
│   ├── review-gate.md               # Review agent 门
│   ├── math-verification.md         # 数学验证：先推导后验证
│   ├── documentation-rules.md       # 文档读写规则
│   └── context-hygiene.md           # Context 卫生 + 多 agent
├── standards/                       # 开发规范 — 怎么做对
│   ├── INDEX.md
│   ├── api-design.md                # API 设计约束
│   ├── adding-format.md             # 新增 Format 规范
│   ├── adding-observer.md           # 新增 Observer / emit_fn 规范
│   ├── adding-transform.md          # 新增 Transform 规范
│   ├── onnx-export.md               # ONNX export 接入规范
│   └── quantization-testing.md      # 量化测试用例编写规范
├── workflow/                        # 开发流程 — 按什么顺序做
│   ├── INDEX.md
│   ├── feature-lifecycle.md         # 功能点开发全生命周期
│   ├── subtask-lifecycle.md         # 子任务生命周期（5步骤）
│   ├── task-protocol.md             # TASK 协议（断点续传）
│   ├── branching-commits.md         # 分支 + Commit 规范
│   └── phase-plan.md                # Phase 计划总览
├── architecture/                    # 架构决策（ADR，不变）
│   └── INDEX.md (新增)
├── plans/                           # 实现计划（清理 review/defect 移出）
│   └── INDEX.md (新增)
├── reviews/                         # Review 报告 + Defect 修复
│   └── INDEX.md (新增)
├── reference/                       # 公式/外部参考
│   └── INDEX.md (新增)
├── verification/                    # 验证文档
│   └── README.md (已有)
└── status/                          # 断点状态
    └── CURRENT.md (已有)
```

## 文件迁移映射

### CLAUDE.md 段落 → 新文件

| CLAUDE.md 段落 | 目标文件 |
|---|---|
| §3.1 三轴量化方案 | 已存在于 `architecture/001-*.md`，不迁移 |
| §3.2 OpQuantConfig | 已存在于 `architecture/005-*.md`，不迁移 |
| §3.3 Observer 模式 | 已存在于 `architecture/002-*.md`，不迁移 |
| §3.4 ONNX Export | 已存在于 `architecture/003-*.md`，不迁移 |
| §4 分支+Commit | `workflow/branching-commits.md` |
| §4 子任务生命周期 | `workflow/subtask-lifecycle.md` |
| §4 测试门 | `workflow/phase-plan.md` |
| §5.1 TDD 原则 | `principles/tdd.md` |
| §5.2 Review Agent 门 | `principles/review-gate.md` |
| §5.3 多 Agent 开发 | `principles/context-hygiene.md` |
| §5.4 API 设计约束 | `standards/api-design.md` |
| §6 Phase 计划 | `workflow/phase-plan.md` |
| §7 TASK 协议 | `workflow/task-protocol.md` |
| §8 文档索引 | `docs/INDEX.md` |

### 现有文件移动

| 原路径 | 新路径 |
|---|---|
| `docs/plans/2026-04-25-phase3-review.md` | `docs/reviews/2026-04-25-phase3-review.md` |
| `docs/plans/2026-04-24-p2f7-findings.md` | `docs/reviews/2026-04-24-p2f7-findings.md` |
| `docs/plans/2026-04-24-p2f2-review-findings.md` | `docs/reviews/2026-04-24-p2f2-review-findings.md` |
| `docs/plans/2026-04-25-defect-fix-specs.md` | `docs/reviews/2026-04-25-defect-fix-specs.md` |
| `docs/plans/2026-05-01-format-study-defect-fix.md` | `docs/reviews/2026-05-01-format-study-defect-fix.md` |
| `docs/issues/fix-quantization-logic.md` | `docs/reviews/fix-quantization-logic.md` |

### 新增文件（需从头编写）

共 25 个新文件：
- `docs/INDEX.md`
- `principles/` 6 个（INDEX + 5 内容）
- `standards/` 7 个（INDEX + 6 内容）
- `workflow/` 6 个（INDEX + 5 内容）
- `architecture/INDEX.md`
- `plans/INDEX.md`
- `reviews/INDEX.md`
- `reference/INDEX.md`

## 执行顺序

1. 创建所有新目录
2. 编写所有 INDEX.md 文件
3. 编写所有内容文件（principles/ → standards/ → workflow/）
4. 移动现有文件（plans/ → reviews/）
5. 精简 CLAUDE.md
6. Commit
