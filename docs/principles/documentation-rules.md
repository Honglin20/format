# 文档读写规则

## 读规则（渐进式披露）

### 新终端启动

1. 读 `CLAUDE.md`（项目入口，~50行）
2. 读 `docs/status/CURRENT.md`（当前 task 断点）
3. 读 CURRENT.md 中"断点续传必读文件"清单（≤5 个文件）

**不要**在没有读 CURRENT.md 的情况下直接开始工作。
**不要**在启动时全量加载所有文档。

### 按需查找

需要什么信息，从 `docs/INDEX.md` → 分类 `INDEX.md` → 具体文件，逐级查找：

| 场景 | 查找路径 |
|------|----------|
| 开始新功能 | `docs/INDEX.md` → `workflow/feature-lifecycle.md` |
| 新增模块 | `docs/INDEX.md` → `standards/INDEX.md` → 对应规范 |
| 理解设计 | `docs/INDEX.md` → `architecture/INDEX.md` → 对应 ADR |
| 查公式 | `docs/INDEX.md` → `reference/INDEX.md` → 公式文件 |

## 写规则（文档驱动开发）

### 功能点开发前
- 必须在 `docs/plans/` 中创建实现计划（≤200 行）
- 计划文档先于代码，不允许"先实现后补文档"

### 涉及数学正确性时
- 先在 `docs/verification/` 中写推导文档
- 手工推导期望值
- 再写测试和代码

### 每次修复/变更后
- 在 `docs/reviews/` 或对应 plan 中追加修复记录
- 不允许"无记录修复"（除非是 typo 级别的单行修正）

### 每个子任务完成后
- 立即更新 `docs/status/CURRENT.md`：打勾、更新下一步、更新必读文件清单
- **不积累，不延后**
- CURRENT.md 是唯一可信的持久化状态

### 提交信息
- Commit message 遵循 `workflow/branching-commits.md` 格式
- 每次 commit 同步包含测试 + 实现
