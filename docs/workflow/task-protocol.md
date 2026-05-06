# TASK 协议 — 断点续传规范

## 新终端启动流程

```
1. 读 CLAUDE.md（项目入口）
2. 读 docs/status/CURRENT.md（当前 task 断点）
3. 只读 CURRENT.md 里"断点续传必读文件"清单中的文件（≤5 个）
4. 若 CURRENT.md 有"待讨论设计决策"，先与用户确认后再继续
5. 确认当前 task 和下一步后，继续工作
```

**不要**在没有读 CURRENT.md 的情况下直接开始工作。

## 启动新 task 时

1. 在 `docs/plans/YYYY-MM-DD-<taskname>.md` 创建实现计划（≤200 行）
2. 更新 `docs/status/CURRENT.md`

## 完成每个子任务后

严格遵循 `subtask-lifecycle.md` 的步骤 4-5：立即更新 CURRENT.md + 发出结束信号。

## CURRENT.md 固定格式

```markdown
# Current Task

**Task ID**: <Phase>-<编号>
**Plan**: docs/plans/YYYY-MM-DD-<name>.md
**Branch**: feature/refactor-src

## Progress

- [x] 子任务 1（已完成）
- [x] 子任务 2（已完成）
- [ ] **子任务 3（进行中）**
- [ ] 子任务 4

## 待讨论设计决策（如有）

- [ ] 决策 A：<描述选项 A vs B，以及影响范围>

## 下一步（具体动作）

<一句话，精确到函数/文件/行号级别>

## 断点续传必读文件

1. `src/scheme/quant_scheme.py`（全文）
2. `src/formats/base.py`（全文）
3. ...

## 关键经验记录

<跨任务复用的发现，每条一句话>
```

## CURRENT.md 更新原则

- 子任务完成 → 立即更新，不批量
- 只列真正需要在新终端读取的文件（≤5 个），带行号范围
- 关键经验记录：跨任务复用的发现，不是当前子任务的流水账
