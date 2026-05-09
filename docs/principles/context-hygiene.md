# Context 卫生 + 多 Agent 策略

## Context 卫生原则

- grep / diff 的长列表输出通过 agent 汇总为结论后返回主 context
- 主 context 只保留"当前子任务直接需要读"的文件内容
- 不要在主 context 里连续读取 5 个以上文件（用 Explore agent 代替）
- 每完成一个子任务就触发"子任务结束信号"，不在同一 context 内连续推进多个子任务

## Context 管理实操

| 操作 | 执行者 | 时机 |
|------|--------|------|
| 子任务完成信号 | Claude（文字说明） | 每个子任务 commit 后 |
| `/clear` 清空 context | **用户** | Claude 发出信号后，用户决定 |
| 新对话开始 | **用户** | 跨大子任务时推荐 |
| 使用 agent 代理大量读取 | Claude | 需要大量文件读取时 |

**关键原则**：每次新对话/清空 context 后，必须从 CURRENT.md 的"断点续传必读文件"重新加载状态，不依赖 context 历史。CURRENT.md 是唯一可信的持久化状态。

## Agent 选择

| 场景 | Agent 类型 | 原因 |
|------|-----------|------|
| 探索代码库 | `Explore` | 大量文件读取不污染主 context |
| 架构决策 | `Plan` | 系统性推理，产出结构化计划 |
| 代码审查 | `general-purpose` | 深度多维度检查 |
| 研究外部 API | `general-purpose` | Web 搜索 + 代码结合 |
| 独立多文件实现 | 并行 agent | 无依赖的子任务并行派遣 |
