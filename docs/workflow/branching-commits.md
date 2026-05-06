# 分支 + Commit 规范

## 分支策略

- 主开发分支：`feature/refactor-src`（所有 src/ 重建工作的 long-lived branch）
- 单次 review / 多任务聚合分支：`claude/<short-desc>`，review 完成后 fast-forward 合入 `feature/refactor-src`
- 不得推送到 `master` 或 `main`

## Commit 格式

```
<type>(<scope>): <简短描述>

[可选正文：说明 why，不说 what]
```

### type

`feat` / `fix` / `refactor` / `test` / `docs` / `chore`

### scope

`scheme` / `formats` / `quantize` / `ops` / `analysis` / `mapping` / `onnx` / `docs`

### 示例

```
feat(scheme): add TransformBase + IdentityTransform to QuantScheme
refactor(quantize): replace MxSpecs dispatch with Format.quantize() Strategy
```

## Commit 内容约束

- 测试 + 实现在同一 commit
- 不允许"先实现后补测试"的拆分 commit
- Commit message 聚焦 why，不描述 what
