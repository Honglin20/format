# 数学验证 — 先推导，后验证

## 核心原则

**禁止"先跑实验再看结果是否合理"。** 涉及量化数值正确性的测试，必须先有手工推导的期望值，再跑代码验证。

## 流程

```
格式原理说明 → 手工推导中间值 → 写出期望值 → 代码验证（torch.equal）
```

每一步不可跳过，不可逆序。

## 推导文档

推导文档写入 `docs/verification/NNN-<short-name>.md`，命名规范见 `docs/verification/README.md`。

### 文档模板

```markdown
# NNN: <测试名称>

**对应测试函数**: test_<name>()
**验证层级**: Layer 1/2/3/4

## 给定数据
（列出本测试使用的张量及其值）

## 手工推导

### 步骤 1: ...
### 步骤 2: ...

## 期望值

expected = torch.tensor([...])

## 验证结果
- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
```

## 固定数据

所有测试用例共享同一组固定数据以确保可复现。具体数据见 `docs/verification/README.md`。
