# TDD — 测试驱动开发

## 核心原则

**测试先于或同步于实现。** 每个子任务的测试与实现代码在同一个 commit 中，绝不允许"先实现后补测试"。

## 节奏

```
写失败测试 → 实现 → 测试通过 → commit → review agent
```

## 等价性测试

从 `mx/` 迁移的代码：
- 先写 `assert torch.equal(mx_output, src_output)`
- 再实现 src 函数
- **Phase 3 及之前的等价性必须 bit-exact，不允许 `allclose` 宽松匹配**
- Dither 固定 seed 确保可复现

后续 Phase（Phase 4+）数值容忍度在对应计划文档中明示。

## 测试命名

用名字表达行为，不是表达错误类型：

```python
# ✓ 好的命名
def test_per_channel_rejects_string_axis():
def test_quantize_1d_vector_preserves_shape():

# ✗ 差的命名
def test_per_channel_error():
def test_quantize_edge():
```

## 负面测试覆盖

每一个 `raise` 点必须各自至少一条 `pytest.raises` 测试：
- 每个 `__post_init__` 里的 `raise`
- 每个工厂方法的类型/值守卫
- 每个公共 API 的错误分支

断言必须包含 `match=` 关键字子串，验证错误信息的可读性。

**新增任何 `raise` 必须伴随新增至少一条负面测试** — 在 commit 里同步提交，不得拆到下一个 subtask。
