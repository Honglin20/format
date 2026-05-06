# 新增 Transform 规范

## 架构位置

Transform 位于 `src/transform/`，接口定义见 `docs/architecture/001-three-axis-quant-scheme.md`。

## 量化流程

Transform 在量化流程中的固定位置：

```
transform.forward(x) → format.quantize(...) → transform.inverse(x_q)
```

## 步骤

### 1. 阅读 ADR

先读 `docs/architecture/001-three-axis-quant-scheme.md`，理解 Transform 在三轴方案中的角色。

### 2. 确定类型

| 类型 | 基类 | 说明 |
|------|------|------|
| 前处理变换 | `TransformBase` | 在量化前对数据做变换，量化后逆变换还原 |
| SmoothQuant | `SmoothQuantTransform` | 通过 per-channel scale 平滑 activation ↔ weight 的量化难度 |
| Hadamard | `HadamardTransform` | Hadamard 旋转矩阵，降低 outlier 影响 |
| PreScale | `PreScaleTransform` | 可学习的 per-tensor/per-channel scale（LSQ，见 ADR-006） |

### 3. 实现必须的方法

- `forward(x) -> Tensor` — 量化前变换
- `inverse(x_q) -> Tensor` — 量化后逆变换
- `__eq__` / `__hash__` — 必须实现（作为 frozen dataclass 字段）

### 4. 注册

无全局注册表（Transform 是实例，不是类型枚举）。只需在 `src/transform/` 下创建文件，从 `src/transform/__init__.py` 导出。

### 5. 测试

- 等价性测试：`torch.equal`（无 learnable 参数时）
- `forward → inverse` roundtrip 正确性
- per-tensor / per-channel 变体
- 负面测试（类型错误、shape 不匹配等）

### 6. 文档

- 数学推导写入 `docs/verification/`
- 实现计划写入 `docs/plans/`

## 接口契约

- `TransformBase` 必须声明 `__eq__` 和 `__hash__` 为 `@abstractmethod`
- `forward(x)` 和 `inverse(x_q)` 返回与输入相同 shape 的张量
- 无参数 Transform（如 `IdentityTransform`）使用单例模式
