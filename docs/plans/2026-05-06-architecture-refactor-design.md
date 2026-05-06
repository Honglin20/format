# Architecture Refactor: 四层依赖架构

## 动机

当前 `src/` 下 14 个包平级排列，存在 5 类结构性问题：

1. **概念碎片化**：量化核心概念被拆分到 `formats/`、`scheme/`、`quantize/`、`transform/`、`mapping/`、`context/` 六个包，但缺乏统一的依赖层级组织。
2. **循环依赖**：`formats/` ↔ `quantize/` 之间存在循环导入。
3. **反向依赖**：`ops/`（核心算子层）依赖 `analysis/`（工具分析层），方向错误——`ObservableMixin` 作为横切关注点被错误定位。
4. **死代码/单文件包/过小文件**：`config/` 空包、`mapping/` 单文件包、5 个文件 < 30 行。
5. **驱动层分散**：`session.py` 孤悬顶层，`quantize_model()`、`QuantizeContext` 分散在三个不同位置。

## 目标架构

```
工具层 (Tools)          calibration  analysis  pipeline  cost  viz  onnx
                          ↓ 使用
驱动层 (Integration)     session   模型生命周期编排 + 模块转换 + inline 拦截
                          ↓ 驱动
算子层 (Ops)             ops   QuantizedXxx autograd 算子
                          ↓ 调用
数学层 (Math)            quantize(tensor, scheme) → tensor
                          ↑ 三轴正交组合
                  format  ×  granularity  ×  transform
```

## 最终文件结构

```
src/
  formats/      数学层 · 数值表示（与 quantize 解耦）
  transform/    数学层 · 张量变换
  scheme/       数学层 · 三轴方案 + TransformBase
  quantize/     数学层 · 统一入口 quantize(tensor, scheme)

  ops/          算子层 · QuantizedXxx autograd 算子（依赖 observer/ 而非 analysis/）

  session/      驱动层 · 吸收 session.py + mapping/quantize_model + context/
                包含：QuantSession + QuantizeContext + quantize_model

  observer/     横切基础设施 · 从 analysis/ 提取
                包含：ObservableMixin + Observer + Events
                被 ops/ 和 analysis/ 同时依赖

  calibration/  工具层 · 不变
  analysis/     工具层 · 删除 export.py
  pipeline/     工具层 · protocol.py 并入 runner.py
  cost/         工具层 · defaults.py + device.py 合并
  viz/          工具层 · save.py 并入 figures.py
  onnx/         工具层 · 不变

  _utils/       私有工具（下划线 = 非公共 API）
  tests/
```

## 变更清单

### 删除
- `src/config/` — 空包，全仓库零 import
- `src/mapping/` — 内容迁入 `session/`
- `src/context/` — 内容迁入 `session/`
- `src/analysis/export.py` — 5 行 docstring stub，零代码

### 新增
- `src/session/` — 吸收 `session.py` + `mapping/quantize_model.py` + `context/`
- `src/observer/` — 吸收 `analysis/events.py` + `analysis/mixin.py` + `analysis/observer.py`
- `src/_utils/` — 吸收跨包零散工具函数（`slicing.py` 等）

### 合并
- `context/_state.py` + `_stack.py` → `session/_stack.py`
- `cost/defaults.py` + `device.py` → `cost/device.py`
- `pipeline/protocol.py` → 并入 `pipeline/runner.py`
- `viz/save.py` → 并入 `viz/figures.py`

### API 级变更（仅一处）
- 打破 `formats/` ↔ `quantize/` 循环依赖：`FormatBase` 中依赖 `_quantize_elemwise_core` 的方法改为由 format 自身实现低阶 mantissa 量化操作，`quantize/` 仅做编排。

## 依赖验证

| 层 | 包 | 依赖 | 被依赖 |
|----|-----|------|--------|
| Math | scheme | (none) | 12 consumers |
| Math | transform | scheme | 4 consumers |
| Math | formats | scheme | cost, pipeline, quantize |
| Math | quantize | formats, scheme | ops |
| Ops | ops | quantize, scheme, observer | session, cost |
| Integration | session | ops, calibration, analysis, observer | pipeline |
| Cross-cut | observer | scheme | ops, analysis, session |
| Tools | calibration | scheme, transform | session, pipeline |
| Tools | analysis | observer, scheme | session, pipeline, viz |
| Tools | pipeline | session, calibration, analysis, formats, viz, scheme, transform | (none) |
| Tools | cost | formats, ops, scheme, transform | session |
| Tools | viz | analysis | pipeline |
| Tools | onnx | scheme | ops, session |

## 不变的部分

- 所有现有类名和公共 API 不变（除 formats/quantize 解耦涉及的低阶方法调整）
- 测试文件位置不变
- `formats/`、`transform/`、`scheme/`、`quantize/`、`ops/`、`calibration/`、`pipeline/`、`cost/`、`viz/`、`onnx/` 内部文件保持现有分解
