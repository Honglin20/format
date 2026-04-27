# Current Task

**Task ID**: P8.X — Phase 8 研究能力扩展（P1 Transform + P2 Calibration + P3 NF4 已完成）
**Next**: P4 — 参数化格式注册 或 P5 — LSQ / PACT 可学习量化参数
**Branch**: feature/refactor-src
**Tests baseline**: 1207 passed, 0 xfail

## Phase 8 完成状态

- [x] 8A.1 Hadamard Transform（18 tests）
- [x] 8B.1 Scale Strategy（20 tests）
- [x] 8B.2 Calibration Pipeline（15 tests）
- [x] 8A.2 SmoothQuant Transform（29 tests）
- [x] 8B.3 Scale Persistence（12 tests）
- [x] **P3 NF4 / Lookup Table Format（45 tests）**

## P3 实现总结

采用 `LookupFormat(FormatBase)` 子类继承设计，不修改 `FormatBase`：
- `LookupFormat` — 通用 LUT 量化，`quantize_elemwise()` 最近邻搜索
- `NF4Format(LookupFormat)` — QLoRA 16 级非对称正态分布优化 levels
- PER_TENSOR / PER_CHANNEL / PER_BLOCK 三种粒度均直接可用（不改 dispatch 代码）
- Round mode 仅支持 "nearest"，其他 raise ValueError
- NaN 保留，Inf 饱和到边界 level

## 优先级调整

CLE（Cross-Layer Equalization）和 Bias Correction 降为最低优先级，暂不实施。

## 下一步：P4 — 参数化格式注册

当前添加新 fp 格式（如 `fp5_e3m1`）需要写 Python 类。目标：一行注册。

```python
register_float_format("fp5_e3m1", ebits=3, mbits=1)
# 或 from_str 自动解析命名约定
FormatBase.from_str("fp5_e3m1")  # 自动解析 e3m1
```

或 P5 — LSQ / PACT 可学习量化参数（scale 作为 nn.Parameter）。

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/architecture/001-three-axis-quant-scheme.md`（FormatBase 接口规范）
3. `src/formats/lookup_formats.py`（LookupFormat / NF4Format 新实现）
4. `src/formats/registry.py`（_init_default_formats 注册方式）
5. `~/.claude/projects/.../memory/format-research-roadmap.md`（P4/P5 详细说明）

## 关键经验记录

1. **Format 子类签名同步**：FormatBase 改签名后，所有子类的 `quantize()` 必须同步更新
2. **__eq__/__hash__ 必须实现**：任何可能用作 frozen dataclass 字段的 ABC，必须在 ABC 层声明 `@abstractmethod`
3. **KL 逐 slice 非逐 position**：`_compute_kl_divergence` 将 axis 排到首维后 reshape(n_slices, -1)
4. **autograd Function 参数计数**：新增 tensor 参数到 `apply()` 后，`backward()` 和 `symbolic()` 返回值数量必须同步 +1
5. **SmoothQuant 不可变**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法
6. **LookupFormat quantize() 不能少**：即使只有 `quantize_elemwise()` 不同，也必须 override `quantize()`（否则 ABC 无法实例化）。默认 dispatch 通过 `super().quantize()` 即可
7. **NaN 污染 amax**：per_channel 路径中 `amax = max(|x|, dim=axis)` 会把 NaN 传播到整个 channel
