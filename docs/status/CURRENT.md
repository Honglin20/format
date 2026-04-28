# Current Task

**Task ID**: P8 — Format Precision Study (Task 1/11)
**Plan**: docs/plans/2026-04-28-format-study-plan.md
**Design**: docs/plans/2026-04-28-format-study-design.md
**Branch**: feature/refactor-src
**Tests baseline**: 1305 passed, 0 xfail

## Progress

- [x] **Task 1: Experiment script skeleton and user-facing API** ✅
- [x] **Task 2: Implement run_experiment() — single config runner** ✅
- [x] **Task 3: Part A — 8-bit format comparison** ✅
- [x] **Task 4: Part B — 4-bit format comparison** ✅
- [x] **Task 5: Part C — FP32 vs PoT scaling comparison** ✅
- [x] **Task 6: Part D — Transform analysis (SmoothQuant + Hadamard)** ✅
- [ ] Task 7: Block size sensitivity sweep
- [ ] Task 8: Table generation (6 tables)
- [ ] Task 9: Figure generation (11 figures)
- [ ] Task 10: Cleanup, defaults, and documentation

## 下一步（具体动作）

Implement Task 7: block size sensitivity sweep over {16, 32, 64, 128} for 8-bit and 4-bit MX formats.

## 断点续传必读文件

1. `examples/experiment_format_study.py`（全文，包含 run_part_a/b/c/d）
2. `docs/plans/2026-04-28-format-study-plan.md`（全文）
3. `docs/plans/2026-04-28-format-study-design.md`（实验矩阵）

## 关键经验记录

1. **Format 子类签名同步**：FormatBase 改签名后，所有子类的 `quantize()` 必须同步更新
2. **__eq__/__hash__ 必须实现**：任何可能用作 frozen dataclass 字段的 ABC，必须在 ABC 层声明 `@abstractmethod`
3. **KL 逐 slice 非逐 position**：`_compute_kl_divergence` 将 axis 排到首维后 reshape(n_slices, -1)
4. **autograd Function 参数计数**：新增 tensor 参数到 `apply()` 后，`backward()` 和 `symbolic()` 返回值数量必须同步 +1
5. **SmoothQuant 不可变**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法
6. **LookupFormat quantize() 不能少**：即使只有 `quantize_elemwise()` 不同，也必须 override `quantize()`（否则 ABC 无法实例化）
7. **NaN 污染 amax**：per_channel 路径中 `amax = max(|x|, dim=axis)` 会把 NaN 传播到整个 channel
8. **QuantSession 设计原则**：新层非替代 — `QuantSession` 是现有 API 之上的薄层
9. **compare_sessions fp32 baseline**：从第一个 session 的 `fp32_model` 自动提取
10. **PreScaleTransform 引用模式**：持有 scale tensor 引用（非拷贝），register_buffer 后再创建 Transform 以保证 load_state_dict 一致性
11. **LSQ 梯度流**：pre_scale 在 custom autograd Function 内部无梯度，必须在 module 外部手动应用：`module(x * pre_scale) / pre_scale`
12. **PoT 投影梯度下降**：每步 optimizer.step() 后 `pre_scale.data = 2**round(log2(scale))`，微小模型 per-tensor 效果有限，per-channel 更有价值
13. **_quantize_per_block 不转发 scale**：CalibrationSession 分配 _output_scale (amax) 给所有模块，但 _quantize_mx 的 scale 参数期望 shared exponent 而非 amax；per_block 必须忽略外部 scale 以避免形状不匹配
