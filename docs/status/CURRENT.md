# Current Task

**Task ID**: P5 — Learnable Pre-Scale（Phase 8 研究能力扩展）
**ADR**: docs/architecture/006-p5-learnable-pre-scale.md
**Plan**: docs/plans/2026-04-28-p5-pre-scale.md
**Branch**: feature/refactor-src
**Tests baseline**: 1305 passed, 0 xfail

## P5 子任务（全部完成 ✅）

- [x] **Task 1: PreScaleTransform**（src/transform/pre_scale.py + 22 tests，含 PoT）
- [x] **Task 2: Fix quantize_mx()**（ADR-001 合规：非 Identity transform 时委托 quantize()）
- [x] **Task 3: LayerwiseScaleOptimizer**（src/calibration/lsq_optimizer.py，9 tests，含 PoT）
- [x] **Task 4: QuantSession 集成**（initialize_pre_scales / optimize_scales，8 tests）
- [x] **Task 5: 内部 scale 固定**（_fix_internal_scales，per-channel amax buffer）
- [x] **Task 6: E2E 集成测试**（1 test，full pipeline）
- [x] **文档与示例更新**：
  - README.md — Transform 表 + LSQ 章节 + 示例 07 + 测试数量 1305
  - examples/07_pre_scale.py — PreScale + LSQ + PoT 完整示例
  - examples/00_comprehensive.py — Section 12（Pre-Scale + LSQ）+ 所有 tuple 语法修复
  - examples/06_transforms.py — tuple 语法修复
  - docs/architecture/006-p5-learnable-pre-scale.md — 判断标准全部打勾

## Phase 8 所有完成项

- [x] 8A.1 Hadamard Transform（18 tests）
- [x] 8B.1 Scale Strategy（20 tests）
- [x] 8B.2 Calibration Pipeline（15 tests）
- [x] 8A.2 SmoothQuant Transform（29 tests）
- [x] 8B.3 Scale Persistence（20 tests）— save_scales/load_scales 磁盘持久化
- [x] P3 NF4 / Lookup Table Format（51 tests）
- [x] QuantSession 统一 API（34 tests）
- [x] P5 Learnable Pre-Scale（40 tests）— PreScaleTransform + LSQ + PoT

## 下一步

P5 全部完成（1305 tests, +40）。下一步：
- P1 收尾：Bias Correction Transform + Cross-Layer Equalization（低优先级）
- P7: Auto Search / 自动化量化配置搜索（低优先级）

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/status/CURRENT.md`（本文件）
3. `docs/architecture/006-p5-learnable-pre-scale.md`（P5 ADR）

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
