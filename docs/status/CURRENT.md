# Current Task

**Task ID**: P8.X — QuantSession 统一 API（Phase 8 研究能力扩展）
**Next**: P5 — LSQ / PACT 可学习量化参数 或 P1 — Bias Correction / CLE
**Branch**: feature/refactor-src
**Tests baseline**: 1265 passed, 0 xfail（+30 format registry, +8 scale persistence）

## QuantSession 完成状态

- [x] CalibrationSession 重构（上下文管理器 + auto-assign，CalibrationPipeline 向后兼容）
- [x] AnalysisSession = AnalysisContext 别名
- [x] Comparator / compare_models / compare_sessions（e2e 对比工具）
- [x] QuantSession 统一 API（session.py，34 tests）
- [x] README 更新（QuantSession 推荐用法 + API 方法表）

## QuantSession API 总结

```python
session = QuantSession(model, cfg, calibrator=..., observers=..., keep_fp32=True)
session.use_fp32() / session.use_quant()  # 模式切换
session(x)                                 # 推理
session.calibrate()                        # → CalibrationSession 上下文管理器
session.analyze()                          # → AnalysisContext 上下文管理器
session.compare(dl, eval_fn)               # 自动 fp32 vs quant 对比
session.comparator()                       # → Comparator 手动对比
session.export_onnx(path, dummy_input?)    # ONNX 导出（自动记录上次推理输入）
session.clear_scales()                     # 清除 _output_scale buffer
session.train() / .eval() / .parameters() / .state_dict()  # 委托
```

## Phase 8 所有完成项

- [x] 8A.1 Hadamard Transform（18 tests）
- [x] 8B.1 Scale Strategy（20 tests）
- [x] 8B.2 Calibration Pipeline（15 tests）
- [x] 8A.2 SmoothQuant Transform（29 tests）
- [x] 8B.3 Scale Persistence（20 tests）— save_scales/load_scales 磁盘持久化
- [x] P3 NF4 / Lookup Table Format（51 tests）
- [x] QuantSession 统一 API（34 tests）

## 下一步

P5 — LSQ / PACT 可学习量化参数，或 P1 — Bias Correction / CLE。

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `src/session.py`（QuantSession 实现）
3. `src/analysis/e2e.py`（Comparator / compare_models / compare_sessions）
4. `src/calibration/pipeline.py`（CalibrationSession 上下文管理器）
5. `~/.claude/projects/.../memory/format-research-roadmap.md`（P4/P5 详细说明）

## 关键经验记录

1. **Format 子类签名同步**：FormatBase 改签名后，所有子类的 `quantize()` 必须同步更新
2. **__eq__/__hash__ 必须实现**：任何可能用作 frozen dataclass 字段的 ABC，必须在 ABC 层声明 `@abstractmethod`
3. **KL 逐 slice 非逐 position**：`_compute_kl_divergence` 将 axis 排到首维后 reshape(n_slices, -1)
4. **autograd Function 参数计数**：新增 tensor 参数到 `apply()` 后，`backward()` 和 `symbolic()` 返回值数量必须同步 +1
5. **SmoothQuant 不可变**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法
6. **LookupFormat quantize() 不能少**：即使只有 `quantize_elemwise()` 不同，也必须 override `quantize()`（否则 ABC 无法实例化）
7. **NaN 污染 amax**：per_channel 路径中 `amax = max(|x|, dim=axis)` 会把 NaN 传播到整个 channel
8. **QuantSession 设计原则**：新层非替代 — `QuantSession` 是现有 API 之上的薄层，`calibrate()`/`analyze()` 返回底层上下文管理器，用户可自由嵌套
9. **compare_sessions fp32 baseline**：从第一个 session 的 `fp32_model` 自动提取，用户无需额外传 fp32 session
10. **naming convention to code mbits**：`fpN_eXmY` 中 Y 是 actual mantissa bits，FPFormat.mbits = Y + 2（sign + implicit）；auto-parser 需加 2 转换
