# Current Task

**Task ID**: P8.1 — Transform + Calibration（全部完成）
**Plan**: docs/plans/2026-04-27-phase8-transform-calibration.md
**Branch**: feature/refactor-src
**Tests**: 1162 passed, 0 xfail

## 完成状态

- [x] 8A.1 Hadamard Transform（`3fad6e3` — 18 tests）
- [x] 8B.1 Scale Strategy（`edb8894` → `c1b51d5` review fixes — 20 tests）
- [x] 8B.2 Calibration Pipeline（`22a81ee` — 15 tests）
- [x] 8A.2 SmoothQuant Transform（`d1cfd81` — 29 tests）
- [x] 8B.3 Scale Persistence（`f3a9999` — 12 tests）

## 范围外（未实现）

- CLE（Cross-Layer Equalization）
- Bias Correction
- 其他 ops（Conv/Norm/Activation）的 scale passthrough（仅 Linear 已集成）

## 下一步

- 继续 Phase 8 研究能力扩展（见 `format-research-roadmap.md`）
- 或派遣 review agent 检查 8B.3
- 或推送到 remote

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/plans/2026-04-27-phase8-transform-calibration.md`（设计决策）
3. `src/calibration/pipeline.py`（CalibrationPipeline + assign_scales）
4. `src/formats/base.py`（_quantize_per_channel scale 路径）
5. `src/ops/linear.py`（LinearFunction + QuantizedLinear scale passthrough）

## 关键经验记录

1. **Scale 线程要覆盖所有 Format 子类**：`FormatBase` 改签名后，`IntFormat`/`FPFormat`/`BF16Format`/`FP16Format` 的 `quantize()` 签名必须同步
2. **autograd Function 计数**：新增 tensor 参数到 `apply()` 后，`backward()` 和 `symbolic()` 的返回值数量必须同步 +1
3. **state_dict roundtrip**：`register_buffer` 的 buffer 会进入 `state_dict`；加载到无此 buffer 的新模型需先注册 dummy buffer 或 `strict=False`
