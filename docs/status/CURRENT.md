# Current Task

**Task ID**: P8.V1 — Format Study Verification Script
**Plan**: docs/plans/2026-04-29-format-study-verification.md
**Branch**: feature/refactor-src
**Tests baseline**: 1322 passed, 0 failures（含 12 个 verification tests）

## Progress

- [x] Task 0: Create docs/verification/ directory + README
- [x] Task 1: int8 per_tensor derivation (001) + test
- [x] Task 2: int8 per_channel derivation (002) + test
- [x] Task 3: int4 per_tensor derivation (003) + test
- [x] Task 4: int4 per_channel derivation (004) + test
- [x] Task 5: fp8_e4m3 per_tensor derivation (005) + test
- [x] Task 6: nf4 per_channel derivation (006) + test
- [x] Task 7: Linear int8 per_channel forward derivation (007) + test
- [x] Task 8: SmoothQuant equivalence derivation (008) + test
- [x] Task 9: Hadamard self-inverse derivation (009) + test
- [x] Task 10: MSEScaleStrategy calibration derivation (010) + test
- [x] Task 11: QSNR/MSE analysis metrics derivation (011) + test
- [x] Task 12: QuantSession E2E derivation (012) + test
- [x] Task 13: Update CURRENT.md
- [x] Task 14: Full test suite run (1322 passed)

## 验证方法论

三层递进验证，所有期望值先手工推导再运行测试：
- **Layer 1** (001-006): Format.quantize() 逐元素正确性
- **Layer 2** (007-009): 算子 + Transform（Linear + SmoothQuant/Hadamard）
- **Layer 3** (010-012): 完整 Pipeline（QuantSession calibrate→analyze）

## 下一步

commit 所有 verification 文件，回到 Phase 8 P1 主线任务。

## 断点续传必读文件

1. `docs/verification/README.md`
2. `examples/test_format_study_verification.py`（1-520 行）
3. `docs/plans/2026-04-29-format-study-verification.md`

## 关键经验记录

1. **float32 bit-exact 陷阱**：手工推导的浮点值是十进制近似，torch.float32 实际值是二进制表示。`torch.equal` 要求 bit-exact，需用 torch 预计算获取精确 float32 值
2. **Hadamard 自逆性**：`1/sqrt(2)` 乘两次引入 ~6e-8 浮点舍入误差，`H(H(x))` 与 `x` 差 1-2 ULP。数学性质在实数域严格成立，测试用 `allclose(atol=1e-7)`
3. **quantize_model 只替换 children**：根模块为 bare nn.Linear 时不会被替换，需包在 nn.Sequential 等容器中
4. **MSEScaleStrategy axis**：axis=0 返回 shape (1, 2)，axis=-1 返回 shape (2, 1)，取决于输入的 dim
