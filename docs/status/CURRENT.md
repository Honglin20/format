# Current Task

**Task ID**: P3.1（Matmul 家族 + Phase 3 基础设施）
**Plan**: `docs/plans/2026-04-24-phase3.md`
**Branch**: `feature/refactor-src`

---

## Progress

### Phase 2（已完成，终态提交 `bcf4031`）

- [x] P2F-1 ~ P2F-6：Phase 2 三轴扶正全完成
- [x] P2F-7：Phase 2 review 后缺陷收口（2C+2M+1m，commit `bcf4031` + `04fb902`）

### Phase 3（当前）

- [x] P3.0：P2F-7 收口
- [x] **P3.1-a**：`OpQuantConfig` 数据类 — commit `e1e6800` + review fix `012eea5`
- [x] **P3.1-b**：`ObservableMixin` + `QuantEvent` + `ObserverBase` / `SliceAwareObserver` + `iter_slices` — commit `8c02240` + review fix `7661dc8`
- [x] **P3.1-c**：`_compat.py::op_config_from_mx_specs` Linear 适配器 — commit `87db188` + fix `73feb9f`（block_axis、output pipeline、_bp key behavior）
- [x] **P3.1-d**：`QuantizedLinear` + `LinearFunction` — commit `945d817`（42 equivalence tests bit-exact）
- [x] **P3.1-e**：`quantized_matmul` + `quantized_bmm` — commit `ac6e02f`（68 total equivalence tests bit-exact）
- [ ] **P3.2**：Conv 家族（Conv1d/2d/3d + TransposeConv{1,2,3}d）
- [ ] **P3.3**：Norm 家族（BatchNorm / LayerNorm / GroupNorm）
- [ ] **P3.4**：激活 / Softmax / AdaptiveAvgPool
- [ ] **P3.5**：Elementwise / SIMD / Vector ops
- [ ] **P3.6**：`src/mapping/quantize_model` 模块替换入口 + 端到端 small model 测试

**不做**：RNN 家族。

等价性门槛：**bit-exact**（`torch.equal`，dither 固定 seed；不允许 atol/rtol）。

---

## 下一步（具体动作）

进入 **P3.2**：Conv 家族。先读 `docs/plans/2026-04-24-phase3.md` §P3.2 和 `mx/conv.py`，创建 `src/ops/conv.py`（Conv1d/2d/3d + TransposeConv）+ 等价性测试。

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-phase3.md`（全文）— Phase 3 子阶段定义
2. `docs/architecture/005-op-quant-config.md`（全文）— OpQuantConfig 设计
3. `src/ops/linear.py`（全文）— P3.1-d 实现参考（forward 保存 post-elemwise 的模式）
4. `src/tests/_compat.py`（全文）— op_config_from_mx_specs 适配器（含 linear + matmul 路径）
5. `src/scheme/op_config.py`（全文）— OpQuantConfig 数据类

---

## 验收门

```bash
pytest src/tests/ -x -q    # 531 tests 全绿
grep -rn "from.*mx" src/ops/ src/analysis/ src/mapping/   # 应无命中
```
