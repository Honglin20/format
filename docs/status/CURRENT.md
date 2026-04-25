# Current Task

**Task ID**: Phase 3 Review — 缺陷记录与修复规划  
**Review文档**: `docs/plans/2026-04-25-phase3-review.md`  
**Plan**: `docs/plans/2026-04-24-phase3.md`  
**Branch**: `feature/refactor-src`（当前工作分支 `claude/review-phase-3-code-q5A6h`）

---

## Phase 3 实际完成状态

- [x] P3.0: P2F-7 缺陷收口
- [x] P3.1: OpQuantConfig + ObservableMixin + Linear/MatMul/BMM
- [x] P3.2: Conv + ConvTranspose（44 tests）
- [x] P3.3: BatchNorm/LayerNorm/GroupNorm/RMSNorm（32 tests）
- [x] P3.4: Activation/Softmax/AdaptiveAvgPool2d（bit-exact）
- [ ] **P3.5: Elemwise/SIMD/Vector ops（未实现）**
- [ ] **P3.6: mapping + 端到端验证（未实现）**

---

## Review 发现的必须修复项（进入 Phase 4 前）

### C1（阻断 Phase 4）：`_emit()` 未在任何算子中调用

所有算子继承 ObservableMixin 但没有调用 `_emit()`，Phase 4 的 AnalysisContext 挂载 observer 也不会收到任何事件。在所有算子的量化关键点埋入 `_emit()` 调用是 Phase 3 骨架的核心职责之一。

**修复前须与用户确认设计**：
- `_emit()` 应在 `QuantizedXxx.forward()` 中调用（持有 self），而非 `XxxFunction.forward()` 中
- 需定义统一的辅助方法以减少重复（见 review 文档 C1 部分）

### M2（阻断 Phase 4 分析正确性）：`iter_slices` PER_BLOCK 忽略 `block_axis`

`src/analysis/slicing.py` 对 PER_BLOCK 模式始终切 last dim，但 Conv 等算子的 block_axis=1。修复方案已在 review 文档 M2 中说明。

---

## 待讨论设计决策（与用户）

### M1：Activation/Softmax/Pool 接口不一致

当前 Activation/Softmax/Pool 使用 `inner_scheme: QuantScheme` 而非 `cfg: OpQuantConfig`，与其他算子族不一致，影响 P3.6 mapping 和 Phase 4 analysis 的统一性。

两个选项：
- A（推荐）：为这些算子补加薄包装，统一接口为 OpQuantConfig
- B：接受双轨制，在 mapping 和 analysis 中做适配，文档化例外

**需用户决策后再实现 P3.5/P3.6。**

---

## 待完成工作（按优先级）

1. **讨论并决策 M1**（接口方案）
2. **修复 C1**（_emit 接入）— 须先确定哪一层调用
3. **修复 M2**（iter_slices block_axis）— 小修改，直接做
4. **修复 m3**（GELUFunction inner_scheme=None 条件）
5. **修复 m1**（QuantizedLinear passthrough 缓存）
6. **修复 M3**（_compat matmul round_key）
7. **清理 m5**（src/ops/nn/ 空目录）
8. **P3.5**: Elemwise/SIMD/Vector ops + test_ops_equiv_elemwise.py
9. **P3.6**: src/mapping/quantize_model.py + test_e2e_small_model.py

---

## 下一步（具体动作）

与用户讨论 M1（接口一致性方案选择），确认后：
1. 修复 M2（iter_slices）
2. 讨论 C1 的 _emit 调用层次
3. 推进 P3.5 → P3.6

---

## 断点续传必读文件

1. `docs/plans/2026-04-25-phase3-review.md`（全文）— 本次 review 所有缺陷和建议
2. `docs/plans/2026-04-24-phase3.md`（P3.5/P3.6 章节）— 未完成子任务定义
3. `src/analysis/slicing.py`（全文）— M2 修复点
4. `src/analysis/mixin.py`（全文）— C1 修复参考
5. `src/ops/linear.py`（全文）— C1 修复样板参考

---

## 关键经验记录（Review 结论）

1. **`_emit()` 调用层次设计**：不能在 `autograd.Function.forward()` 中调用（静态方法无 self），必须在 `QuantizedXxx.forward()` 中调用，Function 返回后再 emit。这意味着 forward 中 Function.apply() 前后需要额外逻辑来捕获量化前后张量对。
2. **Norm 双配置模式的隐患**：`inner_scheme` 与 `cfg.input/weight/bias` 指向相同 scheme，是"约定"而非"约束"，文档应明确声明。
3. **iter_slices PER_BLOCK 的 block_axis 缺陷**：该缺陷在当前 Phase 3 中无实际影响（_emit 从未调用），但在 Phase 4 激活后立即暴露。属于"隐式定时炸弹"，须在 Phase 4 前修复。
4. **P3 完成度约 70%**：P3.5 + P3.6 + C1 + M2 是进入 Phase 4 的前置条件。
