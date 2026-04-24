# Current Task

**Task ID**: P2F-7（Phase 2 收口补丁，随后进入 Phase 3）
**Plan**: `docs/plans/2026-04-24-p2f7-findings.md` → `docs/plans/2026-04-24-phase3.md`
**Branch**: `feature/refactor-src`

---

## Progress

### Phase 2（已完成，终态提交 `bc9152c`）

- [x] P2F-1：`GranularitySpec`
- [x] P2F-2：`TransformBase` + `IdentityTransform` + `QuantScheme` 升级
- [x] P2F-3：`FormatBase.quantize()` 抽象方法 + 子类实现
- [x] P2F-4：`quantize(x, scheme)` 统一入口
- [x] P2F-5：`src/quantize/` 消除 MxSpecs 依赖
- [x] P2F-6：所有等价性测试迁移到 QuantScheme API（319 tests 全绿）

### Phase 2 review 后遗留缺陷（P2F-7 尚未修复）

2026-04-24 新一轮静态 review 发现 2 Critical + 2 Major + 3 Minor（详见 `docs/plans/2026-04-24-p2f7-findings.md`）：

- [ ] **P2F-7（进行中）**：
  - C1: `QuantScheme.__post_init__` 加 granularity 类型 guard
  - C2: `channel_axis` 负值行为文档化 + `FormatBase._quantize_per_channel` 加越界断言
  - M1: `QuantScheme.granularity` 默认值 docstring 补注（字段级）
  - M2: `BFloat16Format` / `Float16Format` 的 shortcut 路径 round_mode 校验加注释
  - m1: `quantize(x, scheme=None)` 的 docstring 补"scheme=None 语义"
  - + 新增 6 条 `pytest.raises` 负面测试（3 条 granularity 类型 / 3 条 negative axis）

### Phase 3（开发计划已定稿，未开工）

计划文档：`docs/plans/2026-04-24-phase3.md`。子阶段（按依赖推进）：

- [ ] P3.0：P2F-7 收口（见上）
- [ ] P3.1：Matmul 家族（Linear / Matmul / BMM）+ Phase 3 基础设施
- [ ] P3.2：Conv 家族（Conv1d/2d/3d + TransposeConv{1,2,3}d）
- [ ] P3.3：Norm 家族（BatchNorm / LayerNorm / GroupNorm）
- [ ] P3.4：激活 / Softmax / AdaptiveAvgPool
- [ ] P3.5：Elementwise / SIMD / Vector ops
- [ ] P3.6：`src/mapping/quantize_model` 模块替换入口 + 端到端 small model 测试

**不做**：RNN 家族（用户明确指示放弃 `mx/rnn.py` 的重建）。

等价性门槛：**bit-exact**（`torch.equal`，dither 固定 seed；不允许 atol/rtol）。

---

## 本次会话产出

- `docs/plans/2026-04-24-p2f7-findings.md`（新增）— Phase 2 review 后缺陷清单
- `docs/architecture/005-op-quant-config.md`（新增）— OpQuantConfig ADR
- `docs/architecture/002-observer-analysis.md`（更新）— 加入 SliceAwareObserver / iter_slices / group_map
- `docs/plans/2026-04-24-phase3.md`（新增）— Phase 3 开发计划（P3.0 ~ P3.6，RNN 排除）
- `CLAUDE.md`（更新）— QAT 进入范围 / §5.1 负面测试细化 / §5.2 验证漏斗 review 项 / §5.4 新增"跨对象一致性验证"规则 / §6 Phase 3 子阶段列表 / 分支说明改为 `feature/refactor-src`
- `docs/status/CURRENT.md`（本文件，更新）

---

## 下一步（具体动作）

**进入 P3.0（P2F-7 缺陷修复）**，按 `docs/plans/2026-04-24-p2f7-findings.md` §"P2F-7 落地清单"逐项落代码：

1. `src/scheme/quant_scheme.py:42-62` — `__post_init__` 加 `isinstance(granularity, GranularitySpec)` 类型 guard（在 round_mode 校验前）
2. `src/formats/base.py:114` — `_quantize_per_channel` 规范化 axis 后加越界断言
3. `src/tests/test_formats_equiv.py` — 追加 6 条 `pytest.raises` 负面测试
4. 文档微调：`GranularitySpec` / `QuantScheme` 字段 docstring
5. commit：`fix(scheme): P2F-7 close Phase 2 review findings`
6. 派 review agent（CLAUDE.md §5.2），检查验证漏斗覆盖
7. 若通过，进入 P3.1-a（`OpQuantConfig` 数据类）

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-p2f7-findings.md`（全文）— P2F-7 具体改法与测试清单
2. `docs/plans/2026-04-24-phase3.md`（全文）— Phase 3 子阶段定义与验收门
3. `docs/architecture/005-op-quant-config.md`（全文）— OpQuantConfig 设计（P3.1 起手必读）
4. `docs/architecture/002-observer-analysis.md`（全文）— Observer 设计（P3.1 基础设施必读）
5. `src/scheme/quant_scheme.py`（全文）— P2F-7 修改目标文件

---

## 验收门

```bash
# Phase 2 + P2F-7 完成条件
grep -r "MxSpecs\|mx_specs\|from.*specs import" src/quantize/ src/scheme/ src/formats/ --include="*.py" && echo FAIL || echo OK
pytest src/tests/ -x -q    # 应当全绿（含新增 P2F-7 的 6 条负面测试）

# Phase 3 每个子阶段结束条件
pytest src/tests/test_ops_equiv_<family>.py -x -q   # bit-exact 全绿
grep -rn "from.*mx" src/ops/ src/analysis/ src/mapping/   # 应无命中
```
