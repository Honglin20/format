# Current Task

**Task ID**: P2F（Phase 2 修正）
**Plan**: `docs/plans/2026-04-24-phase2-fix.md`
**Branch**: `feature/refactor-src`

---

## Progress

- [x] P2F-1：实现 `GranularitySpec`（替换 Granularity enum，加 block_size、channel_axis）— commit `c991b43`
- [x] P2F-1 review 修复：补充 21 个用例，修复 default_factory 和 channel_axis 验证 — commit `6664476`
- [x] P2F-2：TransformBase + IdentityTransform + QuantScheme 升级 — commit `64e8db5`
- [x] P2F-3：升级 `FormatBase`，加 `quantize(x, granularity, round_mode)` 抽象方法 — commit `8e74b3e`
- [ ] **P2F-4：实现各 Format 子类的 `quantize()` 方法，替换 `elemwise.py` 的 if-elif 链**
- [ ] P2F-5：消除 `src/quantize/` 中所有 MxSpecs 依赖
- [ ] P2F-6：更新所有等价性测试，改用新 QuantScheme API；全部通过

---

## P2F-3 完成摘要

- `FormatBase.quantize(x, granularity, round_mode)` 抽象方法 + 默认分发逻辑（PER_TENSOR→elemwise, PER_CHANNEL→amax+elemwise, PER_BLOCK→_quantize_mx）
- `__eq__`/`__hash__` 声明为 `@abstractmethod`，所有子类实现值相等性
- IntFormat/FPFormat.quantize() 委托 `super().quantize()`
- BFloat16Format/Float16Format 添加 `round_mode="even"+per_tensor` 硬件快捷路径
- 52 个新测试（等价性、边界值、快捷旁路、负 axis、hash 一致性）
- 288 总测试全绿

---

## 下一步（具体动作）

P2F-4：重写 `src/quantize/elemwise.py` 中的 `quantize_elemwise_op`，用 `QuantScheme` 替换 `MxSpecs` 依赖。新增统一入口 `quantize(x, scheme: QuantScheme) -> Tensor`。

---

## 断点续传必读文件

1. `src/quantize/elemwise.py`（全文）— P2F-4 主要修改目标
2. `src/formats/base.py`（全文）— 新增的 quantize() 默认分发逻辑
3. `docs/plans/2026-04-24-phase2-fix.md`（60-98 行）— P2F-4 计划
4. `src/scheme/quant_scheme.py`（全文）— QuantScheme 定义
5. `src/quantize/mx_quantize.py`（全文）— P2F-5 的前置理解

---

## 验收标准（P2F 全部完成条件）

```bash
# 1. 无 MxSpecs 泄漏
grep -r "MxSpecs\|mx_specs\|from.*specs import" src/ && echo "FAIL" || echo "OK"

# 2. 所有等价性测试通过
pytest src/tests/ -x -q

# 3. QuantScheme 三轴 smoke test
python -c "
from src.scheme import QuantScheme, GranularitySpec
from src.scheme.transform import IdentityTransform
from src.formats.base import FormatBase
s = QuantScheme(
    format=FormatBase.from_str('fp8_e4m3'),
    granularity=GranularitySpec.per_block(32),
    transform=IdentityTransform(),
)
print('OK:', s)
"
```
