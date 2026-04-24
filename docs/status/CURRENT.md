# Current Task

**Task ID**: P2F（Phase 2 修正）
**Plan**: `docs/plans/2026-04-24-phase2-fix.md`
**Branch**: `feature/refactor-src`

---

## Progress

- [x] P2F-1：实现 `GranularitySpec` — commit `c991b43`
- [x] P2F-1 review 修复 — commit `6664476`
- [x] P2F-2：TransformBase + IdentityTransform + QuantScheme 升级 — commit `64e8db5`
- [x] P2F-3：FormatBase 加 `quantize()` 抽象方法 + 子类实现 — commit `8e74b3e`
- [x] P2F-4：`quantize(x, scheme)` 统一入口 + `quantize_elemwise_op` 改为 compat wrapper — commit `ec3b5ee`
- [x] P2F-5：消除 `src/quantize/` 中所有 MxSpecs 依赖 — commit `cd28f0c`
- [x] **P2F-6：更新所有等价性测试，改用新 QuantScheme API；全部通过**

---

## P2F-6 完成摘要

- 所有等价性测试改为使用 QuantScheme API（`scheme=` 而非 `mx_specs=`）
- `src/quantize/elemwise.py`：删除 `quantize_elemwise_op`、`_format_from_mx_specs` compat wrapper；`quantize()` 加 `scheme=None` guard
- `src/quantize/bfloat_quantize.py`：删除 `quantize_bfloat_from_specs` compat wrapper
- `src/quantize/vector.py`：删除 `mx_specs`/`round_mode` 参数，简化 `_dispatch_quantize`
- `src/quantize/mx_quantize.py`：删除 `quantize_mx_op` compat wrapper
- `src/quantize/__init__.py`：只导出 `quantize`, `quantize_mx`, `quantize_bfloat`, `vec_quantize`
- `src/specs/` 目录整体删除
- `src/tests/test_specs_equiv.py` 删除
- `src/tests/fixtures.py`：从 MxSpecs fixtures 改为 QuantScheme fixtures
- `src/tests/_compat.py`：新建 `scheme_from_mx_specs()` helper（仅测试内部使用，转换 golden .pt 文件中的 mx_specs dict）
- 319 tests 全绿，生产代码零 MxSpecs 泄漏

---

## 下一步

P2F 全部完成。进入 Phase 3（算子层 `src/ops/`）。

---

## 断点续传必读文件

1. `src/scheme/quant_scheme.py`（全文）— QuantScheme 三轴核心
2. `src/quantize/elemwise.py`（全文）— quantize() 统一入口
3. `src/quantize/mx_quantize.py`（全文）— quantize_mx(scheme) MX 块量化
4. `src/tests/_compat.py`（全文）— golden reference mx_specs→scheme 转换
5. `docs/plans/2026-04-24-phase2-fix.md`（全文）— Phase 2 总计划

---

## 验收标准（P2F 全部完成条件）

```bash
# 1. 无 MxSpecs 泄漏（生产代码）
grep -r "MxSpecs\|mx_specs\|from.*specs import" src/quantize/ src/scheme/ src/formats/ --include="*.py" && echo "FAIL" || echo "OK"
# ✅ OK

# 2. 所有等价性测试通过
pytest src/tests/ -x -q
# ✅ 319 passed

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
# ✅ OK
```
