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
- [ ] **P2F-6：更新所有等价性测试，改用新 QuantScheme API；全部通过**

---

## P2F-5 完成摘要

- `quantize_mx(A, scheme, ...)` — 新 QuantScheme 驱动 API，带 granularity/transform 验证
- `quantize_bfloat(x, scheme, backwards_scheme, allow_denorm)` — 新 QuantScheme 驱动 API
- `vec_*` 系列函数新增 `scheme` 参数，`_dispatch_quantize` 统一分发
- 旧签名保留为 compat wrapper（`quantize_mx_op`, `quantize_bfloat_from_specs`）
- `src/quantize/` 不再 import `src.specs.specs`
- Review 发现 3 Critical + 4 Major 问题全部修复
- 18 个新测试（`test_scheme_api.py`），329 总测试全绿

---

## 下一步（具体动作）

P2F-6：更新所有等价性测试，改用新 QuantScheme API；删除或移入 compat 层的 `src/specs/specs.py`；验证 `grep -r "MxSpecs\|mx_specs\|from.*specs import" src/` 无命中。

---

## 断点续传必读文件

1. `src/tests/test_scheme_api.py`（全文）— P2F-5 新增的 QuantScheme API 测试
2. `src/quantize/mx_quantize.py`（201-280 行）— quantize_mx + quantize_mx_op compat
3. `src/quantize/bfloat_quantize.py`（全文）— quantize_bfloat + quantize_bfloat_from_specs compat
4. `src/quantize/vector.py`（全文）— scheme 参数 + _dispatch_quantize
5. `docs/plans/2026-04-24-phase2-fix.md`（129-143 行）— P2F-6 计划

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
