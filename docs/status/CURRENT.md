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
- [ ] **P2F-5：消除 `src/quantize/` 中所有 MxSpecs 依赖**
- [ ] P2F-6：更新所有等价性测试，改用新 QuantScheme API；全部通过

---

## P2F-4 完成摘要

- 新增 `quantize(x, scheme, allow_denorm=True)` 统一入口，遵循 ADR-001 三步流程
- `quantize_elemwise_op` 重写为 compat wrapper，内部构造 QuantScheme 再调用 `quantize()`
- 新增 `_format_from_mx_specs` 辅助函数从 MxSpecs dict 推导 FormatBase
- `FormatBase.quantize()` 增加 `allow_denorm` 参数，支持 `bfloat_subnorms=False` 语义
- `_format_from_mx_specs` 为 bfloat=16 返回 `BFloat16Format`（保留硬件快捷路径）
- 23 个新测试，311 总测试全绿

---

## 下一步（具体动作）

P2F-5：消除 `src/quantize/mx_quantize.py`、`src/quantize/bfloat_quantize.py`、`src/quantize/vector.py` 中的 MxSpecs 依赖。将 `quantize_mx_op(A, mx_specs, ...)` 改为 `quantize_mx(A, scheme, ...)` 等。

---

## 断点续传必读文件

1. `src/quantize/mx_quantize.py`（全文）— P2F-5 主要修改目标
2. `src/quantize/bfloat_quantize.py`（全文）— P2F-5 修改目标
3. `src/quantize/vector.py`（全文）— P2F-5 修改目标
4. `docs/plans/2026-04-24-phase2-fix.md`（100-145 行）— P2F-5 计划
5. `src/quantize/elemwise.py`（全文）— 刚完成的 quantize() 入口

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
