# Current Task

**Task ID**: P2F（Phase 2 修正）
**Plan**: `docs/plans/2026-04-24-phase2-fix.md`
**Branch**: `feature/refactor-src`

---

## Progress

- [x] P2F-1：实现 `GranularitySpec`（替换 Granularity enum，加 block_size、channel_axis）— commit `c991b43`
- [x] P2F-1 review 修复：补充 21 个用例，修复 default_factory 和 channel_axis 验证 — commit `6664476`
- [ ] **P2F-2：TransformBase + IdentityTransform + QuantScheme 升级（实现完成，review 发现问题，待修复后方可进入 P2F-3）**
- [ ] P2F-3：升级 `FormatBase`，加 `quantize(x, granularity, round_mode)` 抽象方法
- [ ] P2F-4：实现各 Format 子类的 `quantize()` 方法，替换 `elemwise.py` 的 if-elif 链
- [ ] P2F-5：消除 `src/quantize/` 中所有 MxSpecs 依赖
- [ ] P2F-6：更新所有等价性测试，改用新 QuantScheme API；全部通过

---

## P2F-2 待修复问题（进入 P2F-3 前必须全部解决）

详情见 `docs/plans/2026-04-24-p2f2-review-findings.md`。

### Critical（阻塞进入 P2F-3）

- **C1** `src/scheme/transform.py`：`TransformBase` 未将 `__eq__`/`__hash__` 声明为 `@abstractmethod`，自定义子类会静默破坏 `QuantScheme` 的 hash 等价性
- **C2** `src/scheme/quant_scheme.py:43-56`：`__post_init__` 未验证 `transform` 字段类型，`transform="invalid"` 静默通过构造
- **C3** `src/scheme/quant_scheme.py:62-65`：`per_channel(fmt, "floor")` 旧调用方式静默将 `"floor"` 当 `axis`，需加字符串类型守卫

### Major（建议同轮修复）

- **M1** `quant_scheme.py:34`：默认 `format="int8"` 无文档说明
- **M2** `test_formats_equiv.py`：缺测试：`IdentityTransform` 在 `QuantScheme` 中的 hash 一致性（两实例 hash 相同、set 去重）
- **M3** `granularity.py`：`channel_axis` 允许负值，需明确支持策略（允许则文档化，拒绝则加 `>= 0` 校验）
- **M4** `test_formats_equiv.py`：缺测试：自定义 `TransformBase` 子类在 `QuantScheme` 中的相等性 / hash 行为

---

## 下一步（具体动作）

修复 **C1**：在 `src/scheme/transform.py` 的 `TransformBase` 中，将 `__eq__` 和 `__hash__` 声明为 `@abstractmethod`；更新 `IdentityTransform` 的对应实现。  
随后依次修复 **C2**（`quant_scheme.py` 加 transform 类型检查 + 测试）、**C3**（`per_channel` 加字符串守卫 + 测试）、M1-M4。

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-p2f2-review-findings.md`（全文）— 所有问题的修复指南（含代码片段）
2. `src/scheme/transform.py`（全文）— C1 修复目标
3. `src/scheme/quant_scheme.py`（全文）— C2、C3、M1 修复目标
4. `src/scheme/granularity.py`（全文）— M3 修复目标
5. `src/tests/test_formats_equiv.py`（340-510 行）— 需补充 M2、M4 测试的区域

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
