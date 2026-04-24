# Current Task

**Task ID**: P2F（Phase 2 修正）
**Plan**: `docs/plans/2026-04-24-phase2-fix.md`
**Branch**: `feature/refactor-src`

---

## 背景（快速上下文）

`src/` 已完成 Phase 2 初版（量化核心迁移），但架构偏离了三轴（format+granularity+transform）目标：
- `QuantScheme` 无 transform 字段，format 是字符串非对象
- 量化函数仍依赖 39-key `MxSpecs` dict
- `Granularity` enum 是虚设，per-channel 无实现

Phase 2 修正目标：扶正三轴架构，让 `src/` 完全不依赖 `MxSpecs`。

---

## Progress

- [x] **P2F-1：实现 `GranularitySpec`**（替换 Granularity enum，加 block_size、channel_axis）— commit c991b43
- [x] **P2F-1 测试审查**：补充 21 个用例，修复 default_factory 和 channel_axis 验证 — commit 6664476
- [ ] **P2F-2：实现 `TransformBase` + `IdentityTransform`；升级 `QuantScheme` 加 transform 字段，format 改为 FormatBase 对象**
- [ ] P2F-3：升级 `FormatBase`，加 `quantize(x, granularity, round_mode)` 抽象方法
- [ ] P2F-4：实现各 Format 子类的 `quantize()` 方法，替换 `elemwise.py` 中的 if-elif 链
- [ ] P2F-5：升级 `mx_quantize.py` / `bfloat_quantize.py` / `vector.py` 消除 MxSpecs 依赖
- [ ] P2F-6：更新所有等价性测试，改用新 QuantScheme API；全部通过

---

## 下一步（具体动作）

**P2F-2**：创建 `src/scheme/transform.py`，实现 `TransformBase(ABC)` + `IdentityTransform`；
升级 `QuantScheme`：`format: str` → `format: FormatBase`，新增 `transform: TransformBase` 字段。
详见 `docs/plans/2026-04-24-phase2-fix.md` P2F-2 节。

---

## 断点续传必读文件

1. `CLAUDE.md`（全文）— 架构约束和 TASK 协议
2. `docs/plans/2026-04-24-phase2-fix.md`（全文）— P2F 完整计划
3. `docs/architecture/001-three-axis-quant-scheme.md`（全文）— 三轴接口规范
4. `src/scheme/quant_scheme.py`（全文）— 当前 QuantScheme（P2F-2 需修改）
5. `src/formats/base.py`（全文）— FormatBase（P2F-2/3 需修改）

---

## 验收标准（P2F 完成条件）

```bash
# 1. 无 MxSpecs 泄漏
grep -r "MxSpecs\|mx_specs\|from.*specs import" src/ && echo "FAIL" || echo "OK"

# 2. 所有等价性测试通过
pytest src/tests/ -x -q

# 3. QuantScheme 能表达三轴（smoke test）
python -c "
from src.scheme import QuantScheme
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec
from src.scheme.transform import IdentityTransform
fmt = FormatBase.from_str('fp8_e4m3')
s = QuantScheme(format=fmt, granularity=GranularitySpec.per_block(32), transform=IdentityTransform())
print('OK:', s)
"
```
