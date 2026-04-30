"""
Format Study Configuration
===========================

修改实验只需改动此文件中的 ``STUDY_CONFIG`` 字典，无需修改 runner 代码。

----
每个 variant 支持的字段
----

============================  ========  ====================================================
字段                          必需      说明
============================  ========  ====================================================
``name``                      是        实验名称（用于结果 key）
``format``                    是        量化格式字符串
``granularity``               是        ``"per_tensor"`` | ``"per_channel"`` | ``"per_block"``
``axis``                      否        per_channel 时的轴（默认 -1，即特征维度）
``block_size``                否        per_block 时的块大小（默认 32）
``transform``                 否        变换类型：``"none"`` | ``"hadamard"`` | ``"smoothquant"``
``weight_only``               否        只量化权重，不量化输入/输出（NF4 应设为 ``true``）
``lsq_steps``                 否        LSQ 优化步数（0 = 不做 LSQ）
``lsq_pot``                   否        LSQ 约束 power-of-two
``lsq_lr``                    否        LSQ 学习率（默认 1e-3）
============================  ========  ====================================================

----
可用 format 字符串
----

- **整数**: ``"int8"``, ``"int4"``, ``"int2"``
- **浮点**: ``"fp8_e4m3"``, ``"fp8_e5m2"``, ``"fp6_e3m2"``, ``"fp6_e2m3"``, ``"fp4_e2m1"``
- **查找表**: ``"nf4"``
- **标准**: ``"bfloat16"``, ``"float16"``

----
添加新实验
----

**加一个 variant**：在对应 part 的 ``variants`` 列表里加一个 dict。
**加一个新 part**：在 ``STUDY_CONFIG`` 里新增一个 key，结构仿照已有 part。
**跳过某个 part**：把该 part 的 key 注释掉或删除即可。

----
Part 类型说明
----

================  ==============================================================
``"simple"``      基础对比实验：遍历 variants，每个跑一次 quantize→calibrate→analyze→evaluate
``"transform"``   变换研究：对每个 format 跑 None / Hadamard / SmoothQuant 三组对比，
                  然后自动选出 per-layer 最优变换组合
``"pot_scaling"`` PoT 缩放对比：每个 variant 跑两次（pot=False + pot=True），启用 LSQ
``"block_sweep"`` 块大小扫描：对指定 format 扫描 [16, 32, 64, 128] 四个块大小
================  ==============================================================
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 主配置 — 修改这里！
# ---------------------------------------------------------------------------

STUDY_CONFIG: dict = {
    # =====================================================================
    # Part A: 8-bit 格式对比
    # =====================================================================
    "part_a": {
        "type": "simple",
        "description": "8-bit Format Comparison",
        "table": "table1",
        "variants": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 32},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel", "axis": -1},
        ],
    },

    # =====================================================================
    # Part B: 4-bit 格式对比
    # =====================================================================
    "part_b": {
        "type": "simple",
        "description": "4-bit Format Comparison",
        "table": "table2",
        "variants": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": -1},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": -1, "weight_only": True},
        ],
    },

    # =====================================================================
    # Part C: FP32 vs PoT 缩放对比（LSQ 优化）
    # =====================================================================
    "part_c": {
        "type": "pot_scaling",
        "description": "FP32 vs PoT Scaling",
        "table": "table3",
        "lsq_steps": 100,
        "variants": [
            {"name": "INT8-PC", "format": "int8", "granularity": "per_channel", "axis": -1},
            {"name": "INT4-PC", "format": "int4", "granularity": "per_channel", "axis": -1},
        ],
    },

    # =====================================================================
    # Part D: 4-bit 变换研究（None / Hadamard / SmoothQuant / PerLayerOpt）
    # =====================================================================
    "part_d": {
        "type": "transform",
        "description": "Transform Study at 4-bit (MLP)",
        "table": "table4",
        "variants": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": -1},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": -1, "weight_only": True},
        ],
    },

    # =====================================================================
    # Block Size 扫描
    # =====================================================================
    "block_sweep": {
        "type": "block_sweep",
        "description": "Block size sensitivity sweep (int8)",
        "format": "int8",
        "block_sizes": [16, 32, 64, 128],
    },
}
