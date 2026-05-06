"""
Format Study Configuration
===========================

修改实验只需改动此文件中的 ``STUDY_CONFIG`` 字典，无需修改 runner 代码。

----
每个 config dict 支持的字段
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
``scale_format``              否        缩放格式：``"fp32"``（默认） | ``"pot"``
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

**加一个 config**：在对应 part 的 ``configs`` 列表里加一个 dict。
**加一个新 part**：在 ``STUDY_CONFIG`` 里新增一个 key，结构仿照已有 part。
**跳过某个 part**：把该 part 的 key 注释掉或删除即可。

----
Part output 声明
----

每个 part 可以声明 ``output``，指定要生成的表格和图表类型：

.. code-block:: python

   "output": {
       "tables": ["accuracy", "pot_delta", "transform_matrix"],
       "figures": ["qsnr_line", "mse_box"],
   }

可用表格 key: ``accuracy``, ``pot_delta``, ``transform_matrix``,
``transform_distribution``, ``sensitivity``
可用图表 key: ``qsnr_line``, ``mse_box``, ``pot_delta_bar``, ``transform_heatmap``,
``transform_pie``, ``transform_delta``, ``histogram``, ``error_vs_dist``,
``layer_type_qsnr``, ``block_sweep``, ``hierarchical_delta``
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
        "description": "8-bit Format Comparison",
        "configs": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 32},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },

    # =====================================================================
    # Part B: 4-bit 格式对比
    # =====================================================================
    "part_b": {
        "description": "4-bit Format Comparison",
        "configs": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "axis": -1, "scale_format": "fp32"},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "axis": -1, "weight_only": True, "scale_format": "fp32"},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },

    # =====================================================================
    # Part C: FP32 vs PoT 缩放对比（LSQ 优化）
    # =====================================================================
    "part_c": {
        "description": "FP32 vs PoT Scaling",
        "configs": [
            {"name": "INT8-PC-FP32", "format": "int8", "granularity": "per_channel", "axis": -1, "scale_format": "fp32",
             "lsq_steps": 100, "lsq_pot": False},
            {"name": "INT8-PC-PoT",  "format": "int8", "granularity": "per_channel", "axis": -1, "scale_format": "pot",
             "lsq_steps": 100, "lsq_pot": True},
            {"name": "INT4-PC-FP32", "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "fp32",
             "lsq_steps": 100, "lsq_pot": False},
            {"name": "INT4-PC-PoT",  "format": "int4", "granularity": "per_channel", "axis": -1, "scale_format": "pot",
             "lsq_steps": 100, "lsq_pot": True},
        ],
        "output": {"tables": ["accuracy", "pot_delta"], "figures": ["pot_delta_bar"]},
    },

    # =====================================================================
    # Part D: 4-bit 变换研究（None / Hadamard / SmoothQuant / PerLayerOpt）
    # =====================================================================
    "part_d": {
        "description": "Transform Study at 4-bit (MLP)",
        "configs": [
            # MXINT-4 variants
            {"name": "MXINT-4-None",         "format": "int4",     "granularity": "per_block",   "block_size": 32, "transform": "none"},
            {"name": "MXINT-4-Hadamard",     "format": "int4",     "granularity": "per_block",   "block_size": 32, "transform": "hadamard"},
            {"name": "MXINT-4-SmoothQuant",  "format": "int4",     "granularity": "per_block",   "block_size": 32, "transform": "smoothquant"},
            # MXFP-4 variants
            {"name": "MXFP-4-None",          "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32, "transform": "none"},
            {"name": "MXFP-4-Hadamard",      "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32, "transform": "hadamard"},
            {"name": "MXFP-4-SmoothQuant",   "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32, "transform": "smoothquant"},
            # INT4-PC variants
            {"name": "INT4-PC-None",         "format": "int4",     "granularity": "per_channel", "axis": -1,       "transform": "none", "scale_format": "fp32"},
            {"name": "INT4-PC-Hadamard",     "format": "int4",     "granularity": "per_channel", "axis": -1,       "transform": "hadamard", "scale_format": "fp32"},
            {"name": "INT4-PC-SmoothQuant",  "format": "int4",     "granularity": "per_channel", "axis": -1,       "transform": "smoothquant", "scale_format": "fp32"},
            # NF4-PC variants (weight-only)
            {"name": "NF4-PC-None",          "format": "nf4",      "granularity": "per_channel", "axis": -1,       "weight_only": True, "transform": "none", "scale_format": "fp32"},
            {"name": "NF4-PC-Hadamard",      "format": "nf4",      "granularity": "per_channel", "axis": -1,       "weight_only": True, "transform": "hadamard", "scale_format": "fp32"},
            {"name": "NF4-PC-SmoothQuant",   "format": "nf4",      "granularity": "per_channel", "axis": -1,       "weight_only": True, "transform": "smoothquant", "scale_format": "fp32"},
        ],
        "output": {"tables": ["accuracy", "transform_matrix", "transform_distribution"], "figures": ["qsnr_line", "transform_heatmap", "transform_pie", "transform_delta"]},
    },

    # =====================================================================
    # Block Size 扫描
    # =====================================================================
    "block_sweep": {
        "description": "Block size sensitivity sweep (int8)",
        "configs": [
            {"name": "int8-blk16", "format": "int8", "granularity": "per_block", "block_size": 16},
            {"name": "int8-blk32", "format": "int8", "granularity": "per_block", "block_size": 32},
            {"name": "int8-blk64", "format": "int8", "granularity": "per_block", "block_size": 64},
            {"name": "int8-blk128", "format": "int8", "granularity": "per_block", "block_size": 128},
        ],
        "output": {"tables": ["accuracy"], "figures": ["block_sweep"]},
    },

    # =====================================================================
    # Hierarchical Pre-Scale Study (pot pre-scale + MX per-block)
    # =====================================================================
    "part_hierarchical": {
        "description": "Hierarchical Pre-Scale Study (pot pre-scale + MX per-block)",
        "configs": [
            {"name": "MXINT-8-HIER", "format": "int8",     "granularity": "per_block", "block_size": 32,
             "lsq_steps": 0, "pre_scale_init": "pot_amax", "pre_scale_pot": True},
            {"name": "MXFP-8-HIER",  "format": "fp8_e4m3", "granularity": "per_block", "block_size": 32,
             "lsq_steps": 0, "pre_scale_init": "pot_amax", "pre_scale_pot": True},
            {"name": "MXINT-4-HIER", "format": "int4",     "granularity": "per_block", "block_size": 32,
             "lsq_steps": 0, "pre_scale_init": "pot_amax", "pre_scale_pot": True},
            {"name": "MXFP-4-HIER",  "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32,
             "lsq_steps": 0, "pre_scale_init": "pot_amax", "pre_scale_pot": True},
        ],
        "output": {"tables": ["accuracy"], "figures": ["hierarchical_delta"]},
    },
}
