"""Format Study search space -- pure data, no framework dependencies.

Each study part defines a set of quantization configs as string-keyed
descriptors.  Descriptors are resolved to ``OpQuantConfig`` by
:func:`src.pipeline.config.resolve_config` at experiment time.

Adding a new format or granularity only requires adding a descriptor
entry -- no code changes needed in the runner or pipeline machinery.
"""

FORMAT_STUDY = {
    "part_a_8bit": {
        "description": "8-bit Format Comparison (PoT scaling)",
        "configs": {
            "MXINT-8": {"format": "int8", "granularity": "per_block", "block_size": 32},
            "MXFP-8": {"format": "fp8_e4m3", "granularity": "per_block", "block_size": 32},
            "INT8-PC": {"format": "int8", "granularity": "per_channel", "axis": 0},
        },
        "calibrator": "mse",
    },
    "part_b_4bit": {
        "description": "4-bit Format Comparison",
        "configs": {
            "MXINT-4": {"format": "int4", "granularity": "per_block", "block_size": 32},
            "MXFP-4": {"format": "fp4_e2m1", "granularity": "per_block", "block_size": 32},
            "INT4-PC": {"format": "int4", "granularity": "per_channel", "axis": 0},
            "NF4-PC": {
                "format": "nf4",
                "granularity": "per_channel",
                "axis": 0,
                "weight_only": True,
            },
        },
        "calibrator": "mse",
    },
    "part_c_pot_scaling": {
        "description": "FP32 vs PoT Scaling (INT8 + INT4 per-channel, LSQ optimized)",
        "configs": {
            "INT8-PC-FP32": {"format": "int8", "granularity": "per_channel", "axis": 0},
            "INT8-PC-PoT": {"format": "int8", "granularity": "per_channel", "axis": 0},
            "INT4-PC-FP32": {"format": "int4", "granularity": "per_channel", "axis": 0},
            "INT4-PC-PoT": {"format": "int4", "granularity": "per_channel", "axis": 0},
        },
        "calibrator": "mse",
        "lsq_steps": 100,
    },
    "part_d_transforms": {
        "description": "Transform Study at 4-bit (None / SmoothQuant / Hadamard)",
        "configs": {
            "MXINT-4": {"format": "int4", "granularity": "per_block", "block_size": 32},
            "MXFP-4": {"format": "fp4_e2m1", "granularity": "per_block", "block_size": 32},
            "INT4-PC": {"format": "int4", "granularity": "per_channel", "axis": 0},
            "NF4-PC": {
                "format": "nf4",
                "granularity": "per_channel",
                "axis": 0,
                "weight_only": True,
            },
        },
        "calibrator": "mse",
        "transforms": ["none", "smoothquant", "hadamard"],
    },
    "block_sweep": {
        "description": "Block size sensitivity sweep (int8, sizes 16/32/64/128)",
        "configs": {
            f"int8-blk{bs}": {"format": "int8", "granularity": "per_block", "block_size": bs}
            for bs in (16, 32, 64, 128)
        },
        "calibrator": "mse",
    },
}
