"""
Format Study Configuration
===========================

To modify experiments, edit the ``STUDY_CONFIG`` dict in this file — no need to touch the runner code.

Each config in the ``configs`` list is a :class:`QuantConfig` instance.

Available format strings
------------------------
- **Integer**: ``"int8"``, ``"int4"``, ``"int2"``
- **Float**: ``"fp8_e4m3"``, ``"fp8_e5m2"``, ``"fp6_e3m2"``, ``"fp6_e2m3"``, ``"fp4_e2m1"``
- **Lookup table**: ``"nf4"``
- **Standard**: ``"bfloat16"``, ``"float16"``

Adding new experiments
----------------------
**Add a config**: add a ``QuantConfig(...)`` to the ``configs`` list of the relevant part.
**Add a new part**: add a new key to ``STUDY_CONFIG``, following the structure of existing parts.
**Skip a part**: comment out or delete that part's key.

Part output declarations
------------------------
Each part can declare ``output`` to specify the tables and figures to generate:

.. code-block:: python

   "output": {
       "tables": ["accuracy", "pot_delta", "transform_matrix"],
       "figures": ["qsnr_line", "mse_box"],
   }

Available table keys: ``accuracy``, ``pot_delta``, ``transform_matrix``,
``transform_distribution``, ``sensitivity``
Available figure keys: ``qsnr_line``, ``mse_box``, ``pot_delta_bar``, ``transform_heatmap``,
``transform_pie``, ``transform_delta``, ``histogram``, ``error_vs_dist``,
``layer_type_qsnr``, ``block_sweep``, ``hierarchical_delta``

.. deprecated::
   ``STUDY_CONFIG`` is deprecated. Use :class:`QuantConfig` instances with
   :class:`Study` for multi-config experiments. See the README for examples.
"""

from __future__ import annotations

from src.session._config import QuantConfig

# ---------------------------------------------------------------------------
# Shorthand for common config patterns
# ---------------------------------------------------------------------------

_MX8 = dict(w_format="int8", w_granularity="per_block", w_block_size=32)
_MX4 = dict(w_format="int4", w_granularity="per_block", w_block_size=32)
_MXFP8 = dict(w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32)
_MXFP4 = dict(w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32)
_PC8 = dict(w_format="int8", w_granularity="per_channel", scale_storage="fp32")
_PC4 = dict(w_format="int4", w_granularity="per_channel", scale_storage="fp32")
_NF4 = dict(w_format="nf4", w_granularity="per_channel", weight_only=True, scale_storage="fp32")

# ---------------------------------------------------------------------------
# Main configuration — edit here!
# ---------------------------------------------------------------------------

STUDY_CONFIG: dict = {
    # =====================================================================
    # Part A: 8-bit format comparison
    # =====================================================================
    "part_a": {
        "description": "8-bit Format Comparison",
        "configs": [
            QuantConfig(name="MXINT-8", **_MX8),
            QuantConfig(name="MXFP-8", **_MXFP8),
            QuantConfig(name="INT8-PC", **_PC8),
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },

    # =====================================================================
    # Part B: 4-bit format comparison
    # =====================================================================
    "part_b": {
        "description": "4-bit Format Comparison",
        "configs": [
            QuantConfig(name="MXINT-4", **_MX4),
            QuantConfig(name="MXFP-4", **_MXFP4),
            QuantConfig(name="INT4-PC", **_PC4),
            QuantConfig(name="NF4-PC", **_NF4),
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },

    # =====================================================================
    # Part C: FP32 vs PoT scaling comparison (LSQ optimized)
    # =====================================================================
    "part_c": {
        "description": "FP32 vs PoT Scaling (LSQ optimized)",
        "configs": [
            QuantConfig(name="INT8-PC-FP32", w_format="int8", w_granularity="per_channel",
                        transform="prescale", scale_storage="fp32", lsq_steps=100),
            QuantConfig(name="INT8-PC-PoT", w_format="int8", w_granularity="per_channel",
                        transform="prescale", scale_storage="pot", lsq_steps=100),
            QuantConfig(name="INT4-PC-FP32", w_format="int4", w_granularity="per_channel",
                        transform="prescale", scale_storage="fp32", lsq_steps=100),
            QuantConfig(name="INT4-PC-PoT", w_format="int4", w_granularity="per_channel",
                        transform="prescale", scale_storage="pot", lsq_steps=100),
        ],
        "output": {"tables": ["accuracy", "pot_delta"], "figures": ["pot_delta_bar"]},
    },

    # =====================================================================
    # Part D: 4-bit transform study (None / Hadamard / SmoothQuant)
    # =====================================================================
    "part_d": {
        "description": "Transform Study at 4-bit (MLP)",
        "configs": [
            # MXINT-4 variants
            QuantConfig(name="MXINT-4-None", **_MX4, transform="none"),
            QuantConfig(name="MXINT-4-Hadamard", **_MX4, transform="hadamard"),
            QuantConfig(name="MXINT-4-SmoothQuant", **_MX4, transform="smoothquant"),
            # MXFP-4 variants
            QuantConfig(name="MXFP-4-None", **_MXFP4, transform="none"),
            QuantConfig(name="MXFP-4-Hadamard", **_MXFP4, transform="hadamard"),
            QuantConfig(name="MXFP-4-SmoothQuant", **_MXFP4, transform="smoothquant"),
            # INT4-PC variants
            QuantConfig(name="INT4-PC-None", **_PC4, transform="none"),
            QuantConfig(name="INT4-PC-Hadamard", **_PC4, transform="hadamard"),
            QuantConfig(name="INT4-PC-SmoothQuant", **_PC4, transform="smoothquant"),
            # NF4-PC variants (weight-only)
            QuantConfig(name="NF4-PC-None", **_NF4, transform="none"),
            QuantConfig(name="NF4-PC-Hadamard", **_NF4, transform="hadamard"),
            QuantConfig(name="NF4-PC-SmoothQuant", **_NF4, transform="smoothquant"),
        ],
        "output": {"tables": ["accuracy", "transform_matrix", "transform_distribution"],
                   "figures": ["qsnr_line", "transform_heatmap", "transform_pie", "transform_delta"]},
    },

    # =====================================================================
    # Block size sweep
    # =====================================================================
    "block_sweep": {
        "description": "Block size sensitivity sweep (int8)",
        "configs": [
            QuantConfig(name="int8-blk16", w_format="int8", w_granularity="per_block", w_block_size=16),
            QuantConfig(name="int8-blk32", w_format="int8", w_granularity="per_block", w_block_size=32),
            QuantConfig(name="int8-blk64", w_format="int8", w_granularity="per_block", w_block_size=64),
            QuantConfig(name="int8-blk128", w_format="int8", w_granularity="per_block", w_block_size=128),
        ],
        "output": {"tables": ["accuracy"], "figures": ["block_sweep"]},
    },

    # =====================================================================
    # Hierarchical Pre-Scale Study (pot pre-scale + MX per-block)
    # =====================================================================
    "part_hierarchical": {
        "description": "Hierarchical Pre-Scale Study (pot pre-scale + MX per-block)",
        "configs": [
            QuantConfig(name="MXINT-8-HIER", **_MX8,
                        transform="prescale", lsq_steps=0,
                        prescale_init="pot_amax", prescale_pot=True),
            QuantConfig(name="MXFP-8-HIER", **_MXFP8,
                        transform="prescale", lsq_steps=0,
                        prescale_init="pot_amax", prescale_pot=True),
            QuantConfig(name="MXINT-4-HIER", **_MX4,
                        transform="prescale", lsq_steps=0,
                        prescale_init="pot_amax", prescale_pot=True),
            QuantConfig(name="MXFP-4-HIER", **_MXFP4,
                        transform="prescale", lsq_steps=0,
                        prescale_init="pot_amax", prescale_pot=True),
        ],
        "output": {"tables": ["accuracy"], "figures": ["hierarchical_delta"]},
    },
}
