# Sparse Format Research

Systematic evaluation of sparse quantization (outlier-based, two-group) vs MXINT (int4 + PER_BLOCK) at 4-bit precision.

## Quick Start

```bash
# Run all experiments
PYTHONPATH=. python research/sparse/experiments/l1_baseline.py
PYTHONPATH=. python research/sparse/experiments/l2_ratio_sweep.py
PYTHONPATH=. python research/sparse/experiments/l3_bank_sweetspot.py

# Generate figures (reads results/*.json)
PYTHONPATH=. python research/sparse/viz/l1_viz.py
PYTHONPATH=. python research/sparse/viz/l2_viz.py
PYTHONPATH=. python research/sparse/viz/l3_viz.py
```

## Research Questions

1. **L1 — QSNR Comparison**: Does sparse QSNR exceed MXINT across granularity modes?
2. **L2 — Ratio Sweep**: How does outlier_ratio affect QSNR and effective bitwidth?
3. **L3 — Bank Sweet Spot**: What is the optimal bank_size for a given tensor dimension?
4. **L4 — Generalization**: What is the mask generalization gap from calibration to test? (skeleton)

## Directory Layout

```
research/sparse/
├── configs/experiments.py    — Experiment parameters
├── experiments/              — Experiment scripts (→ results/*.json)
│   ├── l1_baseline.py
│   ├── l2_ratio_sweep.py
│   ├── l3_bank_sweetspot.py
│   └── l4_real_model.py      — Skeleton only
├── viz/                      — Visualization scripts (→ figures/*.png)
│   ├── common.py
│   ├── l1_viz.py
│   ├── l2_viz.py
│   └── l3_viz.py
├── results/                  — JSON results (versioned)
└── figures/                  — Output PNG/SVG
```

## Design Principles

- **Experiment ↔ Visualization decoupled**: Experiments write JSON, viz scripts read JSON. Changing colors/layout never requires re-running experiments.
- **QuantScheme-driven**: All experiments use `quantize(x, scheme)` — no Session or observer pipeline.
- **No mx/ imports**: All quantization goes through `src/` APIs.
