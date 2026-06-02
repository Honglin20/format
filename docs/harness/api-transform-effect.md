# Agent API: Transform Effect Analysis (Agent 3, 7)

> Available to: `gap_analyzer`, `intervention_evaluator` agents

## `TransformEffectReport`

Quantify how much each transform (SmoothQuant, Hadamard) recovers precision.

```python
from src.analysis.transform_effect import TransformEffectReport

report = TransformEffectReport.from_study(study_report)

# Formatted table: config × transform → accuracy + recovery_pct
print(report.summary())

# Per-config accuracy recovery
recovery = report.per_config_recovery()
# → [{"base_config": "W4A4", "transform": "smoothquant",
#     "accuracy_gain": 0.021, "recovery_pct": 38.9, "qsnr_gain_db": 4.5}]

# Raw pairs for detailed analysis
for p in report.pairs:
    print(f"{p['base_config']} +{p['transform']}: "
          f"gain={p['accuracy_gain']:+.4f}, recovery={p['recovery_pct']:.1f}%, "
          f"qsnr Δ={p['qsnr_gain_db']:+.1f} dB")
```

## Config Naming Convention

Transform effects are auto-detected by matching config names:

| Base config | +SmoothQuant | +Hadamard |
|-------------|-------------|-----------|
| `W8A8`      | `W8A8+SQ`   | `W8A8+HD` |
| `W4A8`      | `W4A8+SQ`   | —         |
| `W4A4`      | `W4A4+SQ`   | `W4A4+HD` |

Detection uses regex: `+SQ` → SmoothQuant, `+HD` → Hadamard.
