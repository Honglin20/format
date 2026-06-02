# Agent API: Block Error Analysis (Agent 6)

> Available to: `block_analyst` agent

## Core API

### `block_error_analysis()`

Extract per-block/per-channel QSNR ranking from a SessionResult.

```python
from src.api.block_error_analysis import block_error_analysis

report = block_error_analysis(
    result,               # SessionResult from Study
    layer="layer_name",   # Module name
    role="weight",        # "weight" or "input"
    top_k=10,             # Number of worst units to return
)

# report.per_unit_qsnr: Dict[int, float]   — unit_idx → qsnr_db
# report.worst_units: List[Tuple[int, float]]  — sorted worst-first
# report.stats: Dict[str, float]  — mean, std, min, max, p10, p90
# report.summary(): str
```

**Prerequisite**: Session must have been run with `PerBlockQSNRObserver`.

### Visualization Functions

```python
from src.viz.block_error_heatmap import (
    block_error_heatmap,
    channel_error_bar,
    multi_config_block_comparison,
)

# 2D heatmap (block × channel)
fig = block_error_heatmap(result, layer="...", role="weight", top_k_blocks=5)
fig.savefig("heatmap.png")

# Per-channel QSNR bar chart (sorted worst-first)
fig = channel_error_bar(result, layer="...", role="input", top_k=20)
fig.savefig("channel_bar.png")

# Cross-config grouped bar chart
fig = multi_config_block_comparison(study_report, layer="...", role="weight", top_k=10)
fig.savefig("comparison.png")
```

## Observer

### `PerBlockQSNRObserver`

Records per-block/per-channel QSNR individually (unlike QSNRObserver which aggregates).

```python
from src.analysis.observers import PerBlockQSNRObserver

observers = [QSNRObserver(), PerBlockQSNRObserver()]
# Output: {layer: {role: {stage: {("block", i): {"qsnr_db": float}}}}}
```
