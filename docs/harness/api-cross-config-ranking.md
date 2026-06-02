# Agent API: Cross-Config Layer Ranking (Agent 4)

> Available to: `layer_attribution` agent

## `CrossConfigLayerRanking`

Compare which layers are consistently worst across multiple quantization configs.

```python
from src.analysis.cross_config_ranking import CrossConfigLayerRanking

ranking = CrossConfigLayerRanking.from_study(study_report)

# Layers worst in ALL configs
worst = ranking.consistent_worst(k=5)
# → List[Tuple[str, float]]  — [(layer_name, avg_qsnr_db)]

# Layers worst only in a specific config
specific = ranking.config_specific_worst(config="W4A4", k=3)
# → List[Tuple[str, float]]

# QSNR improvement for a layer between two configs
delta = ranking.layer_qsnr_delta("fc2", from_config="W4A4", to_config="W8A8")
# → float  — positive means improvement

# Formatted summary
print(ranking.summary())
```

## Reading StudyReport

```python
from src.report._study_report import StudyReport

report = StudyReport.from_file(output_dir)

# Access individual results
for config_name, results in report._results.items():
    for r in results:
        acc = r.accuracy.get("accuracy") if r.accuracy else None
        qsnr_per_layer = r.qsnr_per_layer  # {layer: qsnr_db}
```
