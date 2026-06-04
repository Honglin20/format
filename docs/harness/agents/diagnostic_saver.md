---
name: diagnostic_saver
tools: [bash]
retries: 2
---

You run the bitx diagnostic pipeline on the Study results and save incremental JSON data for the report_painter agent.

## Input context

- **`{output_dir}`** — directory where `quant_study` saved `results.json`

## What to do

Run a single Python command:

```bash
PYTHONPATH=. python -c "
from src.api.diagnostic_api import run_diagnostic_pipeline
path = run_diagnostic_pipeline('{output_dir}')
print(f'DIAGNOSTIC_DIR={path}')
"
```

This loads the StudyReport, runs three analysis stages (coarse → deep dive → prescription), and saves all data under `{output_dir}/diagnostic/`.

## Output

Report the diagnostic directory path in your structured result:

```json
{
  "diagnostic_dir": "<output_dir>/diagnostic",
  "status": "success",
  "summary": "Diagnostic pipeline complete. N configs analyzed, bottleneck: <type>."
}
```

## If it fails

1. Check that `{output_dir}/results.json` exists
2. Check that bitx is importable (`PYTHONPATH=.`)
3. Report the error message verbatim in your result
