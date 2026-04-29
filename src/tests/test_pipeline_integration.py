"""Integration test: verify pipeline + viz produce valid output."""
import torch
import torch.nn as nn
from src.pipeline.config import resolve_config
from src.pipeline.runner import ExperimentRunner
from src.pipeline.studies.format_study import FORMAT_STUDY


class TestPipelineIntegration:
    def test_format_study_search_space_resolves_all_configs(self):
        """Every descriptor in FORMAT_STUDY resolves to OpQuantConfig."""
        for part_name, part_def in FORMAT_STUDY.items():
            for cfg_name, cfg_desc in part_def["configs"].items():
                cfg = resolve_config(cfg_desc)
                assert cfg.weight is not None, f"{part_name}/{cfg_name}: weight scheme missing"

    def test_runner_minimal_end_to_end(self):
        """Runner completes quantize->calibrate->analyze->evaluate for a tiny model."""
        model = nn.Sequential(nn.Linear(4, 3))

        study = {
            "test": {
                "description": "minimal integration test",
                "configs": {
                    "int8": {"format": "int8", "granularity": "per_tensor"},
                },
            },
        }
        runner = ExperimentRunner(study)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                return {"mean": m(data).mean().item()}

        calib = [torch.randn(2, 4)]
        results = runner.run(
            fp32_model=model,
            eval_fn=_eval_fn,
            calib_data=calib,
            eval_data=torch.randn(2, 4),
        )

        r = results["test/int8"]
        assert "fp32" in r
        assert "quant" in r
        assert "delta" in r
        assert "mean" in r["delta"]

    def test_viz_imports_no_pipeline(self):
        """src/viz/ must not import from src/pipeline/ or src/session.py."""
        import ast
        import os

        viz_dir = os.path.join(os.path.dirname(__file__), "..", "viz")
        forbidden = {"src.pipeline", "src.session"}

        for fname in os.listdir(viz_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(viz_dir, fname)
            with open(fpath) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", None)
                    if module:
                        for forbidden_mod in forbidden:
                            assert not module.startswith(forbidden_mod), \
                                f"{fname} imports {module} (forbidden: {forbidden_mod})"
