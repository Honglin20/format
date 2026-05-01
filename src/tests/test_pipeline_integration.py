"""Integration test: verify pipeline + viz produce valid output."""
import json
import os
import tempfile

import torch
import torch.nn as nn
from src.pipeline.config import resolve_config
from src.pipeline.runner import ExperimentRunner


class TestPipelineIntegration:
    def test_resolve_descriptors(self):
        """Key descriptor types resolve to OpQuantConfig."""
        descriptors = {
            "per_tensor": {"format": "int8", "granularity": "per_tensor"},
            "per_channel": {"format": "int8", "granularity": "per_channel", "axis": -1},
            "per_block": {"format": "int8", "granularity": "per_block", "block_size": 32},
            "fp8": {"format": "fp8_e4m3", "granularity": "per_block", "block_size": 32},
            "nf4_weight_only": {"format": "nf4", "granularity": "per_channel", "axis": -1, "weight_only": True},
        }
        for name, desc in descriptors.items():
            cfg = resolve_config(desc)
            assert cfg.weight is not None, f"{name}: weight scheme missing"

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
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean": sum(results) / len(results)}
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


def test_save_results_json_includes_block_sweep():
    """block_sweep key (not starting with 'part_') must be saved to results.json."""
    from src.pipeline.format_study import _save_results_json

    results = {"block_sweep": {"int8-blk32": {"accuracy": 0.85}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_results_json(results, tmpdir)
        with open(os.path.join(tmpdir, "results.json")) as f:
            saved = json.load(f)
        assert "block_sweep" in saved
        assert saved["block_sweep"]["int8-blk32"]["accuracy"] == 0.85


def test_save_results_json_includes_non_part_keys():
    """Any dict key should be saved, not just 'part_*' keys."""
    from src.pipeline.format_study import _save_results_json

    results = {
        "custom_experiment": {"variant_a": {"accuracy": 0.92}},
        "block_sweep": {"int8-blk32": {"accuracy": 0.85}},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_results_json(results, tmpdir)
        with open(os.path.join(tmpdir, "results.json")) as f:
            saved = json.load(f)
        assert "custom_experiment" in saved
        assert "block_sweep" in saved
