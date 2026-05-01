"""Integration test: verify pipeline + viz produce valid output."""
import os
import tempfile
from unittest.mock import patch

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


def test_incremental_save_after_each_part():
    """After running a multi-part study, results.json should be updated per-part."""
    from src.pipeline.format_study import run_format_study, _save_results_json
    from pipeline._model import ToyMLP
    from torch.utils.data import DataLoader, TensorDataset

    mini_config = {
        "part_a": {
            "type": "simple",
            "description": "mini 8-bit",
            "table": "table1",
            "variants": [{"name": "INT8-PT", "format": "int8", "granularity": "per_tensor"}],
        },
        "part_b": {
            "type": "simple",
            "description": "mini 4-bit",
            "table": "table2",
            "variants": [{"name": "INT4-PT", "format": "int4", "granularity": "per_tensor"}],
        },
    }

    def build_model():
        m = ToyMLP(hidden_size=16, intermediate_size=32)
        m.head = nn.Linear(16, 10)
        return m

    def make_calib():
        return [torch.randn(4, 16)]

    def make_eval():
        x = torch.randn(16, 16)
        y = torch.randint(0, 10, (16,))
        return DataLoader(TensorDataset(x, y), batch_size=4)

    def eval_fn(m, loader):
        m.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            # loader can be a list of tensors (calib_data) or a DataLoader
            for item in loader:
                if isinstance(item, (list, tuple)):
                    x = item[0]
                else:
                    x = item
                out = m(x)
                total += x.size(0)
                correct += out.mean().item()
        return {"accuracy": correct / total if total > 0 else 0.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.pipeline.format_study._save_results_json', wraps=_save_results_json) as mock_save:
            run_format_study(
                build_model, make_calib, make_eval, eval_fn,
                output_dir=tmpdir, config=mini_config,
            )
            assert mock_save.call_count >= 2, \
                f"Expected >=2 incremental saves, got {mock_save.call_count}"


def test_plot_from_results_handles_block_sweep_and_hierarchical():
    """plot_from_results should generate tables for block_sweep and part_hierarchical."""
    from src.pipeline.format_study import plot_from_results
    import json

    results = {
        "block_sweep": {
            "int8-blk32": {"accuracy": {"accuracy": 0.85}, "qsnr_per_layer": {}, "mse_per_layer": {}},
        },
        "part_hierarchical": {
            "MXINT-8-HIER": {"accuracy": {"accuracy": 0.88}, "qsnr_per_layer": {}, "mse_per_layer": {}},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f)
        plot_from_results(results_path, output_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "tables", "block_sweep.csv"))
        assert os.path.exists(os.path.join(tmpdir, "tables", "hierarchical.csv"))
        assert os.path.exists(os.path.join(tmpdir, "figures"))
