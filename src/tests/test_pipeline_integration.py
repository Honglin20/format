"""Integration test: verify pipeline + viz produce valid output."""
import json
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
                "configs": [
                    {"name": "int8", "format": "int8", "granularity": "per_tensor"},
                ],
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

        r = results["test"][0]
        assert r.fp32_metrics is not None
        assert r.quant_metrics is not None
        assert r.delta is not None
        assert "mean" in r.delta

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


def test_incremental_save_after_each_part():
    """Study saves results.json and produces output."""
    from src.pipeline.format_study import run_format_study
    from pipeline._model import ToyMLP
    from torch.utils.data import DataLoader, TensorDataset

    mini_config = {
        "part_a": {
            "description": "mini 8-bit",
            "configs": [{"name": "INT8-PT", "format": "int8", "granularity": "per_tensor"}],
            "output": {"tables": ["accuracy"], "figures": ["qsnr_line"]},
        },
    }

    def build_model():
        model = ToyMLP(hidden_size=16, intermediate_size=32)
        model.head = nn.Linear(16, 10)
        return model

    def make_calib():
        return [torch.randn(4, 16)]

    def make_eval():
        x = torch.randn(16, 16)
        y = torch.randint(0, 10, (16,))
        return DataLoader(TensorDataset(x, y), batch_size=4)

    def eval_fn(m, data):
        """Handles both calibration (list of tensors) and eval (DataLoader)."""
        with torch.no_grad():
            m.eval()
            if isinstance(data, (list, tuple)) and isinstance(data[0], torch.Tensor):
                # Calibration: list of batch tensors — just forward pass
                for batch in data:
                    if isinstance(batch, torch.Tensor) and batch.dim() >= 2:
                        m(batch)
                return {"accuracy": 0.0}
            # Evaluation: DataLoader yielding (input, label)
            correct, total = 0, 0
            for batch in data:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    xb, yb = batch
                else:
                    xb = batch
                    yb = torch.zeros(1)
                out = m(xb)
                if out.size(1) >= 2:
                    correct += (out.argmax(1) == yb).sum().item()
                    total += yb.size(0)
            return {"accuracy": correct / total if total > 0 else 0.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_format_study(
            build_model, make_calib, make_eval, eval_fn,
            output_dir=tmpdir, config=mini_config,
        )
        assert "part_a" in results
        assert len(results["part_a"]) >= 1
        # Verify results.json was saved
        assert os.path.exists(os.path.join(tmpdir, "results.json"))
        # Verify tables were generated
        assert os.path.exists(os.path.join(tmpdir, "tables"))


def test_plot_from_results_handles_block_sweep_and_hierarchical():
    """plot_from_results should generate tables for block_sweep and part_hierarchical."""
    from src.pipeline.format_study import plot_from_results

    results = {
        "block_sweep": {"int8-blk32": {"accuracy": {"accuracy": 0.85},
                         "qsnr_per_layer": {}, "mse_per_layer": {}}},
        "part_hierarchical": {"MXINT-8-HIER": {"accuracy": {"accuracy": 0.88},
                                "qsnr_per_layer": {}, "mse_per_layer": {}}},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f)
        plot_from_results(results_path, output_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "tables", "block_sweep.csv"))
        assert os.path.exists(os.path.join(tmpdir, "tables", "hierarchical.csv"))
        assert os.path.exists(os.path.join(tmpdir, "figures"))


def test_plot_from_results_regenerates_tables_and_figures():
    """plot_from_results should regenerate all tables and figures from a saved JSON."""
    from src.pipeline.format_study import plot_from_results

    results = {
        "part_a": {"INT8-PT": {"accuracy": {"accuracy": 0.92},
                    "qsnr_per_layer": {}, "mse_per_layer": {}}},
        "part_b": {"INT4-PT": {"accuracy": {"accuracy": 0.88},
                    "qsnr_per_layer": {}, "mse_per_layer": {}}},
        "part_c": {"INT8-PC-FP32": {"accuracy": {"accuracy": 0.90},
                    "qsnr_per_layer": {}, "mse_per_layer": {}},
                   "FP32 (baseline)": {"accuracy": 0.95}},
        "block_sweep": {"int8-blk32": {"accuracy": {"accuracy": 0.85},
                        "qsnr_per_layer": {}, "mse_per_layer": {}}},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f)
        plot_from_results(results_path, output_dir=tmpdir)
        # All table types regenerated
        assert os.path.exists(os.path.join(tmpdir, "tables", "table1_8bit.csv"))
        assert os.path.exists(os.path.join(tmpdir, "tables", "table2_4bit.csv"))
        assert os.path.exists(os.path.join(tmpdir, "tables", "block_sweep.csv"))
        assert os.path.exists(os.path.join(tmpdir, "figures"))
        # Figures dir populated (part_a and part_c triggers figure generation)
        figures = os.listdir(os.path.join(tmpdir, "figures"))
        assert len(figures) >= 1  # at least one figure produced
