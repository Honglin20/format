"""Integration test: verify session + viz produce valid output."""
import copy
import os
import tempfile

import torch
import torch.nn as nn
from src.session import Session, QuantConfig
from src.session._config import resolve_config


class TestPipelineIntegration:
    def test_resolve_descriptors(self):
        """Key descriptor types resolve to OpQuantConfig via resolve_config."""
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

    def test_session_minimal_end_to_end(self):
        """Session completes quantize->calibrate->analyze->evaluate for a tiny model."""
        model = nn.Sequential(nn.Linear(4, 3))

        cfg = QuantConfig(
            name="int8",
            w_format="int8",
            w_granularity="per_tensor",
        )
        session = Session(model, cfg)

        def _eval_fn(m, data):
            m.eval()
            with torch.no_grad():
                if isinstance(data, (list, tuple)):
                    results = [m(b).mean().item() for b in data]
                    return {"mean": sum(results) / len(results)}
                return {"mean": m(data).mean().item()}

        calib = [torch.randn(2, 4)]
        result = session.run(
            calib_data=calib,
            eval_data=torch.randn(2, 4),
            eval_fn=_eval_fn,
        )

        assert result.fp32_metrics is not None
        assert result.quant_metrics is not None
        assert result.delta is not None
        assert "mean" in result.delta

    def test_viz_imports_no_session(self):
        """src/viz/ must not import from src/session/."""
        import ast

        viz_dir = os.path.join(os.path.dirname(__file__), "..", "viz")
        forbidden = {"src.session"}

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


def test_session_handles_block_sweep_style_config():
    """Session can handle multiple configs via Study."""
    from src.session import Study, QuantConfig

    configs = [
        QuantConfig(name="int8-blk16", w_format="int8", w_granularity="per_block", w_block_size=16),
        QuantConfig(name="int8-blk32", w_format="int8", w_granularity="per_block", w_block_size=32),
    ]
    model = nn.Sequential(nn.Linear(4, 3))
    study = Study(configs, model=model)

    def _eval_fn(m, data):
        m.eval()
        with torch.no_grad():
            if isinstance(data, (list, tuple)):
                for batch in data:
                    m(batch)
                return {"accuracy": 0.0}
            return {"accuracy": 0.0}

    report = study.run(
        calib_data=[torch.randn(2, 4)],
        eval_data=torch.randn(2, 4),
        eval_fn=_eval_fn,
    )
    results = report.to_serializable()
    assert "int8-blk16" in results
    assert "int8-blk32" in results
