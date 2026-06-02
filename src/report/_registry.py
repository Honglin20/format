"""Lazy-loaded registries of table and figure generator functions.

Viz functions are loaded on first access to avoid circular imports
and keep startup fast. All wrapped functions share the signature::

    fn(data: dict, output_dir: str, **kwargs) -> None
"""

from __future__ import annotations

from typing import Callable, Dict

# ---------------------------------------------------------------------------
# Registries (populated on first access)
# ---------------------------------------------------------------------------

_TABLE_REGISTRY: Dict[str, Callable] = {}
_FIGURE_REGISTRY: Dict[str, Callable] = {}


def _ensure_registries():
    """Populate registries on first access (lazy init)."""
    if _TABLE_REGISTRY:
        return

    # ── Tables ──────────────────────────────────────────────────────────
    from src.viz.tables import (
        accuracy_table,
        distribution_fit_table,
        per_layer_qsnr_table,
        pot_delta_table,
        sensitivity_table,
        transform_benefit_table,
        transform_distribution_table,
        transform_matrix_table,
    )
    _TABLE_REGISTRY["accuracy"] = (
        lambda d, od, **kw: accuracy_table(
            d, title=kw.get("title", ""), output_dir=od,
            filename=kw.get("filename", "accuracy.csv"),
        )
    )
    _TABLE_REGISTRY["distribution_fit"] = (
        lambda d, od, **kw: distribution_fit_table(d, od)
    )
    _TABLE_REGISTRY["pot_delta"] = (
        lambda d, od, **kw: pot_delta_table(d, od)
    )
    _TABLE_REGISTRY["sensitivity"] = (
        lambda d, od, **kw: sensitivity_table(d, od)
    )
    _TABLE_REGISTRY["transform_benefit"] = (
        lambda d, od, **kw: transform_benefit_table(d, od)
    )
    _TABLE_REGISTRY["transform_matrix"] = (
        lambda d, od, **kw: transform_matrix_table(d, od)
    )
    _TABLE_REGISTRY["transform_dist"] = (
        lambda d, od, **kw: transform_distribution_table(d, od)
    )
    _TABLE_REGISTRY["per_layer_qsnr"] = (
        lambda d, od, **kw: per_layer_qsnr_table(
            d, output_dir=od, filename=kw.get("filename", "per_layer_qsnr.csv"),
        )
    )

    # ── Figures ─────────────────────────────────────────────────────────
    from src.viz.figures import (
        block_sweep_line_chart,
        correlation_heatmap,
        error_vs_distribution,
        hierarchical_delta_bar,
        histogram_overlay,
        layer_type_qsnr,
        mse_box_plot,
        outlier_analysis,
        per_block_qsnr,
        per_layer_role_histogram,
        pot_delta_bar,
        qsnr_line_chart,
        role_distribution_comparison,
        transform_delta,
        transform_heatmap,
        transform_pie,
    )
    from src.viz.theme import FORMAT_COLORS, TRANSFORM_COLORS

    _FIGURE_REGISTRY["qsnr"] = (
        lambda d, od, **kw: qsnr_line_chart(
            d, title=kw.get("title", "QSNR per Layer (output)"),
            colors=kw.get("colors", FORMAT_COLORS), output_dir=od,
        )
    )
    _FIGURE_REGISTRY["mse"] = (
        lambda d, od, **kw: mse_box_plot(
            d, title=kw.get("title", "MSE per Layer (output)"),
            colors=kw.get("colors", FORMAT_COLORS), output_dir=od,
        )
    )
    _FIGURE_REGISTRY["pot_delta_bar"] = (
        lambda d, od, **kw: pot_delta_bar(d, output_dir=od)
    )
    _FIGURE_REGISTRY["transform_heatmap"] = (
        lambda d, od, **kw: transform_heatmap(
            d, colors=kw.get("colors", FORMAT_COLORS), output_dir=od,
        )
    )
    _FIGURE_REGISTRY["transform_pie"] = (
        lambda d, od, **kw: transform_pie(
            d, colors=kw.get("colors", TRANSFORM_COLORS), output_dir=od,
        )
    )
    _FIGURE_REGISTRY["transform_delta"] = (
        lambda d, od, **kw: transform_delta(
            d, colors=kw.get("colors", TRANSFORM_COLORS), output_dir=od,
        )
    )
    _FIGURE_REGISTRY["histogram"] = (
        lambda d, od, **kw: histogram_overlay(d, output_dir=od)
    )
    _FIGURE_REGISTRY["error_dist"] = (
        lambda d, od, **kw: error_vs_distribution(d, output_dir=od)
    )
    _FIGURE_REGISTRY["layer_qsnr"] = (
        lambda d, od, **kw: layer_type_qsnr(d, output_dir=od)
    )
    _FIGURE_REGISTRY["block_sweep"] = (
        lambda d, od, **kw: block_sweep_line_chart(d, output_dir=od)
    )
    _FIGURE_REGISTRY["hierarchical"] = (
        lambda d, od, **kw: hierarchical_delta_bar(
            d, colors=kw.get("colors", FORMAT_COLORS), output_dir=od,
        )
    )
    _FIGURE_REGISTRY["outlier"] = (
        lambda d, od, **kw: outlier_analysis(
            d, output_dir=od, roles=kw.get("roles", ("input", "weight", "output")),
        )
    )
    _FIGURE_REGISTRY["per_block_qsnr"] = (
        lambda d, od, **kw: per_block_qsnr(
            d, output_dir=od, roles=kw.get("roles", ("input", "weight", "output")),
        )
    )
    _FIGURE_REGISTRY["correlation_heatmap"] = (
        lambda d, od, **kw: correlation_heatmap(d, output_dir=od)
    )
    _FIGURE_REGISTRY["role_distribution"] = (
        lambda d, od, **kw: role_distribution_comparison(d, output_dir=od)
    )
    _FIGURE_REGISTRY["per_layer_role_histogram"] = (
        lambda d, od, **kw: per_layer_role_histogram(
            d, output_dir=od, k=kw.get("k", 5),
        )
    )


def get_table_fn(key: str) -> Callable:
    """Look up a table-generator function by output key.

    Raises:
        KeyError: If the key is not registered.
    """
    _ensure_registries()
    try:
        return _TABLE_REGISTRY[key]
    except KeyError:
        valid = sorted(_TABLE_REGISTRY.keys())
        raise KeyError(f"Unknown table key {key!r}. Valid keys: {valid}")


def get_figure_fn(key: str) -> Callable:
    """Look up a figure-generator function by output key.

    Raises:
        KeyError: If the key is not registered.
    """
    _ensure_registries()
    try:
        return _FIGURE_REGISTRY[key]
    except KeyError:
        valid = sorted(_FIGURE_REGISTRY.keys())
        raise KeyError(f"Unknown figure key {key!r}. Valid keys: {valid}")
