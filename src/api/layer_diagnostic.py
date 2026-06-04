"""Layer-level diagnostic atomic functions for harness integration.

Primitive plotting functions:
  accum_qsnr_bar           — Accumulated QSNR bar chart (linear-only)
  accum_vs_local_line      — Accum vs local QSNR comparison (the one local reference)
  per_role_local_qsnr      — Per-role local QSNR grouped bar
  error_attribution_waterfall — Activation vs weight error contribution
  extreme_layer_table      — Top-K worst + best layers summary

Deep-dive functions:
  layer_deep_dive          — Full 3-role diagnostic for one layer (dist_overlay)
  compare_extreme_layers   — Top-K worst + best layers with distributions
  block_heatmap            — Per-block QSNR distribution
  distribution_table       — All-layer distribution fingerprint
  diagnosis_report         — All-layer causal analysis + scatter

Requires: DistributionObserver, HistogramObserver, PerBlockQSNRObserver
          attached during Session.run().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.session._result import SessionResult

from src.analysis._distribution_diagnosis import classify_distribution
from src.api._chart_helpers import (
    _chart, _get_dist_metrics, _get_hist_data, _get_per_block_qsnr,
    _block_stats, _linear_layer_names, _filter_qsnr,
    _EXCLUDE_KEYWORDS, _DIST_KEYS, QSNR_REF,
)


def _sorted_layers(result: "SessionResult") -> List[Tuple[str, float]]:
    """Return layers sorted by output QSNR (ascending = worst first)."""
    items = [(k, v) for k, v in result.qsnr_per_layer.items()
             if math.isfinite(v)]
    items.sort(key=lambda x: x[1])
    return items


# =====================================================================
# Primitive plotting functions (P1–P5)
# =====================================================================

def accum_qsnr_bar(result: "SessionResult", *, linear_only: bool = True, label: str = ""):
    """P1: Accumulated QSNR bar chart (primary QSNR visualization)."""
    data_dict = _filter_qsnr(result.accum_qsnr_per_layer, linear_only, result.observers_data)
    if not data_dict:
        return
    data = [{"layer": k, "qsnr_db": v} for k, v in data_dict.items()]
    _chart(data, "bar", x="layer", y="qsnr_db",
           label=label, title="Per-Layer Accumulated QSNR (dB)")


def accum_vs_local_line(result: "SessionResult", *, linear_only: bool = True, label: str = ""):
    """P2: The ONE chart comparing accumulated vs local QSNR."""
    accum = _filter_qsnr(result.accum_qsnr_per_layer, linear_only, result.observers_data)
    local = _filter_qsnr(result.qsnr_per_layer, linear_only, result.observers_data)
    if not accum and not local:
        return
    layers = sorted(set(accum.keys()) | set(local.keys()))
    data = []
    for i, layer in enumerate(layers):
        if layer in local:
            data.append({"layer_idx": i, "layer": layer, "qsnr_db": local[layer], "type": "local"})
        if layer in accum:
            data.append({"layer_idx": i, "layer": layer, "qsnr_db": accum[layer], "type": "accumulated"})
    if data:
        _chart(data, "line", x="layer_idx", y="qsnr_db", hue="type",
               label=label, title="Error Propagation: Local vs Accumulated QSNR")


def per_role_local_qsnr(result: "SessionResult", *, linear_only: bool = True, label: str = ""):
    """P3: Per-role local QSNR grouped bar (input / weight / output)."""
    qsnr_by_role = result.qsnr_by_role or {}
    if not qsnr_by_role:
        return
    allowed = _linear_layer_names(result.observers_data) if (linear_only and result.observers_data) else None
    data = []
    for role, layer_map in qsnr_by_role.items():
        for layer, qsnr in layer_map.items():
            if allowed is not None and layer not in allowed:
                continue
            data.append({"layer": layer, "role": role, "qsnr_db": qsnr})
    if data:
        _chart(data, "bar", x="layer", y="qsnr_db", hue="role",
               label=label, title="Per-Role Local QSNR (dB)")


def error_attribution_waterfall(
    result: "SessionResult",
    *,
    linear_only: bool = True,
    k: int = 10,
    label: str = "",
):
    """P4: Error attribution waterfall — activation vs weight contribution."""
    qsnr_by_role = result.qsnr_by_role or {}
    accum = _filter_qsnr(result.accum_qsnr_per_layer, linear_only, result.observers_data)
    if not qsnr_by_role or not accum:
        return

    all_layers = set(accum.keys())
    scored = []
    for layer in all_layers:
        output_qsnr = accum.get(layer)
        if output_qsnr is None:
            continue
        role_qsnrs = {}
        for role in ("input", "weight", "output"):
            v = qsnr_by_role.get(role, {}).get(layer)
            if v is not None:
                role_qsnrs[role] = v
        if role_qsnrs:
            dominant = min(role_qsnrs, key=role_qsnrs.get)
        else:
            dominant = "output"
        scored.append((layer, output_qsnr, dominant, role_qsnrs))

    scored.sort(key=lambda x: x[1])
    worst = scored[:k]

    bar_data = []
    for layer, output_qsnr, dominant, role_qsnrs in worst:
        input_q = role_qsnrs.get("input")
        weight_q = role_qsnrs.get("weight")
        act_loss = QSNR_REF - input_q if input_q is not None else 0.0
        w_loss = QSNR_REF - weight_q if weight_q is not None else 0.0
        bar_data.append({"layer": layer, "error_contribution": round(act_loss, 2),
                         "source": "activation", "dominant": dominant})
        bar_data.append({"layer": layer, "error_contribution": round(w_loss, 2),
                         "source": "weight", "dominant": dominant})

    if bar_data:
        _chart(bar_data, "bar", x="layer", y="error_contribution", hue="source",
               label=label, title="Error Attribution: Activation vs Weight")

    table_data = []
    for layer, output_qsnr, dominant, role_qsnrs in worst:
        table_data.append({
            "layer": layer,
            "output_qsnr": round(output_qsnr, 1),
            "activation_qsnr": round(role_qsnrs["input"], 1) if "input" in role_qsnrs else "N/A",
            "weight_qsnr": round(role_qsnrs["weight"], 1) if "weight" in role_qsnrs else "N/A",
            "dominant_error": dominant,
        })
    if table_data:
        _chart(table_data, "table", x="layer", y="output_qsnr",
               label=label, title="Error Attribution: Worst Layers")


def extreme_layer_table(result: "SessionResult", *, k: int = 3, linear_only: bool = True):
    """P5: Top-K worst + top-K best layers summary table."""
    accum = _filter_qsnr(result.accum_qsnr_per_layer, linear_only, result.observers_data)
    if not accum:
        print("[bitx] No accumulated QSNR data.")
        return

    layers = sorted(accum.items(), key=lambda x: x[1])
    worst = layers[:k]
    best = layers[-k:] if len(layers) > k else layers[:]
    best = list(reversed(best))

    rows = []
    for i, (name, qsnr) in enumerate(worst):
        rows.append({"rank": f"worst-{i+1}", "layer": name, "qsnr_db": round(qsnr, 1)})
    for i, (name, qsnr) in enumerate(best):
        lbl = f"best-{i+1}" if name not in {n for n, _ in worst} else f"best-{i+1}*"
        rows.append({"rank": lbl, "layer": name, "qsnr_db": round(qsnr, 1)})

    _chart(rows, "table", x="rank", y="qsnr_db",
           label="", title=f"Extreme Layers: Top-{k} Worst + Top-{k} Best (Accum QSNR)")
    print(f"\n[bitx] Extreme layers (worst → best):")
    for r in rows:
        print(f"  {r['rank']:<10} {r['layer']:<30} QSNR: {r['qsnr_db']:.1f} dB")
    return rows


def _to_list(x):
    """Convert tensor or array to list."""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return list(x)


# =====================================================================
# 1. layer_deep_dive
# =====================================================================

def layer_deep_dive(result: "SessionResult", layer: str, label: str = ""):
    """Full 3-role diagnostic for a single layer.

    For each role (input, weight, output), outputs:
    - Distribution fingerprint table
    - Distribution overlay (area: fp32 vs quant vs error)
    - Per-block QSNR statistics table
    - Top-5 worst blocks
    - Failure mode classification
    """
    obs = result.observers_data
    qsnr_by_role = result.qsnr_by_role or {}
    mse_by_role = result.mse_by_role or {}
    tag = f"{label} " if label else ""

    print(f"\n{'='*60}")
    print(f"  {tag}Layer Deep Dive: {layer}")
    print(f"{'='*60}")

    # ── Per-role analysis ──────────────────────────────────────────
    roles_with_data = set()
    for role in ("input", "weight", "output"):
        if obs.get(layer, {}).get(role):
            roles_with_data.add(role)

    for role in sorted(roles_with_data):
        qsnr = qsnr_by_role.get(role, {}).get(layer)
        mse = mse_by_role.get(role, {}).get(layer)

        print(f"\n  [{role.upper()}] QSNR: {qsnr:.1f} dB" if qsnr is not None else f"\n  [{role.upper()}] QSNR: N/A")

        # ── Distribution fingerprint table ─────────────────────────
        dist = _get_dist_metrics(obs, layer, role)
        if dist:
            dist_rows = []
            for key, short in _DIST_KEYS:
                v = dist.get(key)
                if v is not None:
                    if key in ("outlier_ratio", "sparse_ratio"):
                        dist_rows.append({"metric": short, "value": f"{v:.2%}"})
                    elif key == "dynamic_range_bits":
                        dist_rows.append({"metric": short, "value": f"{v:.1f} bits"})
                    else:
                        dist_rows.append({"metric": short, "value": round(v, 3)})
            if dist_rows:
                _chart(dist_rows, "table", x="metric", y="value",
                       label=label, title=f"{tag}{layer} ({role}) Distribution Fingerprint")

        # ── Distribution overlay: fp32 vs quant vs error (dist_overlay) ─
        hist = _get_hist_data(obs, layer, role)
        if hist and "fp32_hist" in hist:
            fp32_hist = _to_list(hist["fp32_hist"])
            n_bins = len(fp32_hist)
            fp32_min = hist.get("fp32_min", 0)
            fp32_max = hist.get("fp32_max", 1)
            bin_width = (fp32_max - fp32_min) / n_bins if n_bins > 0 else 1

            dist_rows = []
            for i in range(n_bins):
                bin_center = round(fp32_min + (i + 0.5) * bin_width, 4)
                row = {"bin": bin_center, "fp32": fp32_hist[i]}
                if "quant_hist" in hist:
                    qh = _to_list(hist["quant_hist"])
                    row["quant"] = qh[i] if i < len(qh) else 0
                if "err_hist" in hist:
                    eh = _to_list(hist["err_hist"])
                    row["error"] = eh[i] if i < len(eh) else 0
                dist_rows.append(row)

            has_quant = any(r.get("quant", 0) != 0 for r in dist_rows)
            has_error = any(r.get("error", 0) != 0 for r in dist_rows)
            series = [
                {"key": "fp32", "type": "area", "fillOpacity": 0.25,
                 "step": True, "label": "FP32", "color": "#5B8DB8"},
            ]
            if has_quant:
                series.append({"key": "quant", "type": "line", "dash": "6 3",
                               "label": "Quant", "color": "#D4605A"})
            if has_error:
                series.append({"key": "error", "type": "area", "axis": "right",
                               "fillOpacity": 0.3, "step": True,
                               "label": "Error", "color": "#9CA3AF"})

            if dist_rows and len(series) > 0:
                _chart(dist_rows, "dist_overlay", x="bin", y="fp32",
                       label=label, title=f"{tag}{layer} ({role}) Distribution",
                       series=series)

        # ── Per-block QSNR stats ───────────────────────────────────
        blocks = _get_per_block_qsnr(obs, layer, role)
        if blocks:
            stats = _block_stats(blocks)
            worst_idx = min(blocks, key=blocks.get)

            block_table = [{
                "layer": layer, "role": role,
                "qsnr_mean": stats["mean"], "qsnr_std": stats["std"],
                "qsnr_min": stats["min"], "qsnr_max": stats["max"],
                "n_blocks": stats["n_blocks"],
                "worst_block": worst_idx, "worst_qsnr": round(blocks[worst_idx], 1),
            }]
            _chart(block_table, "table", x="layer", y="qsnr_mean",
                   label=label, title=f"{tag}{layer} ({role}) Per-Block QSNR Statistics")

            # Top-5 worst blocks bar
            sorted_blocks = sorted(blocks.items(), key=lambda x: x[1])[:5]
            if sorted_blocks:
                bar_data = [{"block_idx": idx, "qsnr_db": round(q, 1)}
                            for idx, q in sorted_blocks]
                _chart(bar_data, "bar", x="block_idx", y="qsnr_db",
                       label=label, title=f"{tag}{layer} ({role}) Top-5 Worst Blocks")

        # ── Classification ─────────────────────────────────────────
        if dist:
            cls_label, desc, suggestion = classify_distribution(dist)
            print(f"    Classification: {cls_label}")
            print(f"    Suggestion: {suggestion}")

    # ── Role attribution summary ───────────────────────────────────
    role_qsnrs = {}
    for role in ("input", "weight", "output"):
        v = qsnr_by_role.get(role, {}).get(layer)
        if v is not None and math.isfinite(v):
            role_qsnrs[role] = v

    if role_qsnrs:
        dominant = min(role_qsnrs, key=role_qsnrs.get)
        attr_row = {
            "layer": layer,
            **{f"{r}_qsnr": round(v, 1) for r, v in role_qsnrs.items()},
            "dominant_error": dominant,
        }
        _chart([attr_row], "table", x="layer", y="dominant_error",
               label=label, title=f"{tag}{layer} Role Attribution: Dominant Error Source")
        print(f"\n  Dominant error source: {dominant} (QSNR={role_qsnrs[dominant]:.1f} dB)")


# =====================================================================
# 2. compare_extreme_layers
# =====================================================================

def compare_extreme_layers(
    result: "SessionResult",
    *,
    top_k: int = 3,
    linear_only: bool = True,
):
    """Compare Top-K worst + Top-K best layers by accumulated QSNR.

    Outputs a summary table, cross-layer block std bar, then calls
    layer_deep_dive for each extreme layer (with dist_overlay).
    """
    accum = _filter_qsnr(result.accum_qsnr_per_layer, linear_only, result.observers_data)
    if not accum:
        accum = _filter_qsnr(result.accum_qsnr_per_layer, False, {})
    if not accum:
        print("[bitx] No accumulated QSNR data.")
        return

    layers = sorted(accum.items(), key=lambda x: x[1])
    worst = layers[:top_k]
    best = layers[-top_k:] if len(layers) > top_k else layers[:]
    best = list(reversed(best))

    # Summary table
    rows = []
    for i, (name, qsnr) in enumerate(worst):
        rows.append({"rank": f"worst-{i+1}", "layer": name, "qsnr_db": round(qsnr, 1)})
    for i, (name, qsnr) in enumerate(best):
        lbl = f"best-{i+1}" if name not in {n for n, _ in worst} else f"best-{i+1}*"
        rows.append({"rank": lbl, "layer": name, "qsnr_db": round(qsnr, 1)})

    _chart(rows, "table", x="rank", y="qsnr_db",
           label="", title=f"Extreme Layers: Top-{top_k} Worst + Top-{top_k} Best (Accum QSNR)")
    print(f"\n[bitx] Extreme layers (worst → best, accum QSNR):")
    for r in rows:
        print(f"  {r['rank']:<10} {r['layer']:<30} QSNR: {r['qsnr_db']:.1f} dB")

    # ── Cross-layer block std comparison ───────────────────────────
    obs = result.observers_data
    allowed = _linear_layer_names(obs) if linear_only else None
    std_rows = []
    for layer_name in sorted(obs.keys()):
        if allowed is not None and layer_name not in allowed:
            continue
        for role in ("input", "weight", "output"):
            blocks = _get_per_block_qsnr(obs, layer_name, role)
            if blocks:
                stats = _block_stats(blocks)
                std_rows.append({
                    "layer": layer_name, "role": role,
                    "block_std": stats["std"],
                    "block_mean": stats["mean"],
                    "block_min": stats["min"],
                    "n_blocks": stats["n_blocks"],
                })
    if std_rows:
        std_rows.sort(key=lambda r: r["block_std"], reverse=True)
        _chart(std_rows, "bar", x="layer", y="block_std", hue="role",
               label="", title="Per-Block QSNR Std Dev (higher = more heterogeneous)")

    # ── Deep dive for each extreme layer ───────────────────────────
    seen = set()
    for rank_label, name, _ in [(r["rank"], r["layer"], r["qsnr_db"]) for r in rows]:
        if name in seen:
            continue
        seen.add(name)
        layer_deep_dive(result, name, label=rank_label)


# =====================================================================
# 3. block_heatmap
# =====================================================================

def block_heatmap(result: "SessionResult", layer: str, role: str = "weight"):
    """Per-block QSNR distribution for one (layer, role).

    Since per-block QSNR is 1D data, uses bar chart (not 2D heatmap)
    to show all blocks sorted by QSNR — reveals the distribution shape.
    """
    obs = result.observers_data
    blocks = _get_per_block_qsnr(obs, layer, role)

    if not blocks:
        print(f"[bitx] No per-block QSNR data for {layer}/{role}. "
              f"(Need PerBlockQSNRObserver attached during Session.run)")
        return

    # All blocks sorted by QSNR (bar) — shows distribution of block-level quality
    sorted_all = sorted(blocks.items(), key=lambda x: x[1])
    # Downsample if too many blocks for readable bar chart
    if len(sorted_all) > 100:
        step = len(sorted_all) // 100
        sampled = sorted_all[::step]
    else:
        sampled = sorted_all
    bar_all = [{"block_idx": idx, "qsnr_db": round(q, 1)} for idx, q in sampled]
    _chart(bar_all, "bar", x="block_idx", y="qsnr_db",
           label="", title=f"{layer} ({role}) All Blocks QSNR Distribution (sorted)")

    # Top-10 worst blocks bar
    top10 = sorted_all[:10]
    bar_worst = [{"block_idx": idx, "qsnr_db": round(q, 1)} for idx, q in top10]
    _chart(bar_worst, "bar", x="block_idx", y="qsnr_db",
           label="", title=f"{layer} ({role}) Top-10 Worst Blocks by QSNR")

    # Stats table
    stats = _block_stats(blocks)
    _chart([{"layer": layer, "role": role, **stats}], "table",
           x="layer", y="mean",
           label="", title=f"{layer} ({role}) Block QSNR Statistics")

    print(f"[bitx] {layer}/{role}: {stats['n_blocks']} blocks, "
          f"QSNR mean={stats['mean']:.1f} std={stats['std']:.1f} "
          f"min={stats['min']:.1f} max={stats['max']:.1f}")


# =====================================================================
# 4. distribution_table
# =====================================================================

def distribution_table(result: "SessionResult"):
    """All-layer distribution fingerprint summary table.

    Requires: DistributionObserver.
    """
    obs = result.observers_data
    rows = []

    for layer in sorted(obs.keys()):
        for role in sorted(obs[layer].keys()):
            dist = _get_dist_metrics(obs, layer, role)
            if not dist:
                continue
            cls_label, _, _ = classify_distribution(dist)
            row = {
                "layer": layer, "role": role,
                "classification": cls_label,
            }
            for key, short in _DIST_KEYS:
                v = dist.get(key)
                if v is not None:
                    if key in ("outlier_ratio", "sparse_ratio"):
                        row[short] = f"{v:.1%}"
                    elif key == "dynamic_range_bits":
                        row[short] = f"{v:.1f}"
                    else:
                        row[short] = round(v, 2)
            rows.append(row)

    if rows:
        _chart(rows, "table", x="layer", y="classification",
               label="", title="Distribution Fingerprint: All Layers × All Roles")
        print(f"\n[bitx] Distribution table: {len(rows)} layer×role entries")
    else:
        print("[bitx] No distribution data. "
              "(Need DistributionObserver attached during Session.run)")


# =====================================================================
# 5. diagnosis_report
# =====================================================================

def diagnosis_report(result: "SessionResult"):
    """All-layer causal analysis report.

    Links distribution features to quantization failure modes.
    Outputs:
    - Causal analysis table (layer × role × QSNR × classification × suggestion)
    - Scatter: crest_factor vs QSNR
    - Scatter: outlier_ratio vs QSNR
    """
    obs = result.observers_data
    qsnr_by_role = result.qsnr_by_role or {}
    rows = []

    for layer in sorted(obs.keys()):
        for role in sorted(obs[layer].keys()):
            qsnr = qsnr_by_role.get(role, {}).get(layer)
            dist = _get_dist_metrics(obs, layer, role)

            cls_label = "no_data"
            suggestion = ""
            if dist:
                cls_label, _, suggestion = classify_distribution(dist)

            row = {
                "layer": layer, "role": role,
                "qsnr_db": round(qsnr, 1) if qsnr is not None and math.isfinite(qsnr) else "N/A",
                "classification": cls_label,
                "suggestion": suggestion[:60] if suggestion else "",
            }

            if dist:
                row["crest"] = round(dist.get("crest_factor", 0), 1)
                row["ol_pct"] = f"{dist.get('outlier_ratio', 0):.1%}"
                row["sparse"] = f"{dist.get('sparse_ratio', 0):.1%}"

            rows.append(row)

    if rows:
        # Sort by QSNR (worst first, non-numeric last)
        def _sort_key(r):
            v = r.get("qsnr_db")
            if isinstance(v, (int, float)):
                return v
            return 9999
        rows.sort(key=_sort_key)

        _chart(rows, "table", x="layer", y="qsnr_db",
               label="", title="Causal Analysis: Layer × Role × QSNR × Diagnosis × Suggestion")

        # ── Scatter: crest_factor vs QSNR ──────────────────────────
        crest_scatter = []
        for r in rows:
            if isinstance(r.get("qsnr_db"), (int, float)) and "crest" in r:
                crest_scatter.append({
                    "layer": r["layer"], "role": r["role"],
                    "crest_factor": r["crest"],
                    "qsnr_db": r["qsnr_db"],
                })
        if crest_scatter:
            _chart(crest_scatter, "scatter", x="crest_factor", y="qsnr_db", hue="role",
                   label="", title="Crest Factor vs QSNR (higher crest = worse quantization)")

        # ── Scatter: outlier_ratio vs QSNR ─────────────────────────
        outlier_scatter = []
        for r in rows:
            if isinstance(r.get("qsnr_db"), (int, float)) and "ol_pct" in r:
                ol_str = r["ol_pct"].rstrip("%")
                try:
                    ol_val = float(ol_str) / 100.0
                except ValueError:
                    continue
                outlier_scatter.append({
                    "layer": r["layer"], "role": r["role"],
                    "outlier_ratio": round(ol_val, 4),
                    "qsnr_db": r["qsnr_db"],
                })
        if outlier_scatter:
            _chart(outlier_scatter, "scatter", x="outlier_ratio", y="qsnr_db", hue="role",
                   label="", title="Outlier Ratio vs QSNR (more outliers = worse quantization)")

        # Print terminal summary
        print(f"\n[bitx] Diagnosis report: {len(rows)} entries")
        print(f"  {'Layer':<25} {'Role':<10} {'QSNR':>8} {'Class':<20} Suggestion")
        print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*20} {'-'*40}")
        for r in rows[:15]:
            q = r.get("qsnr_db", "N/A")
            q_str = f"{q:>8.1f}" if isinstance(q, (int, float)) else f"{q:>8}"
            print(f"  {r['layer']:<25} {r['role']:<10} {q_str} {r['classification']:<20} {r.get('suggestion', '')}")
    else:
        print("[bitx] No diagnosis data available.")
