"""
F_GAP_CLOSE2: Single-dataset zoom of SK rank improvement (RQ2).

For the chosen dataset (auto-selected as the one with the highest mean |ΔSK|
across models and metrics, or supplied via `focus_dataset`), shows a grouped bar
chart: x = base model type, grouped bars per metric (MRE/MAE/MBRE/MIBRE).
Bar height = mean(SK_rank_ensemble) − mean(SK_rank_single) across sample sizes.
Negative (below zero line) = ensemble moved to a better SK cluster.

This is the per-dataset "zoom" companion to f_gap_close (which shows all
8 datasets as a heatmap).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS       = ["MRE", "MAE", "MBRE", "MIBRE"]
METRIC_COLORS = {
    "MRE":   "#2166ac",
    "MAE":   "#4dac26",
    "MBRE":  "#d01c8b",
    "MIBRE": "#e08214",
}


def _compute_deltas(sk_mixed):
    """Return DataFrame with columns [base_type, dataset, metric, delta]."""
    records = []
    for metric in METRICS:
        sub = sk_mixed[sk_mixed["metric"] == metric]
        for (bt, ds), grp in sub.groupby(["base_type", "dataset"]):
            s = grp[grp["kind"] == "single"]["sk_rank"]
            e = grp[grp["kind"] == "ensemble"]["sk_rank"]
            if s.empty or e.empty:
                continue
            records.append({
                "base_type": bt,
                "dataset":   ds,
                "metric":    metric,
                "delta":     float(e.mean()) - float(s.mean()),
            })
    return pd.DataFrame(records)


def generate(sk_mixed, figures_dir, model_order=None, focus_dataset=None):
    """Grouped bar chart for one dataset, all 4 metrics."""
    out_dir    = os.path.join(figures_dir, "f_gap_close2")
    base_types = model_order or sorted(sk_mixed["base_type"].unique())

    df = _compute_deltas(sk_mixed)

    if focus_dataset is None:
        score = df.groupby("dataset")["delta"].apply(lambda x: x.abs().mean())
        focus_dataset = score.idxmax()

    df_ds = df[df["dataset"] == focus_dataset]

    x       = np.arange(len(base_types))
    n_met   = len(METRICS)
    width   = 0.18
    offsets = np.linspace(-(n_met - 1) / 2, (n_met - 1) / 2, n_met) * width

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, metric in enumerate(METRICS):
        ys = []
        for bt in base_types:
            row = df_ds[(df_ds["base_type"] == bt) & (df_ds["metric"] == metric)]
            ys.append(float(row["delta"].values[0]) if len(row) > 0 else 0.0)
        ax.bar(x + offsets[i], ys, width, label=metric,
               color=METRIC_COLORS[metric], alpha=0.82)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(base_types, fontsize=8)
    ax.set_ylabel("Δ SK rank (ensemble − single)")
    ax.set_title(
        f"SK rank change: best ensemble vs single — {focus_dataset} (RQ2)\n"
        "Negative = ensemble improved; bars grouped by metric"
    )
    ax.legend(title="Metric", fontsize=7, title_fontsize=7, loc="lower left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f_gap_close2_{focus_dataset}.pdf"))
    return focus_dataset
