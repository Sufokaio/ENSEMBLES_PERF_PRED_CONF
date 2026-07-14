# F_BORDA_MIXED: Total Borda score for all 16 competitors (RQ2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(sk_mixed, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_borda_mixed")
    base_types = model_order or sorted(sk_mixed["base_type"].unique())

    sub = sk_mixed[sk_mixed["metric"].isin(METRICS_EVAL)]
    borda = (
        sub.groupby(["competitor", "base_type", "kind"])["sk_rank"]
        .sum().reset_index().rename(columns={"sk_rank": "borda"})
    )

    x = np.arange(len(base_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    for kind, offset, alpha, label_suffix in [
        ("single",   -width/2, 0.45, " (single)"),
        ("ensemble",  width/2, 1.0,  " (ensemble)"),
    ]:
        heights = []
        for bt in base_types:
            row = borda[(borda["base_type"] == bt) & (borda["kind"] == kind)]
            heights.append(float(row["borda"].values[0]) if len(row) else np.nan)
        colors = [matplotlib.colors.to_rgba(MODEL_COLORS.get(bt, "#333"), alpha) for bt in base_types]
        bars = ax.bar(x + offset, heights, width, color=colors,
                      edgecolor=[MODEL_COLORS.get(bt, "#333") for bt in base_types],
                      linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(base_types, rotation=30, ha="right")
    ax.set_ylabel("Total Borda score (lower = better)")
    ax.set_title("Borda score — singles vs. best ensembles (all 16 competitors together)\nLight = single, dark = ensemble")
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.45, label="Single"),
        Patch(facecolor="gray", alpha=1.0,  label="Ensemble"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_borda_mixed.pdf"))
