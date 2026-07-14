# F15: Win Rate Heatmap — Ensemble vs. Single (RQ2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS_DISPLAY = ["MRE", "MAE", "MBRE", "MIBRE", "SA"]

def generate(wtl_df, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f15")
    models  = model_order or sorted(wtl_df["base_type"].unique())

    mat = np.full((len(models), len(METRICS_DISPLAY)), np.nan)
    for i, model in enumerate(models):
        for j, metric in enumerate(METRICS_DISPLAY):
            row = wtl_df[(wtl_df["base_type"] == model) & (wtl_df["metric"] == metric)]
            if not row.empty:
                mat[i, j] = float(row.iloc[0]["win_rate"]) * 100

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cmap = matplotlib.cm.get_cmap("RdBu")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(METRICS_DISPLAY)))
    ax.set_xticklabels(METRICS_DISPLAY, fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("Evaluation metric")
    ax.set_ylabel("Base model type")
    ax.set_title(
        "Ensemble win rate over single (%)\n"
        "Consistent columns = robust advantage; divergent = metric-dependent"
    )

    for i in range(len(models)):
        for j in range(len(METRICS_DISPLAY)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if abs(v - 50) > 22 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Win rate (%)")
    cbar.ax.axhline(50, color="gray", linewidth=1.0, linestyle="--", alpha=0.8)

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f15_winrate_heatmap.pdf"))
