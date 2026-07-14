# F7: Effect Size Forest Plot — Δ (D) per model (C2 / C4).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(df_singles_best, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f7")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    sub = df_singles_best[df_singles_best["metric"] == "D"]

    per_scenario = (
        sub.groupby(["model_type", "dataset", "sample_size"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_D"})
    )

    agg = (
        per_scenario.groupby("model_type")["mean_D"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "D_mean", "std": "D_sd"})
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    y_pos = np.arange(len(models))

    for i, model in enumerate(models):
        row = agg[agg["model_type"] == model]
        if row.empty:
            continue
        d_mean = float(row["D_mean"].values[0])
        d_sd   = float(row["D_sd"].values[0]) if not np.isnan(row["D_sd"].values[0]) else 0.0
        color  = MODEL_COLORS.get(model, "#333")
        ax.errorbar(d_mean, y_pos[i], xerr=d_sd,
                    fmt="o", color=color, markersize=6, capsize=3,
                    linewidth=1.2, elinewidth=1.0)

    ax.axvline(0, color="red", linewidth=1.0, linestyle="-", label="D = 0 (random)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean effect size D (± SD across scenarios)\nMore negative = better than random")
    ax.set_title("Effect size vs. random baseline — single models")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f7_forest_plot.pdf"))
