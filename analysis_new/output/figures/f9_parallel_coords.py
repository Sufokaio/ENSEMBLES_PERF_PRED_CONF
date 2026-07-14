# F9: Parallel Coordinates — Metric Profiles (C2 / RQ1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, figures_dir, model_order=None, agg="median"):
    out_dir = os.path.join(figures_dir, "f9")
    models  = model_order or sorted(df_singles_best["model_type"].unique())
    fn      = np.median if agg == "median" else np.mean

    central = (
        df_singles_best[df_singles_best["metric"].isin(METRICS_EVAL)]
        .groupby(["model_type", "metric"])["value"]
        .agg(fn)
        .reset_index()
        .rename(columns={"value": "central"})
    )

    normed = {}
    for metric in METRICS_EVAL:
        vals = central[central["metric"] == metric]["central"].values
        vmin, vmax = vals.min(), vals.max()
        rng = vmax - vmin if vmax > vmin else 1.0
        normed[metric] = {
            row["model_type"]: 1.0 - (row["central"] - vmin) / rng
            for _, row in central[central["metric"] == metric].iterrows()
        }

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    x = np.arange(len(METRICS_EVAL))

    for model in models:
        y = [normed[m].get(model, np.nan) for m in METRICS_EVAL]
        color = MODEL_COLORS.get(model, "#333")
        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.85,
                marker="o", markersize=4, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS_EVAL)
    ax.set_ylabel("Normalized score (higher = better)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Metric profiles — single models ({agg} over all scenarios)\n"
                 "Crossing lines = metric disagreements")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    for xi in x:
        ax.axvline(xi, color="gray", linewidth=0.5, alpha=0.4)

    for i, metric in enumerate(METRICS_EVAL):
        vals = normed[metric]
        best_model = max(vals, key=vals.get)
        ax.text(i, 1.03, best_model, ha="center", va="bottom", fontsize=5.5,
                color=MODEL_COLORS.get(best_model, "#333"))

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f9_parallel_coords.pdf"))
