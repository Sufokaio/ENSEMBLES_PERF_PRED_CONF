# F6: MIBRE vs. MRE Disagreement Scatter (C2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(df_singles_best, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f6")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    med = (
        df_singles_best[df_singles_best["metric"].isin(["MRE", "MIBRE"])]
        .groupby(["model_type", "dataset", "sample_size", "metric"])["value"]
        .median()
        .reset_index()
    )
    pivot = med.pivot(
        index=["model_type", "dataset", "sample_size"], columns="metric", values="value"
    ).reset_index()

    records = []
    for (ds, ss), grp in pivot.groupby(["dataset", "sample_size"]):
        g = grp.copy()
        g["mre_rank"]   = g["MRE"].rank(method="min")
        g["mibre_rank"] = g["MIBRE"].rank(method="min")
        records.append(g)
    ranked = pd.concat(records, ignore_index=True)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for model in models:
        sub = ranked[ranked["model_type"] == model]
        ax.scatter(sub["mre_rank"], sub["mibre_rank"],
                   color=MODEL_COLORS.get(model, "#333"),
                   label=model, s=18, alpha=0.7, edgecolors="none")

    max_rank = len(models)
    ax.plot([1, max_rank], [1, max_rank], color="gray", linewidth=1.0,
            linestyle="--", label="Perfect agreement")

    ax.set_xlabel("MRE rank (within scenario)")
    ax.set_ylabel("MIBRE rank (within scenario)")
    ax.set_title("Metric disagreement: MRE vs. MIBRE ranks\n(off-diagonal = protocol changes conclusions)")
    ax.set_xticks(range(1, max_rank + 1))
    ax.set_yticks(range(1, max_rank + 1))
    ax.legend(fontsize=7, markerscale=1.4, loc="upper left", framealpha=0.8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    n_total = len(ranked)
    n_agree = int((ranked["mre_rank"] == ranked["mibre_rank"]).sum())
    ax.text(0.02, 0.97, f"{n_total - n_agree}/{n_total} points off-diagonal",
            transform=ax.transAxes, fontsize=7, va="top")

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f6_mibre_mre_scatter.pdf"))
