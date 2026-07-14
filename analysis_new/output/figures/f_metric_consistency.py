# F_METRIC_CONSISTENCY: Metric-consistency heatmap — models x datasets (RQ1 / Contribution 2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, figures_dir, model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_metric_consistency")
    models   = model_order   or sorted(df_singles_best["model_type"].unique())

    med = (
        df_singles_best
        .groupby(["model_type", "dataset", "metric"])["value"]
        .median().reset_index()
    )
    med["rank"] = med.groupby(["dataset", "metric"])["value"].rank(method="average")

    std_pivot = (
        med.groupby(["model_type", "dataset"])["rank"]
        .std().unstack("dataset")
    )

    ds_order = dataset_order or std_pivot.mean(axis=0).sort_values().index.tolist()
    mat = std_pivot.reindex(index=models, columns=ds_order).values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels(ds_order, rotation=38, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("Dataset (sorted most metric-stable → least)")
    ax.set_title("Metric-consistency score per model × dataset\n"
                 "std(rank across MRE/MAE/MBRE/MIBRE) — low = stable — RQ1")

    vmax = float(np.nanmax(mat))
    for i in range(len(models)):
        for j in range(len(ds_order)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v > vmax * 0.65 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    fig.colorbar(im, ax=ax, fraction=0.03, label="Std of ranks across 4 metrics")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_metric_consistency.pdf"))
