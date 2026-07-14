# F_DATASET_RANK_PROFILES: Per-dataset models x metrics rank heatmap (RQ1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, figures_dir, model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_dataset_rank_profiles")
    models   = model_order   or sorted(df_singles_best["model_type"].unique())
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())

    med = (
        df_singles_best
        .groupby(["model_type", "dataset", "metric"])["value"]
        .median().reset_index()
    )
    med["rank"] = med.groupby(["dataset", "metric"])["value"].rank(method="average")

    ncols = 4
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax  = axes[idx // ncols][idx % ncols]
        sub = med[med["dataset"] == ds]
        pivot = (
            sub.pivot(index="model_type", columns="metric", values="rank")
            .reindex(index=models, columns=METRICS)
        )
        mat = pivot.values.astype(float)

        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=len(models), aspect="auto")
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(METRICS, fontsize=7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7)
        ax.set_title(ds, fontsize=8, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(METRICS)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Model rank per metric per dataset (1=best, 8=worst) — RQ1", fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_dataset_rank_profiles.pdf"))
