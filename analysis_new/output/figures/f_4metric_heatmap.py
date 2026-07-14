# F_4METRIC_HEATMAP: 2x2 grid of models x datasets heatmaps, one panel per metric (RQ1).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, figures_dir, model_order=None, dataset_order=None):
    out_dir = os.path.join(figures_dir, "f_4metric_heatmap")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    sub_mre  = df_singles_best[df_singles_best["metric"] == "MRE"]
    piv_mre  = sub_mre.groupby(["model_type", "dataset"])["value"].median().unstack("dataset")
    ds_order = dataset_order or piv_mre.min(axis=0).sort_values().index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        sub = df_singles_best[df_singles_best["metric"] == metric]
        pivot = (
            sub.groupby(["model_type", "dataset"])["value"]
            .median().unstack("dataset")
            .reindex(index=models, columns=ds_order)
        )
        mat  = pivot.values.astype(float)
        vmax = float(np.nanpercentile(mat[~np.isnan(mat)], 95)) if not np.all(np.isnan(mat)) else 1.0

        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(ds_order)))
        ax.set_xticklabels(ds_order, rotation=38, ha="right", fontsize=6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7)
        ax.set_title(metric, fontsize=9, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(ds_order)):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if v > vmax * 0.65 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=4.5, color=color)

        fig.colorbar(im, ax=ax, fraction=0.04, label=metric)

    fig.suptitle("Median error per model × dataset — all 4 metrics (datasets sorted easy→hard) — RQ1",
                 fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_4metric_heatmap.pdf"))
