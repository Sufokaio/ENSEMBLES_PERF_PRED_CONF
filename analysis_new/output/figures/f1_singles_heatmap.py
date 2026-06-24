"""
F1: Per-Dataset Rank Heatmap — Singles (RQ1).

Matrix: 8 rows (datasets) × 8 cols (model types).
Color = Borda rank of that model on that dataset (aggregated over 5 sample sizes).
Light tint = rank 1 (best). Dark tint = rank 8 (worst).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .plot_utils import save_figure


def generate(borda_per_dataset, figures_dir, model_order=None, dataset_order=None):
    """
    Parameters
    ----------
    borda_per_dataset : [dataset, model_type, borda_total, borda_rank]
    """
    out_dir = os.path.join(figures_dir, "f1")
    models   = model_order   or sorted(borda_per_dataset["model_type"].unique())
    datasets = dataset_order or sorted(borda_per_dataset["dataset"].unique())

    pivot = (
        borda_per_dataset
        .pivot(index="dataset", columns="model_type", values="borda_rank")
        .reindex(index=datasets, columns=models)
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    # 8 possible rank levels: light (1=best) → dark (8=worst)
    n_ranks = len(models)
    cmap = matplotlib.cm.get_cmap("YlOrRd", n_ranks)

    mat = pivot.values.astype(float)
    im  = ax.imshow(mat, cmap=cmap, vmin=0.5, vmax=n_ranks + 0.5, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Model type")
    ax.set_ylabel("Dataset")
    ax.set_title("Borda rank per dataset (1 = best)")

    # Annotate cells with rank value
    for i in range(len(datasets)):
        for j in range(len(models)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Borda rank")
    cbar.set_ticks(range(1, n_ranks + 1))

    save_figure(fig, os.path.join(out_dir, "f1_rank_heatmap.pdf"))
