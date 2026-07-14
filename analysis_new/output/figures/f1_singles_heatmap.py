# F1: Per-Dataset Rank Heatmap — Singles (RQ1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .plot_utils import save_figure, ds_label

def generate(borda_per_dataset, figures_dir, model_order=None, dataset_order=None):
    out_dir = os.path.join(figures_dir, "f1")
    models   = model_order   or sorted(borda_per_dataset["model_type"].unique())
    datasets = dataset_order or sorted(borda_per_dataset["dataset"].unique())

    pivot = (
        borda_per_dataset
        .pivot(index="dataset", columns="model_type", values="borda_rank")
        .reindex(index=datasets, columns=models)
    )

    n_ranks = len(models)
    n_ds    = len(datasets)
    fig, ax = plt.subplots(figsize=(max(5.5, n_ranks * 0.75), max(2.6, n_ds * 0.42)))

    cmap = matplotlib.cm.get_cmap("YlOrRd", n_ranks)
    mat  = pivot.values.astype(float)
    im   = ax.imshow(mat, cmap=cmap, vmin=0.5, vmax=n_ranks + 0.5, aspect="auto")

    ax.set_xticks(range(n_ranks))
    ax.set_xticklabels(models, rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels([ds_label(d) for d in datasets], fontsize=9)

    for i in range(n_ds):
        for j in range(n_ranks):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Borda rank", fontsize=9)
    cbar.set_ticks(range(1, n_ranks + 1))
    fig.tight_layout(pad=0.4)
    save_figure(fig, os.path.join(out_dir, "f1_rank_heatmap.pdf"))

def generate_s1(sk_singles, figures_dir, model_order=None, dataset_order=None):
    from aggregators.sk_borda import compute_borda_per_dataset

    out_dir  = os.path.join(figures_dir, "f1")

    mins = sk_singles.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    sk_s1 = sk_singles.merge(mins, on="dataset")
    sk_s1 = sk_s1[sk_s1["sample_size"] == sk_s1["_min"]].drop(columns="_min")

    borda_s1 = compute_borda_per_dataset(sk_s1, "model_type")

    models   = model_order   or sorted(borda_s1["model_type"].unique())
    datasets = dataset_order or sorted(borda_s1["dataset"].unique())

    pivot = (
        borda_s1.pivot(index="dataset", columns="model_type", values="borda_rank")
        .reindex(index=datasets, columns=models)
    )

    n_ranks = len(models)
    n_ds    = len(datasets)
    fig, ax = plt.subplots(figsize=(max(5.5, n_ranks * 0.75), max(2.6, n_ds * 0.42)))

    cmap = matplotlib.cm.get_cmap("YlOrRd", n_ranks)
    mat  = pivot.values.astype(float)
    im   = ax.imshow(mat, cmap=cmap, vmin=0.5, vmax=n_ranks + 0.5, aspect="auto")

    ax.set_xticks(range(n_ranks))
    ax.set_xticklabels(models, rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels([ds_label(d) for d in datasets], fontsize=9)

    for i in range(n_ds):
        for j in range(n_ranks):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Borda rank", fontsize=9)
    cbar.set_ticks(range(1, n_ranks + 1))
    fig.tight_layout(pad=0.4)
    save_figure(fig, os.path.join(out_dir, "f1_rank_heatmap_s1.pdf"))
