# F_DATASET_CHAMPION: Per-dataset model dominance — who wins most sample sizes (RQ1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(df_singles_best, figures_dir, model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_dataset_champion")
    sub      = df_singles_best[df_singles_best["metric"] == "MRE"]
    models   = model_order   or sorted(sub["model_type"].unique())
    datasets = dataset_order or sorted(sub["dataset"].unique())

    med = (
        sub.groupby(["model_type", "dataset", "sample_size"])["value"]
        .median().reset_index()
    )
    idx     = med.groupby(["dataset", "sample_size"])["value"].idxmin()
    winners = med.loc[idx, ["dataset", "model_type"]]

    counts = (
        winners.groupby(["dataset", "model_type"]).size()
        .unstack(fill_value=0)
        .reindex(index=datasets, columns=models, fill_value=0)
    )

    x     = np.arange(len(datasets))
    n_m   = len(models)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, m in enumerate(models):
        offsets = x + (i - n_m / 2 + 0.5) * width
        ax.bar(offsets, counts[m], width=width * 0.9,
               color=MODEL_COLORS.get(m, "#ccc"), label=m, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Sample sizes won (out of 5)")
    ax.set_title("Per-dataset model dominance on MRE — RQ1")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_yticks(range(6))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_dataset_champion.pdf"))
