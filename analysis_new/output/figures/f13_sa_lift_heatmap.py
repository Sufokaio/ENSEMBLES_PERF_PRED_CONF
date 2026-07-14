# F13: SA Lift Heatmap — Best Ensemble vs. Single (RQ2 / C2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure
from aggregators.comparisons import add_ensemble_sa_d

def generate(df_singles_best, df_ens_best_rq2, df_baseline, figures_dir,
             model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f13")
    models   = model_order   or sorted(df_ens_best_rq2["base_type"].unique())
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())

    ens_aug = add_ensemble_sa_d(df_ens_best_rq2, df_baseline)

    s_sa = (
        df_singles_best[df_singles_best["metric"] == "SA"]
        .groupby(["model_type", "dataset"])["value"].mean()
        .reset_index().rename(columns={"model_type": "base_type", "value": "SA_single"})
    )
    e_sa = (
        ens_aug[ens_aug["metric"] == "SA"]
        .groupby(["base_type", "dataset"])["value"].mean()
        .reset_index().rename(columns={"value": "SA_ens"})
    )
    merged = s_sa.merge(e_sa, on=["base_type", "dataset"], how="outer")
    merged["lift"] = merged["SA_ens"] - merged["SA_single"]

    mat = np.full((len(models), len(datasets)), np.nan)
    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            row = merged[(merged["base_type"] == model) & (merged["dataset"] == ds)]
            if not row.empty:
                mat[i, j] = float(row["lift"].values[0])

    vmax = float(np.nanmax(np.abs(mat))) if not np.all(np.isnan(mat)) else 0.1
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    cmap = matplotlib.cm.get_cmap("RdYlGn")
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_title(
        "SA lift: best ensemble SA − single SA\n"
        "(green = ensemble further from random; red = single is better vs. random)"
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Base model type")

    for i in range(len(models)):
        for j in range(len(datasets)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < vmax * 0.55 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("SA lift (ens − single)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f13_sa_lift_heatmap.pdf"))
