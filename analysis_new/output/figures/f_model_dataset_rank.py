# F_MODEL_DATASET_RANK: Dataset x sample-size rank heatmap for top-2 models (RQ1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

TOP_MODELS = ["HINNPerf", "DeepPerf"]

def _add_sample_rank(df):
    df    = df.copy()
    sizes = (
        df[["dataset", "sample_size"]].drop_duplicates()
        .sort_values(["dataset", "sample_size"])
    )
    sizes["sample_rank"] = sizes.groupby("dataset").cumcount() + 1
    return df.merge(sizes, on=["dataset", "sample_size"])

def generate(df_singles_best, figures_dir, model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_model_dataset_rank")
    sub      = df_singles_best[df_singles_best["metric"] == "MRE"].copy()
    sub      = _add_sample_rank(sub)
    all_mdls = model_order or sorted(sub["model_type"].unique())
    datasets = dataset_order or sorted(sub["dataset"].unique())

    med = sub.groupby(["model_type", "dataset", "sample_rank"])["value"].median().reset_index()
    med["rank"] = med.groupby(["dataset", "sample_rank"])["value"].rank(method="average")

    n_tiers     = sorted(med["sample_rank"].unique())
    tier_labels = [f"S{i}" for i in n_tiers]

    focus = [m for m in TOP_MODELS if m in med["model_type"].unique()]
    if not focus:
        print(f"  f_model_dataset_rank: none of {TOP_MODELS} found in data, skipping")
        return

    fig, axes = plt.subplots(1, len(focus), figsize=(len(focus) * 4.5, 3.8), squeeze=False)

    for ax, m in zip(axes[0], focus):
        pv = (
            med[med["model_type"] == m]
            .pivot(index="dataset", columns="sample_rank", values="rank")
            .reindex(index=datasets, columns=n_tiers)
        )
        mat = pv.values.astype(float)
        im  = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=len(all_mdls), aspect="auto")
        ax.set_xticks(range(len(n_tiers)))
        ax.set_xticklabels(tier_labels)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_title(m)
        ax.set_xlabel("Sample size tier")
        for i in range(len(datasets)):
            for j in range(len(n_tiers)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.05, label="Rank (1=best)")

    fig.suptitle(f"Dataset × sample-size rank: {' vs '.join(focus)} — RQ1")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_model_dataset_rank.pdf"))
