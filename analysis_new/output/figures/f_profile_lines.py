# F_PROFILE_LINES: Model MRE profiles across datasets, easy → hard (RQ1).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(df_singles_best, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f_profile_lines")
    models  = model_order or sorted(df_singles_best["model_type"].unique())
    sub     = df_singles_best[df_singles_best["metric"] == "MRE"]

    med_ds = (
        sub.groupby(["model_type", "dataset"])["value"]
        .median()
        .unstack("dataset")
    )

    ds_sorted = med_ds.min(axis=0).sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(ds_sorted))
    for m in models:
        ys = [med_ds.at[m, ds] if ds in med_ds.columns else np.nan for ds in ds_sorted]
        ax.plot(x, ys, marker="o", ms=4, linewidth=1.4,
                color=MODEL_COLORS.get(m, "#333"), label=m)

    ax.set_xticks(range(len(ds_sorted)))
    ax.set_xticklabels(ds_sorted, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Median MRE (across sample sizes)")
    ax.set_xlabel("Dataset (sorted easy → hard by best achievable MRE)")
    ax.set_title("Model performance profiles across datasets — RQ1")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_profile_lines.pdf"))
