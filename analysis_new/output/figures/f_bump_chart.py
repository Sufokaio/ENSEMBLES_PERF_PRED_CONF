# F_BUMP_CHART: Rank bump chart across metrics per dataset (RQ1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, figures_dir, model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_bump_chart")
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.8), squeeze=False)

    x = range(len(METRICS))

    for idx, ds in enumerate(datasets):
        ax  = axes[idx // ncols][idx % ncols]
        sub = med[med["dataset"] == ds]

        for m in models:
            ys = []
            for metric in METRICS:
                row = sub[(sub["model_type"] == m) & (sub["metric"] == metric)]
                ys.append(float(row["rank"].values[0]) if not row.empty else np.nan)
            ax.plot(x, ys, marker="o", ms=4, linewidth=1.4,
                    color=MODEL_COLORS.get(m, "#333"), label=m, alpha=0.85)
            if not np.isnan(ys[-1]):
                ax.text(len(METRICS) - 1 + 0.05, ys[-1], m,
                        fontsize=5, va="center", color=MODEL_COLORS.get(m, "#333"))

        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(METRICS, fontsize=7)
        ax.set_yticks(range(1, len(models) + 1))
        ax.set_yticklabels(range(1, len(models) + 1), fontsize=6)
        ax.invert_yaxis()
        ax.set_title(ds, fontsize=8, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
        ax.set_xlim(-0.1, len(METRICS) - 0.6)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS.get(m, "#333"), lw=1.4, label=m)
               for m in models]
    fig.legend(handles=handles, loc="lower right", fontsize=7, ncol=2, title="Model")
    fig.suptitle("Rank across metrics per dataset (1=best) — flat line = metric-stable — RQ1",
                 fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_bump_chart.pdf"))
