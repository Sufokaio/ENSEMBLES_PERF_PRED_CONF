"""
F_K_ELBOW_DATASET: MRE vs k per dataset (RQ3.2).

8 panels (one per dataset), x = k, 3 lines per rule (MEAN/IRWM/NN),
y = median MRE aggregated across all base types.

Answers: does the k effect differ by dataset difficulty / structure?
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, save_figure

RULES = ["MEAN", "IRWM", "NN"]


def generate(df_ens_raw, figures_dir, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_k_elbow_dataset")
    sub      = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    datasets = dataset_order or sorted(sub["dataset"].unique())
    ks       = sorted(sub["k"].unique())

    ncols = 4
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax     = axes[idx // ncols][idx % ncols]
        sub_ds = sub[sub["dataset"] == ds]

        for rule in RULES:
            ys = []
            for k in ks:
                vals = sub_ds[(sub_ds["rule"] == rule) & (sub_ds["k"] == k)]["value"].values
                ys.append(float(np.median(vals)) if len(vals) > 0 else np.nan)
            ax.plot(ks, ys, marker=RULE_MARKERS.get(rule, "o"), ms=3.5,
                    linewidth=1.3, color=RULE_COLORS.get(rule, "#333"), label=rule)

        ax.set_title(ds, fontsize=8, fontweight="bold")
        ax.set_xlabel("k", fontsize=7)
        ax.set_ylabel("Median MRE", fontsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [plt.Line2D([0], [0], color=RULE_COLORS.get(r, "#333"),
                          marker=RULE_MARKERS.get(r, "o"), lw=1.3, label=r)
               for r in RULES]
    fig.legend(handles=handles, loc="lower right", fontsize=8, title="Rule")
    fig.suptitle("MRE vs k per dataset — aggregated across base types (RQ3.2)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_k_elbow_dataset.pdf"))
