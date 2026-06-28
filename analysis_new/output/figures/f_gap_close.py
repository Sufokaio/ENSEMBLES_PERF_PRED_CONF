"""
F_GAP_CLOSE: Heatmap of SK rank improvement from single to ensemble per dataset (RQ2).

Cell (base_type, dataset): mean(SK_rank_ensemble) - mean(SK_rank_single) across
sample sizes, using MRE in the MIXED ranking.
Negative (green) = ensemble moved to a better rank cluster.
Positive (red) = ensemble regressed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure


def generate(sk_mixed, figures_dir, model_order=None, dataset_order=None):
    out_dir    = os.path.join(figures_dir, "f_gap_close")
    base_types = model_order   or sorted(sk_mixed["base_type"].unique())
    datasets   = dataset_order or sorted(sk_mixed["dataset"].unique())

    sub = sk_mixed[sk_mixed["metric"] == "MRE"]

    mat = np.full((len(base_types), len(datasets)), np.nan)
    for i, bt in enumerate(base_types):
        for j, ds in enumerate(datasets):
            s = sub[(sub["base_type"] == bt) & (sub["kind"] == "single")  & (sub["dataset"] == ds)]["sk_rank"]
            e = sub[(sub["base_type"] == bt) & (sub["kind"] == "ensemble") & (sub["dataset"] == ds)]["sk_rank"]
            if s.empty or e.empty:
                continue
            mat[i, j] = float(e.mean()) - float(s.mean())

    vmax = max(float(np.nanmax(np.abs(mat))), 0.5)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    cmap = matplotlib.cm.get_cmap("RdYlGn_r")
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(base_types)))
    ax.set_yticklabels(base_types, fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Base model type")
    ax.set_title("SK rank change: ensemble − single (MRE, mixed ranking)\nNegative (green) = ensemble is in a better cluster")

    for i in range(len(base_types)):
        for j in range(len(datasets)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Δ SK rank (ens − single)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_gap_close_mre.pdf"))
