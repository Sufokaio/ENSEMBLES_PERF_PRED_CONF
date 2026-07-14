# F_GAP_CLOSE: Heatmap of SK rank improvement from single to ensemble per dataset (RQ2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure, ds_label

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

    n_bt = len(base_types)
    n_ds = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6.0, n_ds * 0.82), max(2.4, n_bt * 0.38)))
    cmap = matplotlib.cm.get_cmap("RdYlGn_r")
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels([ds_label(d) for d in datasets], rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(n_bt))
    ax.set_yticklabels(base_types, fontsize=9)

    for i in range(n_bt):
        for j in range(n_ds):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Δ SK rank", fontsize=9)
    fig.tight_layout(pad=0.4)
    save_figure(fig, os.path.join(out_dir, "f_gap_close_mre.pdf"))
