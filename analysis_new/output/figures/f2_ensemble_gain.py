# F2: Ensemble Gain Heatmap (RQ2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .plot_utils import save_figure

def generate(df_singles_best, df_ens_best, figures_dir,
             model_order=None, dataset_order=None, agg="median"):
    out_dir = os.path.join(figures_dir, "f2")
    models   = model_order   or sorted(df_ens_best["base_type"].unique())
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())

    fn = np.median if agg == "median" else np.mean

    _make_heatmap(df_singles_best, df_ens_best, models, datasets, ["MRE"],
                  fn, agg, "mre", out_dir)

def generate_s1(df_singles_best, df_ens_best, figures_dir,
                model_order=None, dataset_order=None, agg="median"):
    out_dir  = os.path.join(figures_dir, "f2")
    models   = model_order   or sorted(df_ens_best["base_type"].unique())
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())
    fn = np.median if agg == "median" else np.mean

    mins_s = df_singles_best.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    mins_e = df_ens_best.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    s1_s = df_singles_best.merge(mins_s, on="dataset")
    s1_s = s1_s[s1_s["sample_size"] == s1_s["_min"]].drop(columns="_min")
    s1_e = df_ens_best.merge(mins_e, on="dataset")
    s1_e = s1_e[s1_e["sample_size"] == s1_e["_min"]].drop(columns="_min")

    _make_heatmap(s1_s, s1_e, models, datasets, ["MRE"], fn, agg, "mre_s1", out_dir)

def _make_heatmap(df_s, df_e, models, datasets, metrics, fn, agg_label, tag, out_dir):
    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5), squeeze=False)

    for panel_idx, metric in enumerate(metrics):
        ax = axes[0, panel_idx]
        mat = np.full((len(datasets), len(models)), np.nan)

        for i, ds in enumerate(datasets):
            for j, model in enumerate(models):
                s_vals = df_s[(df_s["model_type"] == model) & (df_s["dataset"] == ds) &
                               (df_s["metric"] == metric)]["value"].values
                e_vals = df_e[(df_e["base_type"] == model) & (df_e["dataset"] == ds) &
                               (df_e["metric"] == metric)]["value"].values
                if len(s_vals) == 0 or len(e_vals) == 0:
                    continue
                s_c = float(fn(s_vals))
                e_c = float(fn(e_vals))
                if abs(s_c) < 1e-12:
                    continue
                imp = (s_c - e_c) / abs(s_c) * 100
                mat[i, j] = imp

        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 10
        cmap = matplotlib.cm.get_cmap("RdYlGn")
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets if panel_idx == 0 else [], fontsize=7)
        ax.set_title(f"{metric} imp\\%")

        for i in range(len(datasets)):
            for j in range(len(models)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                            color="black" if abs(v) < vmax * 0.6 else "white")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("imp% (+ = ensemble wins)")

    fig.suptitle(f"Ensemble gain over single ({agg_label})")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f2_gain_heatmap_{tag}.pdf"))
