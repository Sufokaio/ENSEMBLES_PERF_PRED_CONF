# F12: 4-Metric Ensemble Gain Heatmap (RQ2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, df_ens_best_rq2, figures_dir,
             model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f12")
    models   = model_order   or sorted(df_ens_best_rq2["base_type"].unique())
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())

    mats = {}
    vmax_global = 0.0
    for metric in METRICS_EVAL:
        mat = np.full((len(models), len(datasets)), np.nan)
        for i, model in enumerate(models):
            for j, ds in enumerate(datasets):
                sv = df_singles_best[
                    (df_singles_best["model_type"] == model) &
                    (df_singles_best["dataset"] == ds) &
                    (df_singles_best["metric"] == metric)]["value"].values
                ev = df_ens_best_rq2[
                    (df_ens_best_rq2["base_type"] == model) &
                    (df_ens_best_rq2["dataset"] == ds) &
                    (df_ens_best_rq2["metric"] == metric)]["value"].values
                if len(sv) == 0 or len(ev) == 0:
                    continue
                s_c = float(np.median(sv))
                e_c = float(np.median(ev))
                if abs(s_c) < 1e-12:
                    continue
                mat[i, j] = (s_c - e_c) / abs(s_c) * 100
        mats[metric] = mat
        vmax_global = max(vmax_global, float(np.nanmax(np.abs(mat))))
    vmax_global = max(vmax_global, 5.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0), squeeze=False)
    cmap = matplotlib.cm.get_cmap("RdYlGn")

    for panel_idx, metric in enumerate(METRICS_EVAL):
        ax  = axes[0, panel_idx]
        mat = mats[metric]
        im  = ax.imshow(mat, cmap=cmap, vmin=-vmax_global, vmax=vmax_global, aspect="auto")

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if panel_idx == 0 else [], fontsize=7)
        ax.set_title(metric, fontsize=9, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(datasets)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=5.5,
                            color="black" if abs(v) < vmax_global * 0.55 else "white")

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("imp% (+ = ens wins)")

    fig.suptitle(
        "Ensemble improvement over single (imp%, + = ensemble wins) — all 4 metrics\n"
        "Shared colorscale enables direct metric-to-metric comparison"
    )
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f12_gain_4metric.pdf"))

def generate_s1(df_singles_best, df_ens_best_rq2, figures_dir,
                model_order=None, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f12")
    models   = model_order   or sorted(df_ens_best_rq2["base_type"].unique())
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())

    mins_s = df_singles_best.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    mins_e = df_ens_best_rq2.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    s1_s = df_singles_best.merge(mins_s, on="dataset")
    s1_s = s1_s[s1_s["sample_size"] == s1_s["_min"]].drop(columns="_min")
    s1_e = df_ens_best_rq2.merge(mins_e, on="dataset")
    s1_e = s1_e[s1_e["sample_size"] == s1_e["_min"]].drop(columns="_min")

    mats = {}
    vmax_global = 0.0
    for metric in METRICS_EVAL:
        mat = np.full((len(models), len(datasets)), np.nan)
        for i, model in enumerate(models):
            for j, ds in enumerate(datasets):
                sv = s1_s[(s1_s["model_type"] == model) & (s1_s["dataset"] == ds) &
                          (s1_s["metric"] == metric)]["value"].values
                ev = s1_e[(s1_e["base_type"] == model) & (s1_e["dataset"] == ds) &
                          (s1_e["metric"] == metric)]["value"].values
                if len(sv) == 0 or len(ev) == 0:
                    continue
                s_c, e_c = float(np.median(sv)), float(np.median(ev))
                if abs(s_c) < 1e-12:
                    continue
                mat[i, j] = (s_c - e_c) / abs(s_c) * 100
        mats[metric] = mat
        vmax_global = max(vmax_global, float(np.nanmax(np.abs(mat))))
    vmax_global = max(vmax_global, 5.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0), squeeze=False)
    cmap = matplotlib.cm.get_cmap("RdYlGn")
    for panel_idx, metric in enumerate(METRICS_EVAL):
        ax  = axes[0, panel_idx]
        mat = mats[metric]
        im  = ax.imshow(mat, cmap=cmap, vmin=-vmax_global, vmax=vmax_global, aspect="auto")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if panel_idx == 0 else [], fontsize=7)
        ax.set_title(metric, fontsize=9, fontweight="bold")
        for i in range(len(models)):
            for j in range(len(datasets)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                            color="black" if abs(v) < vmax_global * 0.55 else "white")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("imp% (+ = ens wins)")
    fig.suptitle("Ensemble improvement over single — S1 only (8 scenarios)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f12_gain_4metric_s1.pdf"))
