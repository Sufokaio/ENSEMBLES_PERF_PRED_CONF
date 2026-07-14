# F17: Base-Type × Combination-Rule Interaction Heatmap (RQ3.1 / RQ3.3 bridge).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

RULES = ["MEAN", "IRWM", "NN"]

def generate(df_ens_rq33, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f17")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())

    sub = df_ens_rq33[df_ens_rq33["metric"] == "MRE"]
    mat = np.full((len(models), 3), np.nan)
    for i, model in enumerate(models):
        for j, rule in enumerate(RULES):
            vals = sub[(sub["base_type"] == model) & (sub["rule"] == rule)]["value"].values
            if len(vals) > 0:
                mat[i, j] = float(np.median(vals))

    mat_norm = np.full_like(mat, np.nan)
    for i in range(len(models)):
        row = mat[i]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        rmin, rmax = valid.min(), valid.max()
        rng = rmax - rmin if rmax > rmin else 1e-9
        mat_norm[i] = (row - rmin) / rng

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    ax = axes[0]
    vmin = float(np.nanmin(mat)); vmax = float(np.nanmax(mat))
    cmap0 = matplotlib.cm.get_cmap("YlOrRd_r")
    im0 = ax.imshow(mat, cmap=cmap0, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(RULES, fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Absolute median MRE\n(lower = better)")
    for i in range(len(models)):
        for j in range(3):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                        color="black" if v < (vmin + (vmax - vmin) * 0.65) else "white")
    fig.colorbar(im0, ax=ax, fraction=0.05, pad=0.03)

    ax = axes[1]
    cmap1 = matplotlib.cm.get_cmap("RdYlGn_r")
    im1 = ax.imshow(mat_norm, cmap=cmap1, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(RULES, fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([], fontsize=8)
    ax.set_title("Within-model rule preference\n(0 = best rule, 1 = worst rule)")
    for i in range(len(models)):
        for j in range(3):
            v = mat_norm[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if 0.2 < v < 0.8 else "white")
    fig.colorbar(im1, ax=ax, fraction=0.05, pad=0.03)

    fig.suptitle("Base-type × combination rule interaction — median MRE (RQ3.1 / RQ3.3)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f17_type_rule_heatmap.pdf"))

def _draw_type_rule(ax_abs, ax_norm, df_sub, models, title_suffix=""):
    mat = np.full((len(models), 3), np.nan)
    for i, m in enumerate(models):
        for j, rule in enumerate(RULES):
            vals = df_sub[(df_sub["base_type"] == m) & (df_sub["rule"] == rule)]["value"].values
            if len(vals) > 0:
                mat[i, j] = float(np.median(vals))
    mat_norm = np.full_like(mat, np.nan)
    for i in range(len(models)):
        row   = mat[i]; valid = row[~np.isnan(row)]
        if len(valid) == 0: continue
        rmin, rmax = valid.min(), valid.max()
        rng = rmax - rmin if rmax > rmin else 1e-9
        mat_norm[i] = (row - rmin) / rng

    vmin = float(np.nanmin(mat)); vmax = float(np.nanmax(mat))
    im0 = ax_abs.imshow(mat, cmap="YlOrRd_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax_abs.set_xticks(range(3)); ax_abs.set_xticklabels(RULES, fontsize=7)
    ax_abs.set_yticks(range(len(models))); ax_abs.set_yticklabels(models, fontsize=7)
    ax_abs.set_title(f"Absolute{title_suffix}", fontsize=7)
    for i in range(len(models)):
        for j in range(3):
            v = mat[i, j]
            if not np.isnan(v):
                ax_abs.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.5,
                            color="black" if v < (vmin + (vmax-vmin)*0.65) else "white")

    im1 = ax_norm.imshow(mat_norm, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax_norm.set_xticks(range(3)); ax_norm.set_xticklabels(RULES, fontsize=7)
    ax_norm.set_yticks(range(len(models))); ax_norm.set_yticklabels([], fontsize=7)
    ax_norm.set_title(f"Within-model preference{title_suffix}", fontsize=7)
    for i in range(len(models)):
        for j in range(3):
            v = mat_norm[i, j]
            if not np.isnan(v):
                ax_norm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                             color="black" if 0.2 < v < 0.8 else "white")
    return im0, im1

def generate_s1(df_ens_rq33, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f17")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())
    mins    = df_ens_rq33.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    sub     = df_ens_rq33.merge(mins, on="dataset").query("sample_size == _min").drop(columns="_min")
    sub_mre = sub[sub["metric"] == "MRE"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    im0, im1 = _draw_type_rule(axes[0], axes[1], sub_mre, models)
    fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.03)
    fig.colorbar(im1, ax=axes[1], fraction=0.05, pad=0.03)
    fig.suptitle("Base-type × rule interaction — S1 only, median MRE (RQ3.1 / RQ3.3)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f17_type_rule_heatmap_s1.pdf"))

def generate_4metric(df_ens_rq33, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f17")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())

    METRICS_4 = ["MRE", "MAE", "MBRE", "MIBRE"]
    fig, axes = plt.subplots(len(METRICS_4), 2,
                              figsize=(9.5, len(METRICS_4) * 3.0))

    for row_idx, metric in enumerate(METRICS_4):
        sub_m   = df_ens_rq33[df_ens_rq33["metric"] == metric]
        im0, im1 = _draw_type_rule(axes[row_idx, 0], axes[row_idx, 1],
                                    sub_m, models, title_suffix=f" — {metric}")
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.05, pad=0.03)
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.05, pad=0.03)

    fig.suptitle("Base-type × rule interaction across all 4 metrics (RQ3.1 / RQ3.3)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f17_type_rule_4metric.pdf"))
