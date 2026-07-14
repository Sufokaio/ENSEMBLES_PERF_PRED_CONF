# F_RULE_METRIC_HEATMAP: 3×4 heatmap — rule × metric (RQ3.3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

RULES   = ["MEAN", "IRWM", "NN"]
METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_ens_rq33, figures_dir):
    out_dir = os.path.join(figures_dir, "f_rule_metric_heatmap")

    raw_mat  = np.full((len(RULES), len(METRICS)), np.nan)
    norm_mat = np.full((len(RULES), len(METRICS)), np.nan)

    for j, metric in enumerate(METRICS):
        sub = df_ens_rq33[df_ens_rq33["metric"] == metric]
        medians = []
        for i, rule in enumerate(RULES):
            v = float(sub[sub["rule"] == rule]["value"].median())
            raw_mat[i, j] = v
            medians.append(v)
        mn, mx = min(medians), max(medians)
        span = mx - mn if mx != mn else 1.0
        for i in range(len(RULES)):
            norm_mat[i, j] = (raw_mat[i, j] - mn) / span

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    cmap = matplotlib.cm.get_cmap("YlOrRd")
    im = ax.imshow(norm_mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_yticks(range(len(RULES)))
    ax.set_yticklabels(RULES)
    ax.set_xlabel("Evaluation metric")
    ax.set_ylabel("Combination rule")
    ax.set_title("Rule performance by metric — column-normalized (yellow=best, red=worst)\nAnnotated with raw median across all (base type × dataset × sample size) scenarios")

    for i in range(len(RULES)):
        for j in range(len(METRICS)):
            raw = raw_mat[i, j]
            if not np.isnan(raw):
                if abs(raw) >= 100:
                    txt = f"{raw:.0f}"
                elif abs(raw) >= 1:
                    txt = f"{raw:.2f}"
                else:
                    txt = f"{raw:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black" if norm_mat[i, j] < 0.6 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalized rank (0=best, 1=worst)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_rule_metric_heatmap.pdf"))

def _draw_panel(ax, sub_df, title):
    raw_mat  = np.full((len(RULES), len(METRICS)), np.nan)
    norm_mat = np.full((len(RULES), len(METRICS)), np.nan)
    for j, metric in enumerate(METRICS):
        sub_m = sub_df[sub_df["metric"] == metric]
        vals  = []
        for i, rule in enumerate(RULES):
            v = sub_m[sub_m["rule"] == rule]["value"].median()
            raw_mat[i, j] = float(v) if not pd.isna(v) else np.nan
            vals.append(float(v) if not pd.isna(v) else np.nan)
        valid = [v for v in vals if not np.isnan(v)]
        if len(valid) < 2:
            continue
        mn, mx = min(valid), max(valid)
        span = mx - mn if mx != mn else 1.0
        for i in range(len(RULES)):
            if not np.isnan(raw_mat[i, j]):
                norm_mat[i, j] = (raw_mat[i, j] - mn) / span

    cmap = matplotlib.cm.get_cmap("YlOrRd")
    im = ax.imshow(norm_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels(METRICS, fontsize=6)
    ax.set_yticks(range(len(RULES)));   ax.set_yticklabels(RULES, fontsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold")
    for i in range(len(RULES)):
        for j in range(len(METRICS)):
            v = raw_mat[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}" if abs(v) >= 1 else f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5,
                        color="black" if norm_mat[i, j] < 0.6 else "white")
    return im

def generate_per_base(df_ens_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rule_metric_heatmap")
    base_types = model_order or sorted(df_ens_rq33["base_type"].unique())

    ncols = 4
    nrows = (len(base_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.8), squeeze=False)

    for idx, bt in enumerate(axes.flatten()[:len(base_types)]):
        pass

    for idx, bt in enumerate(base_types):
        ax = axes[idx // ncols][idx % ncols]
        _draw_panel(ax, df_ens_rq33[df_ens_rq33["base_type"] == bt], bt)

    for idx in range(len(base_types), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Rule × metric performance per base type — column-normalized (RQ3.3)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_rule_metric_heatmap_per_base.pdf"))

def generate_s1(df_ens_rq33, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f_rule_metric_heatmap")
    mins    = df_ens_rq33.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    sub_s1  = df_ens_rq33.merge(mins, on="dataset").query("sample_size == _min").drop(columns="_min")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    _draw_panel(ax, sub_s1, "Rule × metric — S1 only (column-normalized)")
    fig.suptitle("Rule performance by metric — S1 scenarios only (RQ3.3)", fontsize=9, y=1.01)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_rule_metric_heatmap_s1.pdf"))
