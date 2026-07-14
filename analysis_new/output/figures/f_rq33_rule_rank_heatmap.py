# F_RQ33_RULE_RANK_HEATMAP: 8x3 heatmap of mean SK rank per (base_type, rule) — RQ3.3.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from .plot_utils import save_figure

RULES = ["MEAN", "IRWM", "NN"]

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _build_mat(sk_rq33, base_types):
    mean_sk = (sk_rq33
               .groupby(["base_type", "rule"])["sk_rank"]
               .mean()
               .reset_index())
    mat = np.full((len(base_types), len(RULES)), np.nan)
    for i, bt in enumerate(base_types):
        for j, rule in enumerate(RULES):
            val = mean_sk[(mean_sk["base_type"] == bt) & (mean_sk["rule"] == rule)]["sk_rank"]
            if not val.empty:
                mat[i, j] = val.values[0]
    return mat

def _draw(mat, base_types, out_dir, fname, title):
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)

    fig, ax = plt.subplots(figsize=(4.2, len(base_types) * 0.58 + 1.2))

    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(RULES, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(base_types)))
    ax.set_yticklabels(base_types, fontsize=9)

    midpoint = (vmin + vmax) / 2
    for i in range(len(base_types)):
        for j in range(len(RULES)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if v > midpoint else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean SK rank (lower = better)", fontsize=8)

    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_rule_rank_heatmap")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    mat        = _build_mat(sk_rq33, base_types)
    _draw(mat, base_types, out_dir,
          "f_rq33_rule_rank_heatmap_all.pdf",
          "Mean SK rank per (base type, rule) — best $k$ per scenario, 40 scenarios (RQ3.3)")

def generate_s1(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_rule_rank_heatmap")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    sub        = _s1_filter(sk_rq33)
    mat        = _build_mat(sub, base_types)
    _draw(mat, base_types, out_dir,
          "f_rq33_rule_rank_heatmap_s1.pdf",
          "Mean SK rank per (base type, rule) — best $k$ per scenario, S1 (RQ3.3)")
