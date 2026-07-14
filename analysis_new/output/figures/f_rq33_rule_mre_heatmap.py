# F_RQ33_RULE_MRE_HEATMAP: 8x3 heatmap of mean MRE per (base_type, rule) — RQ3.3.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

RULES = ["MEAN", "IRWM", "NN"]

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _build_mat(df_mre, base_types):
    mean_mre = (df_mre
                .groupby(["base_type", "rule"])["value"]
                .mean()
                .reset_index())
    mat = np.full((len(base_types), len(RULES)), np.nan)
    for i, bt in enumerate(base_types):
        for j, rule in enumerate(RULES):
            val = mean_mre[(mean_mre["base_type"] == bt) & (mean_mre["rule"] == rule)]["value"]
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
                ax.text(j, i, f"{v:.3f}",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if v > midpoint else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean MRE (lower = better)", fontsize=8)

    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(df_ens_best_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_rule_mre_heatmap")
    df_mre     = df_ens_best_rq33[df_ens_best_rq33["metric"] == "MRE"]
    base_types = model_order or sorted(df_mre["base_type"].unique())
    mat        = _build_mat(df_mre, base_types)
    _draw(mat, base_types, out_dir,
          "f_rq33_rule_mre_heatmap_all.pdf",
          "Mean MRE per (base type, rule) — best $k$ per scenario, 40 scenarios (RQ3.3)")

def generate_s1(df_ens_best_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_rule_mre_heatmap")
    sub        = _s1_filter(df_ens_best_rq33)
    df_mre     = sub[sub["metric"] == "MRE"]
    base_types = model_order or sorted(df_mre["base_type"].unique())
    mat        = _build_mat(df_mre, base_types)
    _draw(mat, base_types, out_dir,
          "f_rq33_rule_mre_heatmap_s1.pdf",
          "Mean MRE per (base type, rule) — best $k$ per scenario, S1 (RQ3.3)")
