"""
F_RQ33_COMBINED_HEATMAP: 8x3 heatmap with mean MRE and mean SK rank in each cell.

Rows = 8 base types, cols = MEAN / IRWM / NN.
Cell top line    : mean MRE across all evaluation scenarios.
Cell bottom line : (mean SK rank) in parentheses, italic.
Background color : mean SK rank — low (green) = statistically better.

Inputs
------
sk_rq33          : [base_type, rule, dataset, sample_size, sk_rank]
df_ens_best_rq33 : long-format ensemble df from select_best_ensembles_rq33
"""
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


def _build_mats(sk_rq33, df_ens, base_types):
    """Return (mre_mat, sk_mat), each shape (len(base_types), 3)."""
    mean_sk = (sk_rq33
               .groupby(["base_type", "rule"])["sk_rank"]
               .mean()
               .reset_index())

    mre_df  = df_ens[df_ens["metric"] == "MRE"]
    mean_mre = (mre_df
                .groupby(["base_type", "rule"])["value"]
                .mean()
                .reset_index())

    sk_mat  = np.full((len(base_types), len(RULES)), np.nan)
    mre_mat = np.full((len(base_types), len(RULES)), np.nan)

    for i, bt in enumerate(base_types):
        for j, rule in enumerate(RULES):
            sk_row  = mean_sk[(mean_sk["base_type"] == bt)  & (mean_sk["rule"] == rule)]["sk_rank"]
            mre_row = mean_mre[(mean_mre["base_type"] == bt) & (mean_mre["rule"] == rule)]["value"]
            if not sk_row.empty:
                sk_mat[i, j]  = sk_row.values[0]
            if not mre_row.empty:
                mre_mat[i, j] = mre_row.values[0]

    return mre_mat, sk_mat


def _draw(mre_mat, sk_mat, base_types, out_dir, fname, title):
    vmin     = np.nanmin(sk_mat)
    vmax     = np.nanmax(sk_mat)
    midpoint = (vmin + vmax) / 2

    fig, ax = plt.subplots(figsize=(4.5, len(base_types) * 0.78 + 1.3))

    im = ax.imshow(sk_mat, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(RULES, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(base_types)))
    ax.set_yticklabels(base_types, fontsize=9)

    for i in range(len(base_types)):
        for j in range(len(RULES)):
            sk_v  = sk_mat[i, j]
            mre_v = mre_mat[i, j]
            if np.isnan(sk_v):
                continue
            txt_color = "white" if sk_v > midpoint else "black"
            mre_str = f"{mre_v:.2f}" if not np.isnan(mre_v) else "?"
            # MRE on top half of cell
            ax.text(j, i - 0.17, mre_str,
                    ha="center", va="center", fontsize=8.0,
                    color=txt_color, fontweight="bold")
            # SK rank on bottom half, smaller and italic
            ax.text(j, i + 0.17, f"({sk_v:.2f})",
                    ha="center", va="center", fontsize=7.0,
                    color=txt_color, style="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean SK rank (lower = better)", fontsize=8)

    ax.set_title(title, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))


def generate(sk_rq33, df_ens_best_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_combined_heatmap")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    mre_mat, sk_mat = _build_mats(sk_rq33, df_ens_best_rq33, base_types)
    _draw(mre_mat, sk_mat, base_types, out_dir,
          "f_rq33_combined_heatmap_all.pdf",
          r"Mean MRE (SK rank) per (base type, rule) — best $k$, 40 scenarios (RQ3.3)")


def generate_s1(sk_rq33, df_ens_best_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_combined_heatmap")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    sk_s1  = _s1_filter(sk_rq33)
    ens_s1 = _s1_filter(df_ens_best_rq33)
    mre_mat, sk_mat = _build_mats(sk_s1, ens_s1, base_types)
    _draw(mre_mat, sk_mat, base_types, out_dir,
          "f_rq33_combined_heatmap_s1.pdf",
          r"Mean MRE (SK rank) per (base type, rule) — best $k$, S1 (RQ3.3)")
