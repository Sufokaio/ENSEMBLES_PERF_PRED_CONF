# F_RQ33_WINNER_MAP: Per-dataset winning rule matrix (RQ3.3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .plot_utils import save_figure

RULES       = ["MEAN", "IRWM", "NN"]
RULE_COLORS = {"MEAN": "#4878CF", "IRWM": "#6ACC65", "NN": "#D65F5F"}

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _winning_rule(df_mre, base_types, datasets):
    agg = (df_mre
           .groupby(["base_type", "dataset", "rule"])["value"]
           .mean()
           .reset_index())

    rule_mat  = np.full((len(base_types), len(datasets)), "", dtype=object)
    color_mat = np.full((len(base_types), len(datasets)), np.nan)
    rule_idx  = {r: i for i, r in enumerate(RULES)}

    for i, bt in enumerate(base_types):
        for j, ds in enumerate(datasets):
            sub = agg[(agg["base_type"] == bt) & (agg["dataset"] == ds)]
            if sub.empty:
                continue
            winner = sub.loc[sub["value"].idxmin(), "rule"]
            rule_mat[i, j]  = winner
            color_mat[i, j] = rule_idx[winner]

    return rule_mat, color_mat

def _draw(rule_mat, color_mat, base_types, datasets, out_dir, fname, title):
    cmap   = matplotlib.colors.ListedColormap([RULE_COLORS[r] for r in RULES])
    norm   = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(5.5, len(datasets) * 0.9),
                                    len(base_types) * 0.7 + 1.5))
    im = ax.imshow(color_mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(base_types)))
    ax.set_yticklabels(base_types, fontsize=9)
    ax.set_xlabel("Dataset", fontsize=9)

    for i in range(len(base_types)):
        for j in range(len(datasets)):
            r = rule_mat[i, j]
            if r:
                ax.text(j, i, r, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    legend_handles = [Patch(facecolor=RULE_COLORS[r], label=r) for r in RULES]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, -0.18), ncol=3, fontsize=8,
              frameon=False, title="Winning rule")

    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def _winning_rule_sk(sk_rq33, base_types, datasets):
    agg = (sk_rq33
           .groupby(["base_type", "dataset", "rule"])["sk_rank"]
           .mean()
           .reset_index())

    rule_mat  = np.full((len(base_types), len(datasets)), "", dtype=object)
    color_mat = np.full((len(base_types), len(datasets)), np.nan)
    rule_idx  = {r: i for i, r in enumerate(RULES)}

    for i, bt in enumerate(base_types):
        for j, ds in enumerate(datasets):
            sub = agg[(agg["base_type"] == bt) & (agg["dataset"] == ds)]
            if sub.empty:
                continue
            winner = sub.loc[sub["sk_rank"].idxmin(), "rule"]
            rule_mat[i, j]  = winner
            color_mat[i, j] = rule_idx[winner]

    return rule_mat, color_mat

def generate(df_ens_best_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_winner_map")
    df_mre     = df_ens_best_rq33[df_ens_best_rq33["metric"] == "MRE"]
    base_types = model_order or sorted(df_mre["base_type"].unique())
    datasets   = sorted(df_mre["dataset"].unique())

    rule_mat, color_mat = _winning_rule(df_mre, base_types, datasets)
    _draw(rule_mat, color_mat, base_types, datasets, out_dir,
          "f_rq33_winner_map_all.pdf",
          "Winning rule per (base type, dataset) — mean MRE, best $k$ per scenario (RQ3.3)")

def generate_s1(df_ens_best_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_winner_map")
    sub        = _s1_filter(df_ens_best_rq33)
    df_mre     = sub[sub["metric"] == "MRE"]
    base_types = model_order or sorted(df_mre["base_type"].unique())
    datasets   = sorted(df_mre["dataset"].unique())

    rule_mat, color_mat = _winning_rule(df_mre, base_types, datasets)
    _draw(rule_mat, color_mat, base_types, datasets, out_dir,
          "f_rq33_winner_map_s1.pdf",
          "Winning rule per (base type, dataset) — mean MRE, S1 (RQ3.3)")

def generate_sk(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_winner_map")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    datasets   = sorted(sk_rq33["dataset"].unique())

    rule_mat, color_mat = _winning_rule_sk(sk_rq33, base_types, datasets)
    _draw(rule_mat, color_mat, base_types, datasets, out_dir,
          "f_rq33_winner_map_sk_all.pdf",
          "Winning rule per (base type, dataset) — SK rank, best $k$ per scenario (RQ3.3)")

def generate_sk_s1(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_winner_map")
    sub        = _s1_filter(sk_rq33)
    base_types = model_order or sorted(sub["base_type"].unique())
    datasets   = sorted(sub["dataset"].unique())

    rule_mat, color_mat = _winning_rule_sk(sub, base_types, datasets)
    _draw(rule_mat, color_mat, base_types, datasets, out_dir,
          "f_rq33_winner_map_sk_s1.pdf",
          "Winning rule per (base type, dataset) — SK rank, S1 (RQ3.3)")
