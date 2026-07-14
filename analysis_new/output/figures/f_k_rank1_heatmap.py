# F_K_RANK1_HEATMAP: % of scenarios where k is in global SK rank-1 group (RQ3.2) — Idea 2.

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

def _pct_rank1(k_sk_ranks, base_types, ks, rules):
    n_scenarios = (k_sk_ranks
                   .groupby(["base_type", "rule"])["dataset"]
                   .nunique().max()
                   * k_sk_ranks.groupby(["base_type", "rule"])["sample_size"]
                   .nunique().max())

    result = {}
    for rule in rules:
        mat = np.full((len(base_types), len(ks)), np.nan)
        sub = k_sk_ranks[k_sk_ranks["rule"] == rule]
        total_scenarios = sub.groupby(["base_type"])[[
            "dataset", "sample_size"]].apply(
                lambda g: g.drop_duplicates().shape[0]).to_dict()
        for i, bt in enumerate(base_types):
            n = total_scenarios.get(bt, 1)
            for j, k in enumerate(ks):
                rows = sub[(sub["base_type"] == bt) & (sub["k"] == k)]
                if rows.empty:
                    continue
                mat[i, j] = float((rows["sk_rank"] == 1).sum()) / n * 100
        result[rule] = mat
    return result

def _draw(pct_mats, base_types, ks, out_dir, fname, suptitle):
    n_rules = len(RULES)
    fig, axes = plt.subplots(1, n_rules,
                              figsize=(n_rules * 3.8, len(base_types) * 0.55 + 1.6),
                              squeeze=False)

    levels = [0, 20, 40, 60, 80, 100]
    cmap   = matplotlib.cm.get_cmap("Blues", len(levels) - 1)
    norm   = BoundaryNorm(levels, cmap.N)

    for col, rule in enumerate(RULES):
        ax  = axes[0][col]
        mat = pct_mats[rule]
        im  = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks], fontsize=8)
        ax.set_yticks(range(len(base_types)))
        ax.set_yticklabels(base_types if col == 0 else [], fontsize=8)
        ax.set_xlabel("$k$", fontsize=8)
        ax.set_title(rule, fontsize=9, fontweight="bold")

        for i in range(len(base_types)):
            for j in range(len(ks)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > 65 else "black")

    cbar = fig.colorbar(im, ax=axes[0], orientation="vertical",
                        fraction=0.03, pad=0.02, ticks=levels)
    cbar.set_label("% scenarios in SK rank-1 group", fontsize=8)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_rank1_heatmap")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    ks         = sorted(k_sk_ranks["k"].unique())
    pct_mats   = _pct_rank1(k_sk_ranks, base_types, ks, RULES)
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw(pct_mats, base_types, ks, out_dir,
          fname=f"f_k_rank1_heatmap_all{suffix}.pdf",
          suptitle=f"% in best SK group — 40 scenarios{tag} (RQ3.2)")

def generate_s1(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_rank1_heatmap")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    sub_s1     = _s1_filter(k_sk_ranks)
    ks         = sorted(sub_s1["k"].unique())
    pct_mats   = _pct_rank1(sub_s1, base_types, ks, RULES)
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw(pct_mats, base_types, ks, out_dir,
          fname=f"f_k_rank1_heatmap_s1{suffix}.pdf",
          suptitle=f"% in best SK group — S1{tag} (RQ3.2)")
