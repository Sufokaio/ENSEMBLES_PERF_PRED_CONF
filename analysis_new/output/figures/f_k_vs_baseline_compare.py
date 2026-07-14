# F_K_VS_BASELINE_COMPARE: side-by-side heatmap comparing global (216) vs per-rule (72) SK competition (RQ3.2).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from .plot_utils import save_figure

RULES      = ["MEAN", "IRWM", "NN"]
K_COMPARE  = list(range(3, 11))

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _pct_better_mat(k_sk_ranks, base_types, ks_compare, rule):
    sub = k_sk_ranks[k_sk_ranks["rule"] == rule]
    mat = np.full((len(base_types), len(ks_compare)), np.nan)

    for (bt, ds, ss), grp in sub.groupby(["base_type", "dataset", "sample_size"]):
        rank_k2 = grp[grp["k"] == 2]["sk_rank"]
        if rank_k2.empty:
            continue
        r2 = int(rank_k2.values[0])
        for j, k in enumerate(ks_compare):
            rk_row = grp[grp["k"] == k]["sk_rank"]
            if rk_row.empty:
                continue
            rk = int(rk_row.values[0])
            i = base_types.index(bt)
            if np.isnan(mat[i, j]):
                mat[i, j] = 0.0
            mat[i, j] += (1 if rk < r2 else 0)

    n_scenarios = sub.groupby("base_type")[["dataset", "sample_size"]].apply(
        lambda g: g.drop_duplicates().shape[0]
    ).to_dict()
    for i, bt in enumerate(base_types):
        n = n_scenarios.get(bt, 1)
        for j in range(len(ks_compare)):
            if not np.isnan(mat[i, j]):
                mat[i, j] = mat[i, j] / n * 100

    return mat

def generate(k_sk_global, k_sk_perrule, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_k_vs_baseline_compare")
    base_types = model_order or sorted(k_sk_global["base_type"].unique())

    _draw(k_sk_global, k_sk_perrule, base_types, K_COMPARE,
          out_dir, "f_k_vs_baseline_compare_all.pdf",
          n_scenarios=40)

def generate_s1(k_sk_global, k_sk_perrule, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_k_vs_baseline_compare")
    base_types = model_order or sorted(k_sk_global["base_type"].unique())
    g_s1 = _s1_filter(k_sk_global)
    p_s1 = _s1_filter(k_sk_perrule)

    _draw(g_s1, p_s1, base_types, K_COMPARE,
          out_dir, "f_k_vs_baseline_compare_s1.pdf",
          n_scenarios=8)

def _draw(k_sk_global, k_sk_perrule, base_types, ks_compare,
          out_dir, fname, n_scenarios):
    datasets = [
        (k_sk_global,  f"Global competition (216 ensembles)"),
        (k_sk_perrule, f"Per-rule competition (72 per rule)"),
    ]

    mats = {}
    for row_idx, (data, _) in enumerate(datasets):
        for rule in RULES:
            mats[(row_idx, rule)] = _pct_better_mat(data, base_types, ks_compare, rule)

    all_vals = np.concatenate([m.flatten() for m in mats.values()])
    all_vals = all_vals[~np.isnan(all_vals)]

    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cmap   = matplotlib.cm.get_cmap("YlOrRd", len(levels) - 1)
    norm   = BoundaryNorm(levels, cmap.N)

    fig, axes = plt.subplots(2, 3,
                              figsize=(3 * 4.2, 2 * (len(base_types) * 0.45 + 1.0)),
                              squeeze=False)

    for row_idx, (_, row_label) in enumerate(datasets):
        for col_idx, rule in enumerate(RULES):
            ax  = axes[row_idx][col_idx]
            mat = mats[(row_idx, rule)]
            im  = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

            ax.set_xticks(range(len(ks_compare)))
            ax.set_xticklabels([str(k) for k in ks_compare], fontsize=7)
            ax.set_yticks(range(len(base_types)))
            ax.set_yticklabels(base_types if col_idx == 0 else [], fontsize=8)
            ax.set_xlabel("$k$", fontsize=8)

            if row_idx == 0:
                ax.set_title(rule, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=8)

            for i in range(len(base_types)):
                for j in range(len(ks_compare)):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{int(round(v))}",
                                ha="center", va="center", fontsize=6.5,
                                color="white" if v > 55 else "black")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.02, ticks=levels[::2])
    cbar.set_label(f"% of {n_scenarios} scenarios where $k$ beats $k$=2 statistically",
                   fontsize=8)

    fig.suptitle("SK competition scope comparison — global (216) vs per-rule (72)\n"
                 "% scenarios where $k > 2$ is in a statistically better group than $k$=2  (RQ3.2)",
                 fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))
