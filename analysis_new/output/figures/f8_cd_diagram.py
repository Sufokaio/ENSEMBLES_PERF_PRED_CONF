# F8: Critical Difference (CD) Diagrams using Nemenyi post-hoc test.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

NEMENYI_Q = {
    2: 1.960,  3: 2.344,  4: 2.569,  5: 2.728,
    6: 2.850,  7: 2.949,  8: 3.031,  9: 3.102, 10: 3.164,
    11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
    24: 3.643,
}

def _get_q(k):
    if k in NEMENYI_Q:
        return NEMENYI_Q[k]
    ks = sorted(NEMENYI_Q.keys())
    for i in range(len(ks) - 1):
        if ks[i] <= k <= ks[i+1]:
            t = (k - ks[i]) / (ks[i+1] - ks[i])
            return NEMENYI_Q[ks[i]] + t * (NEMENYI_Q[ks[i+1]] - NEMENYI_Q[ks[i]])
    return NEMENYI_Q[max(ks)]

def _compute_avg_ranks(df, group_col, metric, lower_better=True):
    sub = df[df["metric"] == metric]
    scenario_ranks = []
    for (ds, ss), grp in sub.groupby(["dataset", "sample_size"]):
        med = grp.groupby(group_col)["value"].median()
        if lower_better:
            ranked = med.rank(method="average")
        else:
            ranked = med.rank(method="average", ascending=False)
        scenario_ranks.append(ranked)
    if not scenario_ranks:
        return pd.Series(dtype=float)
    return pd.concat(scenario_ranks, axis=1).mean(axis=1)

def _draw_cd_diagram(ax, avg_ranks, cd, title, alpha):
    sorted_models = avg_ranks.sort_values().index.tolist()
    sorted_ranks  = avg_ranks.sort_values().values.tolist()
    n = len(sorted_models)

    y_center = 0.5
    ax.set_xlim(sorted_ranks[0] - 0.5, sorted_ranks[-1] + cd + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)

    ax.annotate("", xy=(sorted_ranks[-1] + cd + 0.3, y_center),
                xytext=(sorted_ranks[0] - 0.3, y_center),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.0))

    left_models  = sorted_models[:n // 2]
    right_models = sorted_models[n // 2:][::-1]

    for step, model in enumerate(left_models):
        r = avg_ranks[model]
        y_label = y_center + 0.08 + step * 0.09
        ax.plot([r, r], [y_center, y_label], color="black", lw=0.8)
        ax.plot(r, y_center, "o", color="black", ms=4)
        ax.text(r, y_label + 0.02, f"{model}\n{r:.2f}", ha="center", va="bottom", fontsize=6)

    for step, model in enumerate(right_models):
        r = avg_ranks[model]
        y_label = y_center - 0.08 - step * 0.09
        ax.plot([r, r], [y_center, y_label], color="black", lw=0.8)
        ax.plot(r, y_center, "o", color="black", ms=4)
        ax.text(r, y_label - 0.02, f"{model}\n{r:.2f}", ha="center", va="top", fontsize=6)

    ranks_arr = np.array(sorted_ranks)
    for i in range(n):
        for j in range(i + 1, n):
            if ranks_arr[j] - ranks_arr[i] <= cd:
                ax.plot([ranks_arr[i], ranks_arr[j]],
                        [y_center - 0.035, y_center - 0.035],
                        color="steelblue", lw=3.5, alpha=0.7, solid_capstyle="round")

    r_ref = sorted_ranks[-1] + 0.1
    ax.plot([r_ref, r_ref + cd], [y_center + 0.3, y_center + 0.3], color="black", lw=1.5)
    ax.text(r_ref + cd / 2, y_center + 0.34, f"CD={cd:.2f}", ha="center", fontsize=6.5)

def generate(df_singles_best, df_ens_best_rq2, df_ens_rq33, figures_dir,
             model_order=None, alpha=0.05):
    out_dir = os.path.join(figures_dir, "f8")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    for metric in ["MRE", "MAE", "MBRE", "MIBRE"]:
        k_a   = len(models)
        N_a   = df_singles_best[["dataset", "sample_size"]].drop_duplicates().shape[0]
        cd_a  = _get_q(k_a) * np.sqrt(k_a * (k_a + 1) / (6 * N_a))
        avg_a = _compute_avg_ranks(df_singles_best, "model_type", metric).reindex(models)

        fig, ax = plt.subplots(figsize=(7, 2.8))
        _draw_cd_diagram(ax, avg_a, cd_a,
                         f"Singles — {metric}  (N={N_a}, k={k_a})", alpha)
        fig.suptitle(f"CD diagram (Nemenyi α={alpha})", y=1.01)
        fig.tight_layout()
        fname = f"f8a_cd_singles_{metric.lower()}.pdf"
        save_figure(fig, os.path.join(out_dir, fname))

    s_mre = df_singles_best[df_singles_best["metric"] == "MRE"].copy()
    s_mre["competitor"] = s_mre["model_type"].astype(str) + "_S"
    e_mre = df_ens_best_rq2[df_ens_best_rq2["metric"] == "MRE"].copy()
    e_mre["competitor"] = e_mre["base_type"].astype(str) + "_E"
    mixed = pd.concat([
        s_mre.rename(columns={"model_type": "_mt"})[["dataset","sample_size","metric","competitor","value","run"]],
        e_mre.rename(columns={"base_type":  "_bt"})[["dataset","sample_size","metric","competitor","value","run"]],
    ], ignore_index=True)
    k_b   = mixed["competitor"].nunique()
    N_b   = mixed[["dataset", "sample_size"]].drop_duplicates().shape[0]
    cd_b  = _get_q(k_b) * np.sqrt(k_b * (k_b + 1) / (6 * N_b))
    avg_b = _compute_avg_ranks(mixed, "competitor", "MRE")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    _draw_cd_diagram(ax, avg_b, cd_b, f"Singles (_S) + Ensembles (_E) — MRE  (N={N_b}, k={k_b})", alpha)
    fig.suptitle(f"CD diagram (Nemenyi α={alpha})", y=1.01)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f8b_cd_mixed_mre.pdf"))

    ens33_mre = df_ens_rq33[df_ens_rq33["metric"] == "MRE"].copy()
    ens33_mre["competitor"] = ens33_mre["base_type"].astype(str) + "/" + ens33_mre["rule"].astype(str)
    k_c   = ens33_mre["competitor"].nunique()
    N_c   = ens33_mre[["dataset", "sample_size"]].drop_duplicates().shape[0]
    cd_c  = _get_q(k_c) * np.sqrt(k_c * (k_c + 1) / (6 * N_c))
    avg_c = _compute_avg_ranks(ens33_mre, "competitor", "MRE")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    _draw_cd_diagram(ax, avg_c, cd_c, f"Ensembles by base/rule — MRE  (N={N_c}, k={k_c})", alpha)
    fig.suptitle(f"CD diagram (Nemenyi α={alpha})", y=1.01)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f8c_cd_ens_rule_mre.pdf"))
