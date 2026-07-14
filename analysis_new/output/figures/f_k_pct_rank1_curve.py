# F_K_PCT_RANK1_CURVE: % scenarios in SK rank-1 group vs k, per base type (RQ3.2).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, save_figure

RULES = ["MEAN", "IRWM", "NN"]

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _pct_rank1(k_sk_ranks):
    rows = []
    for (bt, rule, k), grp in k_sk_ranks.groupby(["base_type", "rule", "k"]):
        n = grp.groupby(["dataset", "sample_size"]).ngroups
        pct = float((grp["sk_rank"] == 1).sum()) / n * 100 if n > 0 else np.nan
        rows.append({"base_type": bt, "rule": rule, "k": k, "pct_rank1": pct})
    import pandas as pd
    return pd.DataFrame(rows)

def _draw(pct_df, base_types, ks, out_dir, fname, suptitle):
    ncols = 4
    nrows = (len(base_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.0), squeeze=False)

    for idx, bt in enumerate(base_types):
        ax      = axes[idx // ncols][idx % ncols]
        bt_data = pct_df[pct_df["base_type"] == bt]

        best_pct = bt_data["pct_rank1"].max()
        ax.axhline(best_pct, color="#888888", linewidth=0.9,
                   linestyle="--", zorder=0, label="_nolegend_")

        for rule in RULES:
            rd = bt_data[bt_data["rule"] == rule].sort_values("k")
            ax.plot(rd["k"], rd["pct_rank1"],
                    color=RULE_COLORS.get(rule, "#333"),
                    marker=RULE_MARKERS.get(rule, "o"),
                    markersize=4, linewidth=1.4, label=rule)

        ax.set_title(bt, fontsize=9)
        ax.set_xlabel("$k$", fontsize=8)
        ax.set_ylabel("% in rank-1 group", fontsize=8)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.2)

    for idx in range(len(base_types), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, title="Rule")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_pct_rank1_curve")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    pct_df     = _pct_rank1(k_sk_ranks)
    ks         = sorted(pct_df["k"].unique())
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw(pct_df, base_types, ks, out_dir,
          fname=f"f_k_pct_rank1_curve_all{suffix}.pdf",
          suptitle=f"% in best SK group vs $k$ — 40 scenarios{tag} (RQ3.2)\n"
                   "Dashed = best % achieved by this base type across all rules/k")

def generate_s1(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_pct_rank1_curve")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    sub_s1     = _s1_filter(k_sk_ranks)
    pct_df     = _pct_rank1(sub_s1)
    ks         = sorted(pct_df["k"].unique())
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw(pct_df, base_types, ks, out_dir,
          fname=f"f_k_pct_rank1_curve_s1{suffix}.pdf",
          suptitle=f"% in best SK group vs $k$ — S1{tag} (RQ3.2)\n"
                   "Dashed = best % achieved by this base type across all rules/k")
