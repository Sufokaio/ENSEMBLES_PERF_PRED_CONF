# F_K_SK_RANK_CURVE: Mean global SK rank vs k per base type (RQ3.2) — Idea 1.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, MODEL_COLORS, save_figure

RULES = ["MEAN", "IRWM", "NN"]

BASE_MARKERS = {
    "LR":       "o", "SVR":  "s", "RT":       "^",
    "RF":       "D", "KNN":  "v", "KRR":      "P",
    "DeepPerf": "X", "HINNPerf": "*",
}

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _draw(mean_sk, base_types, ks, out_dir, fname, suptitle):
    n_bt  = len(base_types)
    ncols = 4
    nrows = (n_bt + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.0), squeeze=False)

    for idx, bt in enumerate(base_types):
        ax      = axes[idx // ncols][idx % ncols]
        bt_data = mean_sk[mean_sk["base_type"] == bt]

        best_rank = bt_data["sk_rank"].min()
        ax.axhline(best_rank, color="#888888", linewidth=0.9,
                   linestyle="--", zorder=0, label="_nolegend_")

        for rule in RULES:
            rd = bt_data[bt_data["rule"] == rule].sort_values("k")
            ax.plot(rd["k"], rd["sk_rank"],
                    color=RULE_COLORS.get(rule, "#333"),
                    marker=RULE_MARKERS.get(rule, "o"),
                    markersize=4, linewidth=1.4, label=rule)

        ax.set_title(bt, fontsize=9)
        ax.set_xlabel("$k$")
        ax.set_ylabel("Mean SK rank")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.set_ylim(bottom=0.9)
        ax.grid(True, alpha=0.2)

    for idx in range(n_bt, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, title="Rule")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_sk_rank_curve")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    mean_sk    = (k_sk_ranks
                  .groupby(["base_type", "rule", "k"])["sk_rank"]
                  .mean().reset_index())
    ks = sorted(mean_sk["k"].unique())
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw(mean_sk, base_types, ks, out_dir,
          fname=f"f_k_sk_rank_curve_all{suffix}.pdf",
          suptitle=f"Mean SK rank vs $k$ -- all 40 scenarios{tag} (RQ3.2)\n"
                   "Dashed line at 1 = best group threshold")

def generate_s1(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_sk_rank_curve")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    sub_s1     = _s1_filter(k_sk_ranks)
    mean_sk    = (sub_s1
                  .groupby(["base_type", "rule", "k"])["sk_rank"]
                  .mean().reset_index())
    ks = sorted(mean_sk["k"].unique())
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw(mean_sk, base_types, ks, out_dir,
          fname=f"f_k_sk_rank_curve_s1{suffix}.pdf",
          suptitle=f"Mean SK rank vs $k$ -- S1 per dataset{tag} (RQ3.2)\n"
                   "Dashed line at 1 = best group threshold")

def _draw_byrule(mean_sk, base_types, ks, out_dir, fname, suptitle):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), squeeze=False)

    for col, rule in enumerate(RULES):
        ax        = axes[0][col]
        rule_data = mean_sk[mean_sk["rule"] == rule]

        best_rank = rule_data["sk_rank"].min() if not rule_data.empty else 1.0
        ax.axhline(best_rank, color="#888888", linewidth=0.9,
                   linestyle="--", zorder=0, label="_nolegend_")

        for bt in base_types:
            bt_data = rule_data[rule_data["base_type"] == bt].sort_values("k")
            ax.plot(bt_data["k"], bt_data["sk_rank"],
                    color=MODEL_COLORS.get(bt, "#333"),
                    marker=BASE_MARKERS.get(bt, "o"),
                    markersize=5, linewidth=1.5, label=bt)

        ax.set_title(rule, fontsize=11, fontweight="bold")
        ax.set_xlabel("$k$", fontsize=10)
        if col == 0:
            ax.set_ylabel("Mean SK rank", fontsize=10)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    ax_nn = axes[0][2]
    ax_nn.legend(handles, labels,
                 loc="center left", bbox_to_anchor=(1.04, 0.5),
                 fontsize=9, title="Base type", ncol=2,
                 borderaxespad=0)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate_byrule(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_sk_rank_curve")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    mean_sk    = (k_sk_ranks
                  .groupby(["base_type", "rule", "k"])["sk_rank"]
                  .mean().reset_index())
    ks  = sorted(mean_sk["k"].unique())
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw_byrule(mean_sk, base_types, ks, out_dir,
                 fname=f"f_k_sk_rank_curve_all{suffix}_byrule.pdf",
                 suptitle=f"Mean SK rank vs $k$ per rule — 8 base types{tag} (all 40 scenarios, RQ3.2)")

def generate_s1_byrule(k_sk_ranks, figures_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(figures_dir, "f_k_sk_rank_curve")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    sub_s1     = _s1_filter(k_sk_ranks)
    mean_sk    = (sub_s1
                  .groupby(["base_type", "rule", "k"])["sk_rank"]
                  .mean().reset_index())
    ks  = sorted(mean_sk["k"].unique())
    tag = f" [{suffix.strip('_')}]" if suffix else ""
    _draw_byrule(mean_sk, base_types, ks, out_dir,
                 fname=f"f_k_sk_rank_curve_s1{suffix}_byrule.pdf",
                 suptitle=f"Mean SK rank vs $k$ per rule — 8 base types{tag} (S1 per dataset, RQ3.2)")
