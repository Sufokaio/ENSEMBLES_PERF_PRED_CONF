"""
F_K_SK_RANK_CURVE: Mean global SK rank vs k per base type (RQ3.2) — Idea 1.

Same 2x4 panel layout as f_k_elbow_mean but y = mean SK rank across all
40 scenarios. SK is run GLOBALLY: all 8x3x9 = 216 (base_type, rule, k)
ensembles compete together per (dataset, sample_size).

Three lines per panel (MEAN/IRWM/NN). Reference line at y = 1.

Reading key:
  Low y (close to 1) -> ensemble is in the globally best group in most scenarios
  Decreasing curve   -> adding learners moves this (base_type, rule) up in the global ranking
  Flat curve         -> k doesn't change this model's global standing

Input: k_sk_ranks DataFrame from compute_k_sk_ranks_global or compute_k_sk_ranks_perrule
  [base_type, rule, dataset, sample_size, k, sk_rank]
"""
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


def _draw(mean_sk, base_types, ks, out_dir, fname, suptitle):
    n_bt  = len(base_types)
    ncols = 4
    nrows = (n_bt + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.0), squeeze=False)

    for idx, bt in enumerate(base_types):
        ax      = axes[idx // ncols][idx % ncols]
        bt_data = mean_sk[mean_sk["base_type"] == bt]

        # Dashed line at best mean SK rank actually achieved by this base type
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
    """Mean SK rank vs k -- all 40 scenarios."""
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
    """Mean SK rank vs k -- S1 scenarios only (per-dataset min sample size)."""
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
