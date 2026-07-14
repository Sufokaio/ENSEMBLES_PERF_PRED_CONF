# F_RQ33_SLOPE: Slope graph — mean SK rank per rule at best-k-per-scenario (RQ3.3).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

RULES = ["MEAN", "IRWM", "NN"]
RULE_X = {r: i for i, r in enumerate(RULES)}

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _draw(mean_sk, base_types, out_dir, fname, suptitle):
    cmap   = matplotlib.cm.get_cmap("tab10", len(base_types))
    colors = {bt: cmap(i) for i, bt in enumerate(base_types)}

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for bt in base_types:
        sub = mean_sk[mean_sk["base_type"] == bt].set_index("rule")["sk_rank"]
        xs  = [RULE_X[r] for r in RULES if r in sub.index]
        ys  = [sub[r]    for r in RULES if r in sub.index]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.6,
                color=colors[bt], label=bt, zorder=3)

        last_x = xs[-1]
        last_y = ys[-1]
        ax.text(last_x + 0.05, last_y, bt, fontsize=7.5,
                va="center", color=colors[bt])

    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(RULES, fontsize=10)
    ax.set_ylabel("Mean SK rank (lower = better)", fontsize=9)
    ax.set_xlim(-0.25, len(RULES) - 0.55)
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(suptitle, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_slope")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    mean_sk    = (sk_rq33
                  .groupby(["base_type", "rule"])["sk_rank"]
                  .mean().reset_index())
    _draw(mean_sk, base_types, out_dir,
          "f_rq33_slope_all.pdf",
          "Mean SK rank per combination rule — best $k$ per scenario (RQ3.3, 40 scenarios)")

def generate_s1(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_slope")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    sub        = _s1_filter(sk_rq33)
    mean_sk    = (sub
                  .groupby(["base_type", "rule"])["sk_rank"]
                  .mean().reset_index())
    _draw(mean_sk, base_types, out_dir,
          "f_rq33_slope_s1.pdf",
          "Mean SK rank per combination rule — best $k$ per scenario (RQ3.3, S1)")
