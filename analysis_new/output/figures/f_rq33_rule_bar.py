# F_RQ33_RULE_BAR: Grouped bar chart — mean SK rank per rule, per base type (RQ3.3).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

RULES       = ["MEAN", "IRWM", "NN"]
RULE_COLORS = {"MEAN": "#4878CF", "IRWM": "#6ACC65", "NN": "#D65F5F"}

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _draw(sk_rq33, base_types, out_dir, fname, title):
    agg = (sk_rq33
           .groupby(["base_type", "rule"])["sk_rank"]
           .agg(["mean", "std"])
           .reset_index())

    n_bt  = len(base_types)
    n_r   = len(RULES)
    width = 0.22
    x     = np.arange(n_bt)

    fig, ax = plt.subplots(figsize=(max(6.0, n_bt * 0.9), 4.0))

    for i, rule in enumerate(RULES):
        sub    = agg[agg["rule"] == rule].set_index("base_type")
        means  = [sub.loc[bt, "mean"] if bt in sub.index else np.nan for bt in base_types]
        stds   = [sub.loc[bt, "std"]  if bt in sub.index else 0.0    for bt in base_types]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width,
               label=rule, color=RULE_COLORS[rule],
               yerr=stds, capsize=3, error_kw={"linewidth": 0.8},
               alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(base_types, fontsize=9)
    ax.set_ylabel("Mean SK rank (lower = better)", fontsize=9)
    ax.invert_yaxis()
    ax.legend(title="Rule", fontsize=8, title_fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_rule_bar")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    _draw(sk_rq33, base_types, out_dir,
          "f_rq33_rule_bar_all.pdf",
          "Mean SK rank per rule — best $k$ per scenario, 40 scenarios (RQ3.3)")

def generate_s1(sk_rq33, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_rq33_rule_bar")
    base_types = model_order or sorted(sk_rq33["base_type"].unique())
    sub        = _s1_filter(sk_rq33)
    _draw(sub, base_types, out_dir,
          "f_rq33_rule_bar_s1.pdf",
          "Mean SK rank per rule — best $k$ per scenario, S1 (RQ3.3)")
