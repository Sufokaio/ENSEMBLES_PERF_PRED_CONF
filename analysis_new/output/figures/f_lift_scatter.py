# F_LIFT_SCATTER: SK rank lift — single vs. ensemble (RQ2/mixed).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(sk_mixed, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f_lift_scatter")
    base_types = model_order or sorted(
        sk_mixed[sk_mixed["kind"] == "single"]["base_type"].unique()
    )

    sub = sk_mixed[sk_mixed["metric"] == "MRE"]
    s_ranks = (
        sub[sub["kind"] == "single"]
        .groupby("base_type")["sk_rank"].mean()
        .reindex(base_types)
    )
    e_ranks = (
        sub[sub["kind"] == "ensemble"]
        .groupby("base_type")["sk_rank"].mean()
        .reindex(base_types)
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    for bt in base_types:
        sx, ey = float(s_ranks[bt]), float(e_ranks[bt])
        color = MODEL_COLORS.get(bt, "#333")
        ax.scatter(sx, ey, color=color, s=80, zorder=5)
        ax.annotate(bt, (sx, ey), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color=color)

    lims_all = [s_ranks.tolist() + e_ranks.tolist()]
    lo = min(v for v in s_ranks.tolist() + e_ranks.tolist() if not np.isnan(v)) - 0.3
    hi = max(v for v in s_ranks.tolist() + e_ranks.tolist() if not np.isnan(v)) + 0.3
    ax.plot([lo, hi], [lo, hi], color="gray", lw=1.0, ls="--", label="No change")
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], alpha=0.06, color="green")
    ax.fill_between([lo, hi], [hi, hi], [lo, hi], alpha=0.06, color="red")

    ax.text(lo + 0.1, lo + 0.05, "Ensemble improved", fontsize=7, color="darkgreen", va="bottom")
    ax.text(lo + 0.1, hi - 0.05, "Ensemble worse", fontsize=7, color="darkred", va="top")

    ax.set_xlabel("Mean SK rank — single (in mixed ranking)")
    ax.set_ylabel("Mean SK rank — ensemble (in mixed ranking)")
    ax.set_title("SK rank lift: single → ensemble (MRE)\nBoth ranked together with all 16 competitors")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_lift_scatter_mre.pdf"))
