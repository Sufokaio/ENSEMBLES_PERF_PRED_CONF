# F19: SA vs. Ensemble Size k (RQ3.2 / C2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, save_figure
from aggregators.comparisons import add_ensemble_sa_d

def generate(df_ens_raw, df_baseline, figures_dir, sel_agg="median"):
    out_dir = os.path.join(figures_dir, "f19")
    fn      = np.median if sel_agg == "median" else np.mean
    rules   = ["MEAN", "IRWM", "NN"]
    k_vals  = sorted(df_ens_raw["k"].unique())

    ens_aug = add_ensemble_sa_d(df_ens_raw, df_baseline)
    sa_sub  = ens_aug[ens_aug["metric"] == "SA"]

    sa5_ref = float(df_baseline["SA_5"].mean()) if "SA_5" in df_baseline.columns else None

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for rule in rules:
        ys = []
        for k in k_vals:
            vals = sa_sub[(sa_sub["rule"] == rule) & (sa_sub["k"] == k)]["value"].values
            ys.append(float(fn(vals)) if len(vals) > 0 else np.nan)
        ax.plot(k_vals, ys,
                marker=RULE_MARKERS.get(rule, "o"), markersize=5, linewidth=1.5,
                color=RULE_COLORS.get(rule, "#333"), label=rule)

    ax.axhline(0, color="red", linewidth=1.0, linestyle="-",
               alpha=0.8, label="SA = 0 (no better than random)")
    if sa5_ref is not None:
        ax.axhline(sa5_ref, color="gray", linewidth=0.9, linestyle="--",
                   label=f"Baseline SA₅ ≈ {sa5_ref:.3f}")

    ax.set_xlabel("Ensemble size k")
    ax.set_ylabel(f"Mean SA  [{sel_agg} across base × dataset × sample_size]")
    ax.set_xticks(k_vals)
    ax.set_title(
        "SA vs. ensemble size k (RQ3.2)\n"
        "Does a larger ensemble improve the random-baseline distance?"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f19_k_sa_trend.pdf"))
