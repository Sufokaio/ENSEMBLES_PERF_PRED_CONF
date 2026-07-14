# F20: SA by Combination Rule (RQ3.3 / C2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, save_figure
from aggregators.comparisons import add_ensemble_sa_d

RULES = ["MEAN", "IRWM", "NN"]

def generate(df_ens_rq33, df_baseline, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f20")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())

    ens_aug = add_ensemble_sa_d(df_ens_rq33, df_baseline)
    sa_sub  = ens_aug[ens_aug["metric"] == "SA"]
    sa5_ref = float(df_baseline["SA_5"].mean()) if "SA_5" in df_baseline.columns else None

    x       = np.arange(len(models))
    n_rules = len(RULES)
    bar_w   = 0.22
    offsets = np.linspace(-(n_rules - 1) / 2 * bar_w,
                           (n_rules - 1) / 2 * bar_w, n_rules)

    fig, ax = plt.subplots(figsize=(9, 4.2))

    for ri, rule in enumerate(RULES):
        means, sds = [], []
        for model in models:
            vals = sa_sub[(sa_sub["base_type"] == model) & (sa_sub["rule"] == rule)]["value"].values
            means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
            sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        ax.bar(x + offsets[ri], means, bar_w,
               color=RULE_COLORS.get(rule, "#666"), label=rule,
               edgecolor="white", linewidth=0.4)
        ax.errorbar(x + offsets[ri], means, yerr=sds,
                    fmt="none", color="black", capsize=2.5, linewidth=0.9)

    ax.axhline(0, color="red", linewidth=1.0, linestyle="-",
               alpha=0.8, label="SA = 0 (random baseline)")
    if sa5_ref is not None:
        ax.axhline(sa5_ref, color="gray", linewidth=0.9, linestyle="--",
                   label=f"SA₅ (5-learner baseline) ≈ {sa5_ref:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean SA (higher = better)")
    ax.set_title(
        "SA by combination rule (RQ3.3)\n"
        "Does MEAN/IRWM beat random more reliably than NN?"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f20_rule_sa_bar.pdf"))
