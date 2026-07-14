# F3: Number-of-Learners Trend (RQ3.2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, RULE_COLORS, save_figure

def generate(df_ensembles_all, figures_dir,
             model_order=None, metric="MRE", agg="median"):
    out_dir = os.path.join(figures_dir, "f3")
    models  = model_order or sorted(df_ensembles_all["base_type"].unique())
    rules   = ["MEAN", "IRWM", "NN"]
    k_vals  = sorted(df_ensembles_all["k"].unique())
    fn      = np.median if agg == "median" else np.mean

    sub = df_ensembles_all[df_ensembles_all["metric"] == metric]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

    for col, rule in enumerate(rules):
        ax = axes[col]
        rule_sub = sub[sub["rule"] == rule]

        for model in models:
            ys = []
            for k in k_vals:
                vals = rule_sub[
                    (rule_sub["base_type"] == model) & (rule_sub["k"] == k)
                ]["value"].values
                ys.append(float(fn(vals)) if len(vals) > 0 else np.nan)
            ax.plot(k_vals, ys, marker="o", markersize=3, linewidth=1.2,
                    color=MODEL_COLORS.get(model, "#333333"), label=model)

        ax.set_title(f"Rule: {rule}")
        ax.set_xlabel("Ensemble size $k$")
        if col == 0:
            ax.set_ylabel(f"{metric} ({agg})")
        ax.set_xticks(k_vals)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1),
               fontsize=7, frameon=True)
    fig.suptitle(f"Effect of ensemble size on {metric} ({agg} across all datasets)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f3_k_trend_{metric.lower()}_{agg}.pdf"))

    _f3b_single_panel(sub, rules, k_vals, fn, metric, agg, out_dir)

def _f3b_single_panel(sub, rules, k_vals, fn, metric, agg, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for rule in rules:
        ys = []
        for k in k_vals:
            vals = sub[(sub["rule"] == rule) & (sub["k"] == k)]["value"].values
            ys.append(float(fn(vals)) if len(vals) > 0 else np.nan)
        ax.plot(k_vals, ys, marker="o", markersize=4, linewidth=1.5,
                color=RULE_COLORS.get(rule, "#333"), label=rule)
    ax.set_xlabel("Ensemble size $k$")
    ax.set_ylabel(f"{metric} ({agg})")
    ax.set_xticks(k_vals)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(f"K-trend (all base types aggregated) — {metric}")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f3b_k_trend_agg_{metric.lower()}_{agg}.pdf"))
