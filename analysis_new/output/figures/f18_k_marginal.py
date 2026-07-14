# F18: Optimal k Analysis — Two Panels (RQ3.2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, save_figure

def generate(df_ens_raw, figures_dir, sel_agg="median"):
    out_dir = os.path.join(figures_dir, "f18")
    fn      = np.median if sel_agg == "median" else np.mean
    rules   = ["MEAN", "IRWM", "NN"]

    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"]

    agg = (
        sub.groupby(["base_type", "rule", "dataset", "sample_size", "k"])["value"]
        .agg(fn).reset_index().rename(columns={"value": "agg_val"})
    )
    best_k_idx = agg.groupby(["base_type", "rule", "dataset", "sample_size"])["agg_val"].idxmin()
    best_k_series = agg.loc[best_k_idx, "k"].values

    k_vals = sorted(sub["k"].unique())
    marginal = {rule: [] for rule in rules}
    k_diff_vals = []
    for i in range(1, len(k_vals)):
        k_curr, k_prev = k_vals[i], k_vals[i - 1]
        if k_curr - k_prev != 1:
            continue
        k_diff_vals.append(k_curr)
        for rule in rules:
            cv = sub[(sub["rule"] == rule) & (sub["k"] == k_curr)]["value"].values
            pv = sub[(sub["rule"] == rule) & (sub["k"] == k_prev)]["value"].values
            if len(cv) > 0 and len(pv) > 0:
                marginal[rule].append(float(fn(cv)) - float(fn(pv)))
            else:
                marginal[rule].append(np.nan)

    fig, (ax_hist, ax_mg) = plt.subplots(1, 2, figsize=(10, 3.8))

    k_min, k_max = int(best_k_series.min()), int(best_k_series.max())
    bins = np.arange(k_min - 0.5, k_max + 1.5, 1)
    ax_hist.hist(best_k_series, bins=bins, color="#4878d0", edgecolor="white", linewidth=0.6)
    mode_k = int(np.bincount(best_k_series.astype(int)).argmax())
    ax_hist.axvline(mode_k, color="red", linewidth=1.2, linestyle="--",
                    label=f"mode k = {mode_k}")
    ax_hist.set_xlabel("Optimal k (best ensemble size)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of selected best k\n(across all base×rule×dataset×sample_size)")
    ax_hist.set_xticks(range(k_min, k_max + 1))
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    for rule in rules:
        ax_mg.plot(k_diff_vals, marginal[rule],
                   marker=RULE_MARKERS.get(rule, "o"), markersize=5, linewidth=1.5,
                   color=RULE_COLORS.get(rule, "#333"), label=rule)
    ax_mg.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.7,
                  label="Zero gain threshold")
    ax_mg.set_xlabel("Ensemble size k")
    ax_mg.set_ylabel(f"MRE(k) − MRE(k−1)  [{sel_agg}]")
    ax_mg.set_title("Marginal gain from adding one learner\n(negative = adding k still helps)")
    ax_mg.set_xticks(k_diff_vals)
    ax_mg.legend(fontsize=8)
    ax_mg.grid(True, alpha=0.25, linewidth=0.5)

    fig.suptitle("Ensemble size analysis (RQ3.2)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f18_k_marginal.pdf"))
