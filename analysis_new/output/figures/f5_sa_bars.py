# F5: SA Bar Chart — Singles vs. Ensembles (C2 / C4 / all RQs).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure
from aggregators.comparisons import add_ensemble_sa_d

_SINGLE_ALPHA = 0.55
_ENS_ALPHA    = 1.00
_BAR_WIDTH    = 0.35

def generate(df_singles_best, df_ens_best, df_baseline, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f5")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    ens_aug = add_ensemble_sa_d(df_ens_best, df_baseline)

    sa5_ref = float(df_baseline["SA_5"].mean()) if "SA_5" in df_baseline.columns else None

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    for i, model in enumerate(models):
        s_vals = df_singles_best[
            (df_singles_best["model_type"] == model) & (df_singles_best["metric"] == "SA")
        ]["value"].values
        e_vals = ens_aug[
            (ens_aug["base_type"] == model) & (ens_aug["metric"] == "SA")
        ]["value"].values

        s_mean = float(np.mean(s_vals)) if len(s_vals) > 0 else np.nan
        s_sd   = float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else 0.0
        e_mean = float(np.mean(e_vals)) if len(e_vals) > 0 else np.nan
        e_sd   = float(np.std(e_vals, ddof=1)) if len(e_vals) > 1 else 0.0

        color = MODEL_COLORS.get(model, "#333333")
        ax.bar(x[i] - _BAR_WIDTH / 2, s_mean, _BAR_WIDTH,
               color=color, alpha=_SINGLE_ALPHA, label="Single" if i == 0 else "")
        ax.errorbar(x[i] - _BAR_WIDTH / 2, s_mean, yerr=s_sd,
                    fmt="none", color="black", capsize=2, linewidth=0.8)

        ax.bar(x[i] + _BAR_WIDTH / 2, e_mean, _BAR_WIDTH,
               color=color, alpha=_ENS_ALPHA, label="Ensemble" if i == 0 else "",
               edgecolor="black", linewidth=0.5)
        ax.errorbar(x[i] + _BAR_WIDTH / 2, e_mean, yerr=e_sd,
                    fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.axhline(0, color="red", linewidth=1.0, linestyle="-", label="SA = 0 (random)")
    if sa5_ref is not None:
        ax.axhline(sa5_ref, color="gray", linewidth=0.8, linestyle="--",
                   label=f"Baseline SA₅ = {sa5_ref:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Mean SA (across all 40 scenarios)")
    ax.set_title("Standardized Accuracy: singles vs. best ensembles")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=_SINGLE_ALPHA, label="Single (best variant)"),
        Patch(facecolor="gray", alpha=_ENS_ALPHA, edgecolor="black", label="Best ensemble"),
        plt.Line2D([0], [0], color="red", linewidth=1.0, label="SA = 0"),
    ]
    if sa5_ref is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--",
                       label=f"Baseline SA₅")
        )
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f5_sa_bars.pdf"))
