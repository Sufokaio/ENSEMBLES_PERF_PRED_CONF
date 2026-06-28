"""
F14: Δ Forest Plot — Singles vs. Best Ensembles (RQ2 / C2).

One row per base_type.  Two points per row:
  open circle  = mean Δ for the single (best variant)
  filled circle = mean Δ for the best ensemble
Connected by a thin line.

x-axis: mean Δ (more negative = better than random).
Vertical red line at Δ = 0.

The shift from open → filled circle quantifies how much ensembling
improves the model's standing relative to the random baseline —
which is a stronger claim than just "error went down."
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plot_utils import MODEL_COLORS, save_figure
from aggregators.comparisons import add_ensemble_sa_d


def generate(df_singles_best, df_ens_best_rq2, df_baseline, figures_dir, model_order=None):
    """
    Parameters
    ----------
    df_singles_best : must include metric = "D"
    df_ens_best_rq2 : D will be computed via add_ensemble_sa_d
    df_baseline     : [dataset, sample_size, MAEp0, Sp0]
    """
    out_dir = os.path.join(figures_dir, "f14")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    ens_aug = add_ensemble_sa_d(df_ens_best_rq2, df_baseline)

    def _mean_d_per_scenario(df, model_col, model_name):
        """Mean Δ aggregated per scenario (dataset, sample_size), then mean of that."""
        sub = df[(df[model_col] == model_name) & (df["metric"] == "D")]
        per_sc = sub.groupby(["dataset", "sample_size"])["value"].mean()
        return (float(per_sc.mean()), float(per_sc.std(ddof=1))) if len(per_sc) > 1 \
               else (float(per_sc.mean()), 0.0) if len(per_sc) == 1 else (np.nan, 0.0)

    rows = []
    for model in models:
        ds_m, ds_s = _mean_d_per_scenario(df_singles_best, "model_type", model)
        de_m, de_s = _mean_d_per_scenario(ens_aug,         "base_type",  model)
        rows.append({"model": model, "D_single": ds_m, "D_ens": de_m,
                     "D_single_sd": ds_s, "D_ens_sd": de_s})
    res = pd.DataFrame(rows).sort_values("D_single").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.arange(len(res))

    for i, row in res.iterrows():
        color = MODEL_COLORS.get(row["model"], "#333")
        if not (np.isnan(row["D_single"]) or np.isnan(row["D_ens"])):
            ax.plot([row["D_single"], row["D_ens"]], [i, i],
                    color=color, linewidth=1.1, alpha=0.65, zorder=1)
        if not np.isnan(row["D_single"]):
            ax.errorbar(row["D_single"], i, xerr=row["D_single_sd"],
                        fmt="none", color=color, capsize=2, linewidth=0.7, alpha=0.5, zorder=2)
            ax.scatter(row["D_single"], i, marker="o", s=48,
                       facecolors="none", edgecolors=color, linewidth=1.5, zorder=3)
        if not np.isnan(row["D_ens"]):
            ax.errorbar(row["D_ens"], i, xerr=row["D_ens_sd"],
                        fmt="none", color=color, capsize=2, linewidth=0.7, alpha=0.5, zorder=2)
            ax.scatter(row["D_ens"], i, marker="o", s=48, color=color, zorder=3)

    ax.axvline(0, color="red", linewidth=1.0, linestyle="-")

    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["model"].tolist())
    ax.set_xlabel("Mean Δ effect size (± SD across scenarios)\nMore negative = better than random")
    ax.set_title("Δ forest: single (open) vs. best ensemble (filled)")

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markersize=7, linestyle="none", label="Single (best variant)"),
        Line2D([0], [0], marker="o", color="gray", markersize=7,
               linestyle="none", label="Best ensemble"),
        Line2D([0], [0], color="red", linewidth=1.0, label="Δ = 0 (no better than random)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f14_d_forest_rq2.pdf"))
