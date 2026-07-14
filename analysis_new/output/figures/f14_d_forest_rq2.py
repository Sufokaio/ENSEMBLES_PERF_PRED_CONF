# F14: Δ Forest Plot — Singles vs. Best Ensembles (RQ2 / C2).

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
    out_dir = os.path.join(figures_dir, "f14")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    ens_aug = add_ensemble_sa_d(df_ens_best_rq2, df_baseline)

    def _mean_d_per_scenario(df, model_col, model_name):
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

    OFF = 0.15

    for i, row in res.iterrows():
        color = MODEL_COLORS.get(row["model"], "#333")
        ys = i + OFF
        ye = i - OFF
        if not (np.isnan(row["D_single"]) or np.isnan(row["D_ens"])):
            ax.plot([row["D_single"], row["D_ens"]], [ys, ye],
                    color=color, linewidth=1.1, alpha=0.65, zorder=1)
        if not np.isnan(row["D_single"]):
            ax.errorbar(row["D_single"], ys, xerr=row["D_single_sd"],
                        fmt="none", color=color, capsize=2, linewidth=0.7, alpha=0.5, zorder=2)
            ax.scatter(row["D_single"], ys, marker="o", s=48,
                       facecolors="none", edgecolors=color, linewidth=1.5, zorder=3)
        if not np.isnan(row["D_ens"]):
            ax.errorbar(row["D_ens"], ye, xerr=row["D_ens_sd"],
                        fmt="none", color=color, capsize=2, linewidth=0.7, alpha=0.5, zorder=2)
            ax.scatter(row["D_ens"], ye, marker="o", s=48, color=color, zorder=3)

    ax.axvline(0, color="red", linewidth=1.0, linestyle="-")

    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["model"].tolist())

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markersize=7, linestyle="none", label="Single (best variant)"),
        Line2D([0], [0], marker="o", color="gray", markersize=7,
               linestyle="none", label="Best ensemble"),
        Line2D([0], [0], color="red", linewidth=1.0, label="Δ = 0"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes,
              borderaxespad=0.15)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f14_d_forest_rq2.pdf"))
