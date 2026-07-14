# F10: Performance by Sample Size (RQ1 / RQ2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def _add_sample_rank(df, model_col):
    df = df.copy()
    sizes = (
        df[["dataset", "sample_size"]].drop_duplicates()
        .sort_values(["dataset", "sample_size"])
    )
    sizes["sample_rank"] = sizes.groupby("dataset").cumcount() + 1
    return df.merge(sizes, on=["dataset", "sample_size"])

def _make_figure(df_singles_best, df_ens_best_rq2, models, agg_fn, agg_label):
    sub_s = df_singles_best[df_singles_best["metric"] == "MRE"]
    sub_e = df_ens_best_rq2[df_ens_best_rq2["metric"] == "MRE"]

    sub_s = _add_sample_rank(sub_s, "model_type")
    sub_e = _add_sample_rank(sub_e, "base_type")

    ranks = sorted(sub_s["sample_rank"].unique())
    labels = [f"S{r}" for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)

    for panel_idx, (sub, model_col, title) in enumerate([
        (sub_s, "model_type", "Single models"),
        (sub_e, "base_type",  "Best ensembles"),
    ]):
        ax = axes[panel_idx]
        for model in models:
            ys = []
            for r in ranks:
                vals = sub[(sub[model_col] == model) & (sub["sample_rank"] == r)]["value"].values
                ys.append(float(agg_fn(vals)) if len(vals) > 0 else np.nan)
            ax.plot(range(len(ranks)), ys, marker="o", markersize=4, linewidth=1.4,
                    color=MODEL_COLORS.get(model, "#333"), label=model)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Sample size tier")
        ax.set_ylabel(f"{agg_label.capitalize()} MRE (across all 8 datasets)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", bbox_to_anchor=(1.02, 1),
               fontsize=7, frameon=True, title="Model type")
    fig.suptitle(f"MRE vs. training set size tier ({agg_label}) — singles and best ensembles")
    fig.tight_layout()
    return fig

def generate(df_singles_best, df_ens_best_rq2, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f10")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    for agg_label, agg_fn in [("median", np.median), ("mean", np.mean)]:
        fig = _make_figure(df_singles_best, df_ens_best_rq2, models, agg_fn, agg_label)
        save_figure(fig, os.path.join(out_dir, f"f10_samplesize_trend_mre_{agg_label}.pdf"))
        print(f"  wrote f10_samplesize_trend_mre_{agg_label}.pdf")
