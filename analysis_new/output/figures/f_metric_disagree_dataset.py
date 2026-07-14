# F_METRIC_DISAGREE_DATASET: Per-dataset metric disagreement on best model (RQ1 / Contribution 2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

COMPARE = ["MAE", "MBRE", "MIBRE"]
COLORS  = {"MAE": "#2166ac", "MBRE": "#d01c8b", "MIBRE": "#e6813a"}

def generate(df_singles_best, figures_dir, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_metric_disagree_dataset")
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())

    med = (
        df_singles_best
        .groupby(["model_type", "dataset", "sample_size", "metric"])["value"]
        .median().reset_index()
    )

    records = []
    for ds in datasets:
        for ss in med["sample_size"].unique():
            s = med[(med["dataset"] == ds) & (med["sample_size"] == ss)]
            sub_mre = s[s["metric"] == "MRE"]
            if sub_mre.empty:
                continue
            best_mre = sub_mre.set_index("model_type")["value"].idxmin()
            for m in COMPARE:
                sub_m = s[s["metric"] == m]
                if sub_m.empty:
                    continue
                best_m = sub_m.set_index("model_type")["value"].idxmin()
                records.append({
                    "dataset": ds, "sample_size": ss,
                    "metric": m, "disagree": int(best_mre != best_m)
                })

    df_rec = pd.DataFrame(records)
    pct = (
        df_rec.groupby(["dataset", "metric"])["disagree"]
        .mean().mul(100).unstack("metric")
        .reindex(index=datasets, columns=COMPARE, fill_value=0)
    )

    x     = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, m in enumerate(COMPARE):
        ax.bar(x + (i - 1) * width, pct[m], width=width * 0.9,
               color=COLORS[m], label=f"MRE vs {m}", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% sample sizes with different best model")
    ax.set_title("Per-dataset metric disagreement on best single model — RQ1 / multi-metric")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_metric_disagree_dataset.pdf"))
