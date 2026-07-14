# F_RANK_FLIP_HEATMAP: Metric rank-flip rate per dataset (RQ1 / Contribution 2).

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]
PAIRS   = list(itertools.combinations(METRICS, 2))

def generate(df_singles_best, figures_dir, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_rank_flip_heatmap")
    datasets = dataset_order or sorted(df_singles_best["dataset"].unique())
    models   = sorted(df_singles_best["model_type"].unique())
    mpairs   = list(itertools.combinations(models, 2))

    med = (
        df_singles_best
        .groupby(["model_type", "dataset", "sample_size", "metric"])["value"]
        .median().reset_index()
    )

    records = []
    for ds in datasets:
        sub_ds = med[med["dataset"] == ds]
        for ss in sub_ds["sample_size"].unique():
            pivot = (
                sub_ds[sub_ds["sample_size"] == ss]
                .pivot(index="model_type", columns="metric", values="value")
            )
            if pivot.empty:
                continue
            for m1, m2 in PAIRS:
                if m1 not in pivot.columns or m2 not in pivot.columns:
                    continue
                flips, total = 0, 0
                for ma, mb in mpairs:
                    if ma not in pivot.index or mb not in pivot.index:
                        continue
                    v1a = pivot.at[ma, m1]; v1b = pivot.at[mb, m1]
                    v2a = pivot.at[ma, m2]; v2b = pivot.at[mb, m2]
                    if any(np.isnan(v) for v in [v1a, v1b, v2a, v2b]):
                        continue
                    total += 1
                    if np.sign(v1a - v1b) != 0 and np.sign(v2a - v2b) != 0:
                        if np.sign(v1a - v1b) != np.sign(v2a - v2b):
                            flips += 1
                if total > 0:
                    records.append({
                        "dataset": ds, "sample_size": ss,
                        "pair": f"{m1}-{m2}",
                        "flip_pct": flips / total * 100
                    })

    df_rec = pd.DataFrame(records)
    if df_rec.empty:
        print("  f_rank_flip_heatmap: no data, skipping")
        return

    pair_labels = [f"{m1} vs {m2}" for m1, m2 in PAIRS]
    pct = (
        df_rec.groupby(["pair", "dataset"])["flip_pct"]
        .mean().unstack("dataset")
        .reindex(index=[f"{m1}-{m2}" for m1, m2 in PAIRS], columns=datasets, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(pct.values, cmap="Oranges", vmin=0, vmax=50, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_title("Metric rank-flip rate per dataset — % model pairs that swap relative order (RQ1)")

    for i in range(len(PAIRS)):
        for j in range(len(datasets)):
            v = pct.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 35 else "black")

    fig.colorbar(im, ax=ax, fraction=0.025, label="% model pairs flipped")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_rank_flip_heatmap.pdf"))
