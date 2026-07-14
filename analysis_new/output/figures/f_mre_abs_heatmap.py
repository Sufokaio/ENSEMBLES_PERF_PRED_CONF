# F_MRE_ABS_HEATMAP: Absolute MRE heatmap — models x datasets (RQ1).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure, ds_label

def _sk_pivot(sk_singles, sample_sizes=None):
    sub = sk_singles[sk_singles["metric"] == "MRE"].copy()
    if sample_sizes is not None:
        sub = sub[sub["sample_size"].isin(sample_sizes)]
    return (
        sub.groupby(["model_type", "dataset"])["sk_rank"]
        .mean().unstack("dataset")
    )

def _draw(pivot_mre, models, ds_order, agg_label, scope_label,
          out_dir, fname, sk_piv=None):
    mat     = pivot_mre.reindex(index=models, columns=ds_order).values.astype(float)
    mat_sk  = sk_piv.reindex(index=models, columns=ds_order).values.astype(float) \
              if sk_piv is not None else None
    vmax = float(np.nanpercentile(mat[~np.isnan(mat)], 95)) if not np.all(np.isnan(mat)) else 1.0

    n_ds  = len(ds_order)
    n_mod = len(models)
    fig, ax = plt.subplots(figsize=(max(7.0, n_ds * 0.85), max(2.6, n_mod * 0.42)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels([ds_label(d) for d in ds_order], rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(n_mod))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n_mod):
        for j in range(n_ds):
            v = mat[i, j]
            if np.isnan(v):
                continue
            if mat_sk is not None and not np.isnan(mat_sk[i, j]):
                ax.text(j, i - 0.18, f"{v:.2f}",
                        ha="center", va="center", fontsize=8, color="black")
                ax.text(j, i + 0.22, f"({mat_sk[i, j]:.1f})",
                        ha="center", va="center", fontsize=6.5,
                        color="black", alpha=0.85)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label(f"{agg_label} MRE", fontsize=9)
    fig.tight_layout(pad=0.4)
    save_figure(fig, os.path.join(out_dir, fname))

def generate(df_singles_best, figures_dir, sk_singles=None,
             model_order=None, dataset_order=None):
    out_dir = os.path.join(figures_dir, "f_mre_abs_heatmap")
    models  = model_order or sorted(df_singles_best["model_type"].unique())
    sub     = df_singles_best[df_singles_best["metric"] == "MRE"]

    med_pivot = sub.groupby(["model_type", "dataset"])["value"].median().unstack("dataset")
    ds_order  = dataset_order or med_pivot.min(axis=0).sort_values().index.tolist()

    mins   = sub.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    sub_s1 = sub.merge(mins, on="dataset").query("sample_size == _min").drop(columns="_min")
    s1_sizes = sub_s1["sample_size"].unique() if not sub_s1.empty else None

    sk_all = _sk_pivot(sk_singles) if sk_singles is not None else None
    sk_s1  = _sk_pivot(sk_singles, sample_sizes=s1_sizes) \
             if sk_singles is not None and s1_sizes is not None else None

    for agg_label, agg_fn in [("Mean", "mean")]:
        pivot_all = getattr(
            sub.groupby(["model_type", "dataset"])["value"], agg_fn
        )().unstack("dataset")
        _draw(pivot_all, models, ds_order, agg_label,
              "all 40 scenarios", out_dir,
              f"f_mre_abs_heatmap_{agg_fn}_all.pdf",
              sk_piv=sk_all)

        pivot_s1 = getattr(
            sub_s1.groupby(["model_type", "dataset"])["value"], agg_fn
        )().unstack("dataset")
        _draw(pivot_s1, models, ds_order, agg_label,
              "S1 only (8 scenarios)", out_dir,
              f"f_mre_abs_heatmap_{agg_fn}_s1.pdf",
              sk_piv=sk_s1)
