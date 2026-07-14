# F21: Rule Performance by Sample Size tier (RQ3.3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, save_figure
from aggregators.comparisons import add_ensemble_sa_d

RULES = ["MEAN", "IRWM", "NN"]

def _add_sample_rank(df):
    df = df.copy()
    sizes = (
        df[["dataset", "sample_size"]].drop_duplicates()
        .sort_values(["dataset", "sample_size"])
    )
    sizes["sample_rank"] = sizes.groupby("dataset").cumcount() + 1
    return df.merge(sizes, on=["dataset", "sample_size"])

def _make_figure(df_ens_rq33, df_baseline, agg_fn, agg_label):
    ens_aug = add_ensemble_sa_d(df_ens_rq33, df_baseline)

    df_mre = _add_sample_rank(df_ens_rq33[df_ens_rq33["metric"] == "MRE"])
    df_sa  = _add_sample_rank(ens_aug[ens_aug["metric"] == "SA"])

    ranks  = sorted(df_mre["sample_rank"].unique())
    labels = [f"S{r}" for r in ranks]

    fig, (ax_mre, ax_sa) = plt.subplots(1, 2, figsize=(10, 3.8))

    for rule in RULES:
        mre_ys, sa_ys = [], []
        for r in ranks:
            mre_v = df_mre[(df_mre["rule"] == rule) & (df_mre["sample_rank"] == r)]["value"].values
            sa_v  = df_sa[ (df_sa["rule"]  == rule) & (df_sa["sample_rank"]  == r)]["value"].values
            mre_ys.append(float(agg_fn(mre_v)) if len(mre_v) > 0 else np.nan)
            sa_ys.append( float(agg_fn(sa_v))  if len(sa_v)  > 0 else np.nan)

        kw = dict(marker=RULE_MARKERS.get(rule, "o"), markersize=5, linewidth=1.5,
                  color=RULE_COLORS.get(rule, "#333"), label=rule)
        ax_mre.plot(range(len(ranks)), mre_ys, **kw)
        ax_sa.plot( range(len(ranks)), sa_ys,  **kw)

    for ax, ylabel, title, add_sa0 in [
        (ax_mre, f"{agg_label.capitalize()} MRE", "MRE by sample size tier", False),
        (ax_sa,  f"{agg_label.capitalize()} SA",  "SA by sample size tier",  True),
    ]:
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Sample size tier")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if add_sa0:
            ax.axhline(0, color="red", linewidth=0.9, linestyle="-", alpha=0.7)

    fig.suptitle(
        f"Rule comparison by sample size tier ({agg_label}) — RQ3.3\n"
        "Key question: does NN close the gap at larger N?"
    )
    fig.tight_layout()
    return fig

def generate_per_base(df_ens_rq33, figures_dir, model_order=None):
    from .plot_utils import RULE_COLORS, RULE_MARKERS

    out_dir    = os.path.join(figures_dir, "f21")
    base_types = model_order or sorted(df_ens_rq33["base_type"].unique())

    df_mre = _add_sample_rank(df_ens_rq33[df_ens_rq33["metric"] == "MRE"])
    ranks  = sorted(df_mre["sample_rank"].unique())
    labels = [f"S{r}" for r in ranks]

    ncols, nrows = 4, (len(base_types) + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0), squeeze=False)

    for idx, bt in enumerate(base_types):
        ax = axes[idx // ncols][idx % ncols]
        sub_bt = df_mre[df_mre["base_type"] == bt]
        for rule in RULES:
            ys = []
            for r in ranks:
                vals = sub_bt[(sub_bt["rule"] == rule) & (sub_bt["sample_rank"] == r)]["value"].values
                ys.append(float(np.median(vals)) if len(vals) > 0 else np.nan)
            ax.plot(range(len(ranks)), ys,
                    color=RULE_COLORS.get(rule, "#333"),
                    marker=RULE_MARKERS.get(rule, "o"),
                    markersize=4, linewidth=1.4, label=rule)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(bt, fontsize=9)
        ax.set_xlabel("Sample tier")
        ax.set_ylabel("Median MRE")
        ax.grid(True, alpha=0.2)

    for idx in range(len(base_types), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower right", fontsize=9, title="Rule")
    fig.suptitle("MRE vs. sample size tier per base type — rule comparison (RQ3.3)")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f21_per_base_mre.pdf"))

def generate(df_ens_rq33, df_baseline, figures_dir, sel_agg="median"):
    out_dir = os.path.join(figures_dir, "f21")

    for agg_label, agg_fn in [("median", np.median), ("mean", np.mean)]:
        fig = _make_figure(df_ens_rq33, df_baseline, agg_fn, agg_label)
        save_figure(fig, os.path.join(out_dir, f"f21_rule_samplesize_{agg_label}.pdf"))
        print(f"  wrote f21_rule_samplesize_{agg_label}.pdf")
