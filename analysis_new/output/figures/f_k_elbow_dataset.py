# F_K_ELBOW_DATASET: MRE vs k (RQ3.2) — by-dataset variants.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, RULE_MARKERS, MODEL_COLORS, save_figure

RULES = ["MEAN", "IRWM", "NN"]

def generate(df_ens_raw, figures_dir, dataset_order=None):
    out_dir  = os.path.join(figures_dir, "f_k_elbow_dataset")
    sub      = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    datasets = dataset_order or sorted(sub["dataset"].unique())
    ks       = sorted(sub["k"].unique())

    ncols = 4
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax     = axes[idx // ncols][idx % ncols]
        sub_ds = sub[sub["dataset"] == ds]

        for rule in RULES:
            ys = []
            for k in ks:
                vals = sub_ds[(sub_ds["rule"] == rule) & (sub_ds["k"] == k)]["value"].values
                ys.append(float(np.median(vals)) if len(vals) > 0 else np.nan)
            ax.plot(ks, ys, marker=RULE_MARKERS.get(rule, "o"), ms=3.5,
                    linewidth=1.3, color=RULE_COLORS.get(rule, "#333"), label=rule)

        ax.set_title(ds, fontsize=8, fontweight="bold")
        ax.set_xlabel("k", fontsize=7)
        ax.set_ylabel("Median MRE", fontsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [plt.Line2D([0], [0], color=RULE_COLORS.get(r, "#333"),
                          marker=RULE_MARKERS.get(r, "o"), lw=1.3, label=r)
               for r in RULES]
    fig.legend(handles=handles, loc="lower right", fontsize=8, title="Rule")
    fig.suptitle("MRE vs k per dataset — aggregated across base types (RQ3.2)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_k_elbow_dataset.pdf"))

def _plot_bybase_panels(sub, datasets, models, ks, out_dir, fname, ylabel, suptitle):
    ncols = 4
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax     = axes[idx // ncols][idx % ncols]
        sub_ds = sub[sub["dataset"] == ds]
        for model in models:
            ys = []
            for k in ks:
                vals = sub_ds[(sub_ds["base_type"] == model) & (sub_ds["k"] == k)]["value"].values
                ys.append(float(np.median(vals)) if len(vals) > 0 else np.nan)
            ax.plot(ks, ys, marker="o", ms=3, linewidth=1.3,
                    color=MODEL_COLORS.get(model, "#333"), label=model)
        ax.set_title(ds, fontsize=8, fontweight="bold")
        ax.set_xlabel("k", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS.get(m, "#333"), marker="o", lw=1.3, label=m)
        for m in models
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=7, title="Base type", ncol=2)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate_by_base(df_ens_raw, figures_dir, dataset_order=None, model_order=None):
    out_dir  = os.path.join(figures_dir, "f_k_elbow_dataset")
    sub      = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    datasets = dataset_order or sorted(sub["dataset"].unique())
    models   = model_order   or sorted(sub["base_type"].unique())
    ks       = sorted(sub["k"].unique())
    _plot_bybase_panels(
        sub, datasets, models, ks, out_dir,
        fname="f_k_elbow_bybase_allds.pdf",
        ylabel="Median MRE (agg. across rules)",
        suptitle="MRE vs k per dataset — 1 line per base type, rules aggregated (RQ3.2)",
    )

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def generate_by_base_s1(df_ens_raw, figures_dir, dataset_order=None, model_order=None):
    out_dir  = os.path.join(figures_dir, "f_k_elbow_dataset")
    sub      = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    sub      = _s1_filter(sub)
    datasets = dataset_order or sorted(sub["dataset"].unique())
    models   = model_order   or sorted(sub["base_type"].unique())
    ks       = sorted(sub["k"].unique())
    _plot_bybase_panels(
        sub, datasets, models, ks, out_dir,
        fname="f_k_elbow_bybase_s1.pdf",
        ylabel="Median MRE (S1 per dataset, rules agg.)",
        suptitle="MRE vs k per dataset (S1 per dataset) — 1 line per base type, rules aggregated (RQ3.2)",
    )

def _plot_bybase_agg(sub, models, ks, out_dir, fname, ylabel, title):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for model in models:
        ys = []
        for k in ks:
            vals = sub[(sub["base_type"] == model) & (sub["k"] == k)]["value"].values
            ys.append(float(np.median(vals)) if len(vals) > 0 else np.nan)
        ax.plot(ks, ys, marker="o", ms=4, linewidth=1.5,
                color=MODEL_COLORS.get(model, "#333"), label=model)
    ax.set_xlabel("k (ensemble size)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate_by_base_agg(df_ens_raw, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f_k_elbow_dataset")
    sub     = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    models  = model_order or sorted(sub["base_type"].unique())
    ks      = sorted(sub["k"].unique())
    _plot_bybase_agg(
        sub, models, ks, out_dir,
        fname="f_k_elbow_bybase_agg.pdf",
        ylabel="Median MRE (agg. across datasets and rules)",
        title="MRE vs k — 1 line per base type,\naggregated across all datasets and rules (RQ3.2)",
    )

def generate_by_base_agg_s1(df_ens_raw, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f_k_elbow_dataset")
    sub     = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    sub     = _s1_filter(sub)
    models  = model_order or sorted(sub["base_type"].unique())
    ks      = sorted(sub["k"].unique())
    _plot_bybase_agg(
        sub, models, ks, out_dir,
        fname="f_k_elbow_bybase_agg_s1.pdf",
        ylabel="Median MRE (S1 per dataset, rules agg.)",
        title="MRE vs k — 1 line per base type,\nS1 per dataset, aggregated across datasets and rules (RQ3.2)",
    )
