"""
F_K_ELBOW: MRE vs k per base type (RQ3.2).

8 subplots (2×4), one per base type.
Each subplot: x = k (2..10), 3 lines (MEAN/IRWM/NN).

Variants:
  generate()              — median MRE, all 40 evaluation scenarios
  generate_mean()         — mean MRE, all 40 evaluation scenarios
  generate_s1_mean()      — mean MRE, S1 scenarios only (per-dataset min sample_size = 8 scenarios)
  generate_mean_byrule()  — 3 panels (one per rule), 8 lines (one per base type), mean MRE
  generate_s1_mean_byrule() — same, S1 only
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, MODEL_COLORS, save_figure

RULES = ["MEAN", "IRWM", "NN"]

BASE_MARKERS = {
    "LR":       "o", "SVR":  "s", "RT":       "^",
    "RF":       "D", "KNN":  "v", "KRR":      "P",
    "DeepPerf": "X", "HINNPerf": "*",
}


def _s1_filter(df):
    """Keep only the minimum sample_size per dataset."""
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]


def _draw(agg, base_types, ks, out_dir, fname, ylabel, suptitle):
    n_bt  = len(base_types)
    ncols = 4
    nrows = (n_bt + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.0), squeeze=False)

    for idx, bt in enumerate(base_types):
        ax      = axes[idx // ncols][idx % ncols]
        bt_data = agg[agg["base_type"] == bt]
        for rule in RULES:
            ys = []
            for k in ks:
                row = bt_data[(bt_data["rule"] == rule) & (bt_data["k"] == k)]
                ys.append(float(row["value"].values[0]) if len(row) else np.nan)
            ax.plot(ks, ys,
                    color=RULE_COLORS.get(rule, "#333"),
                    markersize=4, linewidth=1.4, label=rule)
        ax.set_title(bt, fontsize=9)
        ax.set_xlabel("$k$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.grid(True, alpha=0.2)

    for idx in range(n_bt, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, title="Rule")
    fig.suptitle(suptitle)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))


def _draw_byrule(agg, base_types, ks, out_dir, fname, ylabel, suptitle):
    """3 panels (one per rule), 8 lines (one per base type)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), squeeze=False)

    for col, rule in enumerate(RULES):
        ax       = axes[0][col]
        rule_data = agg[agg["rule"] == rule]
        for bt in base_types:
            ys = []
            for k in ks:
                row = rule_data[(rule_data["base_type"] == bt) & (rule_data["k"] == k)]
                ys.append(float(row["value"].values[0]) if len(row) else np.nan)
            ax.plot(ks, ys,
                    color=MODEL_COLORS.get(bt, "#333"),
                    marker=BASE_MARKERS.get(bt, "o"),
                    markersize=4, linewidth=1.4, label=bt)
        ax.set_title(rule, fontsize=10, fontweight="bold")
        ax.set_xlabel("$k$")
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8,
               title="Base type", ncol=2)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))


def generate(df_ens_raw, figures_dir, model_order=None):
    """Original: median MRE, all 40 evaluation scenarios."""
    out_dir    = os.path.join(figures_dir, "f_k_elbow")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    agg        = sub.groupby(["base_type", "rule", "k"])["value"].median().reset_index()
    ks         = sorted(agg["k"].unique())
    _draw(agg, base_types, ks, out_dir,
          fname="f_k_elbow.pdf",
          ylabel="Median MRE",
          suptitle="MRE vs. $k$ per base type (median across all 40 scenarios)")


def generate_mean(df_ens_raw, figures_dir, model_order=None):
    """Mean MRE, all 40 evaluation scenarios."""
    out_dir    = os.path.join(figures_dir, "f_k_elbow")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    agg        = sub.groupby(["base_type", "rule", "k"])["value"].mean().reset_index()
    ks         = sorted(agg["k"].unique())
    _draw(agg, base_types, ks, out_dir,
          fname="f_k_elbow_mean.pdf",
          ylabel="Mean MRE",
          suptitle="MRE vs. $k$ per base type (mean across all 40 scenarios)")


def generate_s1_mean(df_ens_raw, figures_dir, model_order=None):
    """Mean MRE, S1 only (per-dataset minimum sample size = 8 scenarios)."""
    out_dir    = os.path.join(figures_dir, "f_k_elbow")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = _s1_filter(df_ens_raw[df_ens_raw["metric"] == "MRE"])
    agg        = sub.groupby(["base_type", "rule", "k"])["value"].mean().reset_index()
    ks         = sorted(agg["k"].unique())
    _draw(agg, base_types, ks, out_dir,
          fname="f_k_elbow_s1_mean.pdf",
          ylabel="Mean MRE",
          suptitle="MRE vs. $k$ per base type (mean across S1 scenarios, 1 per dataset)")


def generate_mean_byrule(df_ens_raw, figures_dir, model_order=None):
    """3 panels (per rule), 8 lines (per base type), mean MRE — all 40 scenarios."""
    out_dir    = os.path.join(figures_dir, "f_k_elbow")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = df_ens_raw[df_ens_raw["metric"] == "MRE"]
    agg        = sub.groupby(["base_type", "rule", "k"])["value"].mean().reset_index()
    ks         = sorted(agg["k"].unique())
    _draw_byrule(agg, base_types, ks, out_dir,
                 fname="f_k_elbow_mean_byrule.pdf",
                 ylabel="Mean MRE",
                 suptitle="Mean MRE vs. $k$ per rule — 8 base types (all 40 scenarios, RQ3.2)")


def generate_s1_mean_byrule(df_ens_raw, figures_dir, model_order=None):
    """3 panels (per rule), 8 lines (per base type), mean MRE — S1 only."""
    out_dir    = os.path.join(figures_dir, "f_k_elbow")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = _s1_filter(df_ens_raw[df_ens_raw["metric"] == "MRE"])
    agg        = sub.groupby(["base_type", "rule", "k"])["value"].mean().reset_index()
    ks         = sorted(agg["k"].unique())
    _draw_byrule(agg, base_types, ks, out_dir,
                 fname="f_k_elbow_s1_mean_byrule.pdf",
                 ylabel="Mean MRE",
                 suptitle="Mean MRE vs. $k$ per rule — 8 base types (S1 per dataset, RQ3.2)")
