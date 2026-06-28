"""
F11: SA / Δ Scatter — Single Models (RQ1 / C2).

One point per (model_type, dataset, sample_size) triple (40 points per model type).
x = mean SA (Standardized Accuracy, higher = better).
y = mean Δ (effect size vs random, more negative = better).
Color = model type.

Reference lines:
  SA = 0 (vertical red)  → no better than random on SA
  Δ  = 0 (horizontal gray) → no better than random on Δ

All interesting models land in the top-left quadrant (SA > 0, Δ < 0).
Models in the top-right or bottom-left show metric inconsistency.
This figure is direct evidence that SA pulls its weight as a distinct criterion.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure


def generate(df_singles_best, figures_dir, model_order=None):
    """
    Parameters
    ----------
    df_singles_best : best-variant singles long-format (must include metric SA and D)
    """
    out_dir = os.path.join(figures_dir, "f11")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    # Per (model, dataset, sample_size): mean SA and mean D across 30 runs
    sa_agg = (
        df_singles_best[df_singles_best["metric"] == "SA"]
        .groupby(["model_type", "dataset", "sample_size"])["value"]
        .mean().reset_index().rename(columns={"value": "SA"})
    )
    d_agg = (
        df_singles_best[df_singles_best["metric"] == "D"]
        .groupby(["model_type", "dataset", "sample_size"])["value"]
        .mean().reset_index().rename(columns={"value": "D"})
    )
    merged = sa_agg.merge(d_agg, on=["model_type", "dataset", "sample_size"])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    for model in models:
        sub = merged[merged["model_type"] == model]
        ax.scatter(sub["SA"], sub["D"],
                   color=MODEL_COLORS.get(model, "#333"),
                   label=model, s=16, alpha=0.65, edgecolors="none")

    ax.axvline(0, color="red",  linewidth=1.0, linestyle="-",  alpha=0.8, label="SA = 0")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7, label="Δ = 0")

    # Quadrant labels
    ax.text(0.02, 0.98,
            "SA > 0, Δ < 0\n(better than random)",
            transform=ax.transAxes, fontsize=6.5, va="top", color="#2ca02c",
            bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.text(0.98, 0.98,
            "SA < 0, Δ < 0\n(mixed)",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="right", color="#7f7f7f",
            bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.text(0.02, 0.02,
            "SA < 0, Δ > 0\n(worse than random)",
            transform=ax.transAxes, fontsize=6.5, va="bottom", color="#d62728",
            bbox=dict(fc="white", ec="none", alpha=0.7))

    ax.set_xlabel("SA (Standardized Accuracy; higher = better)")
    ax.set_ylabel("Δ effect size (more negative = better than random)")
    ax.set_title("SA vs. Δ — single models\n"
                 "(each point = one [model, dataset, sample_size] triple)")
    ax.legend(fontsize=7, markerscale=1.6, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f11_sa_d_scatter.pdf"))
