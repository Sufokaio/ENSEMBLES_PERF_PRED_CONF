"""
F8: Critical Difference (CD) Diagram — 4-panel (RQ1).

One CD diagram per metric (MRE, MAE, MBRE, MIBRE): 2×2 grid.
Uses Nemenyi post-hoc test: CD = q_alpha * sqrt(k*(k+1) / (6*N)).
Average ranks computed over all (dataset, sample_size) scenarios.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

# Demsar 2006 Table 5, alpha=0.05 two-tailed
NEMENYI_Q = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
             6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def generate(df_singles_best, figures_dir, model_order=None, alpha=0.05):
    """
    Parameters
    ----------
    df_singles_best : best-variant singles long-format
    """
    out_dir = os.path.join(figures_dir, "f8")
    models  = model_order or sorted(df_singles_best["model_type"].unique())
    k = len(models)

    # Number of evaluation scenarios = unique (dataset, sample_size)
    scenarios = df_singles_best[["dataset", "sample_size"]].drop_duplicates()
    N = len(scenarios)

    q_alpha = NEMENYI_Q.get(k, 3.031)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for panel_idx, metric in enumerate(METRICS_EVAL):
        ax = axes[panel_idx]
        sub = df_singles_best[df_singles_best["metric"] == metric]

        # Average rank per model across all N scenarios
        # For each scenario, rank models by median value (lower = better rank for error metrics)
        scenario_ranks = []
        for (ds, ss), grp in sub.groupby(["dataset", "sample_size"]):
            med = grp.groupby("model_type")["value"].median()
            ranked = med.rank(method="average")  # lower value → lower rank (rank 1 = best)
            scenario_ranks.append(ranked)

        avg_ranks = (
            pd.concat(scenario_ranks, axis=1)
            .mean(axis=1)
            .reindex(models)
            .sort_values()
        )

        _draw_cd_diagram(ax, avg_ranks, cd, metric, k, N)

    fig.suptitle(f"Critical Difference diagrams (Nemenyi, α={alpha}, CD={cd:.2f})")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f8_cd_diagram_4panel.pdf"))


def _draw_cd_diagram(ax, avg_ranks, cd, title, k, N):
    """Draw a single CD diagram on the given axes."""
    sorted_models = avg_ranks.index.tolist()
    sorted_ranks  = avg_ranks.values.tolist()

    y_center = 0.5
    ax.set_xlim(sorted_ranks[0] - 0.5, sorted_ranks[-1] + cd + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)

    # Draw axis line
    ax.annotate("", xy=(sorted_ranks[-1] + cd + 0.3, y_center),
                xytext=(sorted_ranks[0] - 0.3, y_center),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.0))

    # Place models
    n = len(sorted_models)
    left_models  = sorted_models[:n // 2]
    right_models = sorted_models[n // 2:][::-1]

    # Left side: names above the axis
    for step, model in enumerate(left_models):
        r = avg_ranks[model]
        y_label = y_center + 0.08 + step * 0.09
        ax.plot([r, r], [y_center, y_label], color="black", lw=0.8)
        ax.plot(r, y_center, "o", color="black", ms=4)
        ax.text(r, y_label + 0.02, f"{model}\n{r:.2f}", ha="center", va="bottom", fontsize=6.5)

    # Right side: names below
    for step, model in enumerate(right_models):
        r = avg_ranks[model]
        y_label = y_center - 0.08 - step * 0.09
        ax.plot([r, r], [y_center, y_label], color="black", lw=0.8)
        ax.plot(r, y_center, "o", color="black", ms=4)
        ax.text(r, y_label - 0.02, f"{model}\n{r:.2f}", ha="center", va="top", fontsize=6.5)

    # Draw CD brackets between models within CD of each other
    ranks_arr = np.array(sorted_ranks)
    for i in range(n):
        for j in range(i + 1, n):
            if ranks_arr[j] - ranks_arr[i] <= cd:
                ax.plot([ranks_arr[i], ranks_arr[j]],
                        [y_center - 0.035, y_center - 0.035],
                        color="steelblue", lw=3.5, alpha=0.7, solid_capstyle="round")

    # CD bar in upper-right
    r_ref = sorted_ranks[-1] + 0.1
    ax.plot([r_ref, r_ref + cd], [y_center + 0.3, y_center + 0.3],
            color="black", lw=1.5)
    ax.text(r_ref + cd / 2, y_center + 0.34, f"CD={cd:.2f}", ha="center", fontsize=6.5)
