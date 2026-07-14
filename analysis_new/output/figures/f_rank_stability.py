# F_RANK_STABILITY: Model rank stability across datasets (RQ1).

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(borda_ds_singles, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f_rank_stability")
    models  = model_order or sorted(borda_ds_singles["model_type"].unique())

    data = [
        borda_ds_singles[borda_ds_singles["model_type"] == m]["borda_rank"].values
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bp = ax.boxplot(data, labels=models, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(m, "#cccccc"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Borda rank across datasets (1 = best)")
    ax.set_title("Per-model rank stability across datasets — RQ1")
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_rank_stability_box.pdf"))
