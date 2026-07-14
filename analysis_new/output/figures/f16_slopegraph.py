# F16: Slopegraph — Single → Ensemble Base Borda Rank (RQ3.1).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(borda_global_singles, borda_global_ens, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f16")
    models  = model_order or sorted(borda_global_singles["model_type"].unique())

    s_rank = borda_global_singles.set_index("model_type")["borda_rank"].to_dict()
    ens_col = "base_type" if "base_type" in borda_global_ens.columns else "model_type"
    e_rank  = borda_global_ens.set_index(ens_col)["borda_rank"].to_dict()

    fig, ax = plt.subplots(figsize=(4.8, 5.5))

    ax.set_xlim(-0.6, 1.6)
    n_models = len(models)
    ax.set_ylim(0.3, n_models + 0.7)
    ax.invert_yaxis()

    for model in models:
        sr = s_rank.get(model)
        er = e_rank.get(model)
        if sr is None or er is None:
            continue
        color = MODEL_COLORS.get(model, "#333")
        lw = 2.0 if sr != er else 1.2
        ax.plot([0, 1], [sr, er], color=color, linewidth=lw, alpha=0.85)
        ax.text(-0.07, sr, f"{model}  ({sr})",
                ha="right", va="center", fontsize=8, color=color)
        ax.text(1.07, er, f"({er})",
                ha="left",  va="center", fontsize=8, color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Single model\nBorda rank", "Ensemble base\nBorda rank"], fontsize=8)
    ax.set_yticks([])
    ax.set_title(
        "Rank shift: single model → ensemble base (RQ3.1)\n"
        "1 = best.  Downward slope = improved rank when ensembled."
    )
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f16_slopegraph.pdf"))
