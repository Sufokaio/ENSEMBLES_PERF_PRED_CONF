# F_RULE_VIOLIN: MRE distribution per combination rule (RQ3.3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, save_figure

RULES = ["MEAN", "IRWM", "NN"]

def generate(df_ens_rq33, figures_dir):
    out_dir = os.path.join(figures_dir, "f_rule_violin")
    sub = df_ens_rq33[df_ens_rq33["metric"] == "MRE"]
    agg = (
        sub.groupby(["base_type", "dataset", "sample_size", "rule"])["value"]
        .median().reset_index()
    )

    data = [agg[agg["rule"] == r]["value"].values for r in RULES]
    positions = list(range(1, len(RULES) + 1))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    rng = np.random.default_rng(42)

    parts = ax.violinplot(data, positions=positions, showmedians=True,
                          showextrema=True, widths=0.6)

    for i, (body, rule) in enumerate(zip(parts["bodies"], RULES)):
        color = RULE_COLORS.get(rule, "#333")
        body.set_facecolor(matplotlib.colors.to_rgba(color, 0.35))
        body.set_edgecolor(color)
        body.set_linewidth(1.2)
        parts["cmedians"].set_color("black")
        parts["cbars"].set_color([RULE_COLORS.get(r, "#333") for r in RULES])
        parts["cmins"].set_color([RULE_COLORS.get(r, "#333") for r in RULES])
        parts["cmaxes"].set_color([RULE_COLORS.get(r, "#333") for r in RULES])

        jx = positions[i] + rng.uniform(-0.12, 0.12, size=len(data[i]))
        ax.scatter(jx, data[i], color=color, s=8, alpha=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(RULES)
    ax.set_ylabel("Median MRE per scenario")
    ax.set_xlabel("Combination rule")
    ax.set_title("MRE distribution per combination rule\n(across all base type × dataset × sample size scenarios)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_rule_violin.pdf"))
