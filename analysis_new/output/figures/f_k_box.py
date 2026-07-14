# F_K_BOX: Distribution of MRE across (base_type × dataset) combos per k (RQ3.2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import RULE_COLORS, save_figure

RULES = ["MEAN", "IRWM", "NN"]

def generate(df_ens_raw, figures_dir):
    out_dir = os.path.join(figures_dir, "f_k_box")
    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"]

    agg = (
        sub.groupby(["base_type", "dataset", "rule", "k"])["value"]
        .median().reset_index()
    )
    ks = sorted(agg["k"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ax, rule in zip(axes, RULES):
        rule_data = agg[agg["rule"] == rule]
        data_per_k = [rule_data[rule_data["k"] == k]["value"].values for k in ks]
        bp = ax.boxplot(data_per_k, positions=ks, widths=0.6,
                        patch_artist=True, medianprops=dict(color="black", lw=2))
        color = RULE_COLORS.get(rule, "#333")
        for patch in bp["boxes"]:
            patch.set_facecolor(matplotlib.colors.to_rgba(color, 0.35))
            patch.set_edgecolor(color)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color(color)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Number of learners $k$")
        ax.set_ylabel("Median MRE")
        ax.set_title(f"Rule: {rule}")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("MRE distribution across (base type × dataset) combos per $k$")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "f_k_box.pdf"))
