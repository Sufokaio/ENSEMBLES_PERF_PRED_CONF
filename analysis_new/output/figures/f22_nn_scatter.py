# F22: NN vs. MEAN Scatter — Direct Evidence for R2-4 (RQ3.3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import MODEL_COLORS, save_figure

def generate(df_ens_rq33, figures_dir, model_order=None):
    out_dir = os.path.join(figures_dir, "f22")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())

    sub = df_ens_rq33[df_ens_rq33["metric"] == "MRE"]
    central = (
        sub.groupby(["base_type", "dataset", "rule"])["value"]
        .median().reset_index()
    )

    for rule_a, rule_b, tag in [
        ("MEAN", "NN",   "mean_vs_nn"),
        ("IRWM", "NN",   "irwm_vs_nn"),
        ("MEAN", "IRWM", "mean_vs_irwm"),
    ]:
        _scatter(central, models, rule_a, rule_b, tag, out_dir)

def _scatter(central, models, rule_a, rule_b, tag, out_dir):
    ra = central[central["rule"] == rule_a].rename(columns={"value": f"MRE_{rule_a}"})
    rb = central[central["rule"] == rule_b].rename(columns={"value": f"MRE_{rule_b}"})
    merge_cols = [c for c in ["base_type", "dataset", "sample_size"] if c in ra.columns and c in rb.columns]
    merged = ra.merge(rb, on=merge_cols)

    n_above = int((merged[f"MRE_{rule_b}"] > merged[f"MRE_{rule_a}"]).sum())
    n_total = len(merged)
    pct_above = n_above / n_total * 100 if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(5.5, 5.2))

    for model in models:
        sub_m = merged[merged["base_type"] == model]
        ax.scatter(sub_m[f"MRE_{rule_a}"], sub_m[f"MRE_{rule_b}"],
                   color=MODEL_COLORS.get(model, "#333"),
                   label=model, s=18, alpha=0.65, edgecolors="none")

    all_vals = pd.concat([merged[f"MRE_{rule_a}"], merged[f"MRE_{rule_b}"]])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = (hi - lo) * 0.03
    lims = [lo - pad, hi + pad]

    ax.plot(lims, lims, color="gray", linewidth=1.0, linestyle="--", label="Parity")
    ax.fill_between(lims, lims, [lims[1], lims[1]], alpha=0.04, color="red")
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.04, color="green")

    ax.text(0.98, 0.98,
            f"{pct_above:.0f}% of scenarios:\n{rule_b} > {rule_a}  ({rule_b} loses)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="darkred",
            bbox=dict(fc="white", ec="lightgray", alpha=0.85, pad=3))
    ax.text(0.02, 0.02,
            f"{100 - pct_above:.0f}% of scenarios:\n{rule_a} ≥ {rule_b}  ({rule_b} wins / ties)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="darkgreen",
            bbox=dict(fc="white", ec="lightgray", alpha=0.85, pad=3))

    ax.set_xlabel(f"Median MRE — {rule_a}")
    ax.set_ylabel(f"Median MRE — {rule_b}")
    ax.set_title(
        f"{rule_b} vs. {rule_a}: per-scenario MRE\n"
        "(above diagonal = {rule_b} has higher error → {rule_b} loses)"
    )
    ax.legend(fontsize=7, markerscale=1.5, loc="upper left")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f22_scatter_{tag}.pdf"))
