# F4: Win/Tie/Loss Stacked Bars (RQ2 and RQ3.3).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

WIN_COLOR  = "#2166ac"
TIE_COLOR  = "#f7f7f7"
LOSS_COLOR = "#d73027"

def generate_f4a(wtl_df, figures_dir, model_order=None, metric="MRE"):
    out_dir = os.path.join(figures_dir, "f4")
    models  = model_order or sorted(wtl_df["base_type"].unique())
    sub     = wtl_df[wtl_df["metric"] == metric]

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    _draw_bars(ax, sub, models, "base_type")
    ax.set_title(f"Ensemble vs. single — {metric} (win/tie/loss across 40 scenarios)")
    ax.set_xlabel("Proportion of scenarios")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f4a_wtl_{metric.lower()}.pdf"))

def generate_f4b(df_ens_rq33, figures_dir, model_order=None, metric="MRE"):
    out_dir = os.path.join(figures_dir, "f4")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())
    pairs   = [("MEAN", "NN"), ("IRWM", "NN"), ("MEAN", "IRWM")]
    sub_m   = df_ens_rq33[df_ens_rq33["metric"] == metric]

    rows = []
    for model in models:
        for (r1, r2) in pairs:
            s1 = sub_m[(sub_m["base_type"] == model) & (sub_m["rule"] == r1)]
            s2 = sub_m[(sub_m["base_type"] == model) & (sub_m["rule"] == r2)]
            idx = ["dataset", "sample_size", "run"]
            m  = s1.merge(s2, on=idx, suffixes=("_1", "_2"))
            if m.empty:
                rows.append({"label": f"{model}\n{r1} vs {r2}", "W": 0, "T": 0, "L": 0})
                continue
            w = int((m["value_1"] < m["value_2"]).sum())
            l_ = int((m["value_1"] > m["value_2"]).sum())
            t = len(m) - w - l_
            rows.append({"label": f"{model}\n{r1} vs {r2}", "W": w, "T": t, "L": l_})

    wtl = pd.DataFrame(rows)
    n_total = wtl[["W", "T", "L"]].sum(axis=1)
    n_total = n_total.replace(0, 1)

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.35)))
    lefts = np.zeros(len(rows))
    for val_col, color, label in [("W", WIN_COLOR, "Win"), ("T", TIE_COLOR, "Tie"), ("L", LOSS_COLOR, "Loss")]:
        vals = (wtl[val_col] / n_total * 100).values
        ax.barh(range(len(rows)), vals, left=lefts, color=color, label=label, edgecolor="white")
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 5:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6)
        lefts += vals

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(wtl["label"].tolist(), fontsize=7)
    ax.set_xlabel("% scenarios")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(f"Rule pairwise W/T/L — {metric}")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"f4b_rule_wtl_{metric.lower()}.pdf"))

def _draw_bars(ax, sub, models, group_col):
    w_vals = []; t_vals = []; l_vals = []
    for model in models:
        row = sub[sub[group_col] == model]
        if row.empty:
            w_vals.append(0); t_vals.append(0); l_vals.append(0); continue
        r = row.iloc[0]
        n = max(r["N"], 1)
        w_vals.append(r["W"] / n * 100)
        t_vals.append(r["T"] / n * 100)
        l_vals.append(r["L"] / n * 100)

    lefts = np.zeros(len(models))
    for vals, color, label in zip([w_vals, t_vals, l_vals],
                                   [WIN_COLOR, TIE_COLOR, LOSS_COLOR],
                                   ["Win", "Tie", "Loss"]):
        ax.barh(range(len(models)), vals, left=lefts, color=color, label=label, edgecolor="white")
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 5:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7)
        lefts += np.array(vals)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("% of scenarios")
    ax.legend(loc="lower right", fontsize=7)
