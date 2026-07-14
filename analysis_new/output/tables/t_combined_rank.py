# T_COMBINED_RANK: SK rank table for all 16 competitors (8 singles + 8 ensembles), RQ2 supplement.

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(sk_mixed, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t_combined_rank")

    competitors = sorted(sk_mixed["competitor"].unique())

    pivot = (
        sk_mixed[sk_mixed["metric"].isin(METRICS_EVAL)]
        .groupby(["competitor", "metric"])["sk_rank"]
        .mean()
        .reset_index()
        .pivot(index="competitor", columns="metric", values="sk_rank")
        .reindex(columns=METRICS_EVAL)
    )

    pivot["mean_all"] = pivot[METRICS_EVAL].mean(axis=1)
    pivot = pivot.sort_values("mean_all")
    competitors_sorted = pivot.index.tolist()

    best_per_metric = {m: pivot[m].min() for m in METRICS_EVAL}
    best_all = pivot["mean_all"].min()

    col_spec = "l" + "c" * (len(METRICS_EVAL) + 1)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Competitor & " + " & ".join(METRICS_EVAL) + r" & Mean \\ ",
        r"\midrule",
    ]

    singles_done = False
    for comp in competitors_sorted:
        kind = sk_mixed[sk_mixed["competitor"] == comp]["kind"].iloc[0]
        if not singles_done and kind == "ensemble":
            lines.append(r"\midrule")
            singles_done = True

        cells = [comp.replace("_S", "\\_S").replace("_E", "\\_E")]
        for metric in METRICS_EVAL:
            v = pivot.at[comp, metric]
            cell = f"{v:.2f}" if not np.isnan(v) else "--"
            if not np.isnan(v) and abs(v - best_per_metric[metric]) < 1e-9:
                cell = bold(cell)
            cells.append(cell)
        mean_v = pivot.at[comp, "mean_all"]
        mean_cell = f"{mean_v:.2f}" if not np.isnan(mean_v) else "--"
        if not np.isnan(mean_v) and abs(mean_v - best_all) < 1e-9:
            mean_cell = bold(mean_cell)
        cells.append(mean_cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(len(METRICS_EVAL)+2) + r"}{l}{\footnotesize \_S = single model, \_E = best ensemble. Sorted by mean SK rank across all metrics.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_combined_rank.tex"))
