# T6: SK Group Evolution Across Metrics (RQ1 / C2).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(sk_singles, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t6")
    models  = model_order or sorted(sk_singles["model_type"].unique())

    mean_sk = (
        sk_singles[sk_singles["metric"].isin(METRICS_EVAL)]
        .groupby(["model_type", "metric"])["sk_rank"]
        .mean()
        .reset_index()
    )
    pivot = (
        mean_sk.pivot(index="model_type", columns="metric", values="sk_rank")
        .reindex(index=models, columns=METRICS_EVAL)
    )

    best = {m: pivot[m].min() for m in METRICS_EVAL}

    col_spec = "l" + "c" * len(METRICS_EVAL)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Mean SK group (lower = better rank cluster)} \\",
        r"\cmidrule{2-5}",
        "Model & " + " & ".join(METRICS_EVAL) + r" \\",
        r"\midrule",
    ]

    n_daggers = 0
    for model in models:
        mre_val = pivot.at[model, "MRE"] if "MRE" in pivot.columns else np.nan
        cells = [model]
        for metric in METRICS_EVAL:
            val = pivot.at[model, metric] if metric in pivot.columns else np.nan
            if np.isnan(val):
                cells.append("--")
                continue
            cell = f"{val:.2f}"
            if not np.isnan(best[metric]) and abs(val - best[metric]) < 1e-9:
                cell = bold(cell)
            if metric != "MRE" and not np.isnan(mre_val) and abs(val - mre_val) > 0.5:
                cell += r"$^\dagger$"
                n_daggers += 1
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize $^\dagger$ mean SK group deviates $>0.5$ from the MRE group; metric changes the grouping (" + str(n_daggers) + r" cases).}",
        r"\end{tabular}",
    ]
    os.makedirs(out_dir, exist_ok=True)
    save_tex(lines, os.path.join(out_dir, "t6_sk_groups.tex"))
