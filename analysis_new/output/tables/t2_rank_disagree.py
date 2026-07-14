# T2: Metric Rank-Disagreement Evidence (RQ1 / C2).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(borda_per_metric, borda_global, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t2")
    models = model_order or list(borda_global.sort_values("borda_rank")["model_type"])

    pivot = (
        borda_per_metric[borda_per_metric["metric"].isin(METRICS_EVAL)]
        .pivot(index="model_type", columns="metric", values="borda_total")
        .reindex(index=models, columns=METRICS_EVAL)
    )

    mre_order = (
        borda_per_metric[borda_per_metric["metric"] == "MRE"]
        .sort_values("borda_total")["model_type"]
        .reset_index(drop=True)
        .to_list()
    )
    mre_rank = {m: i + 1 for i, m in enumerate(mre_order)}

    best = {m: pivot[m].min() for m in METRICS_EVAL}
    global_rank = borda_global.set_index("model_type")["borda_rank"].to_dict()

    col_spec = "l" + "c" * (len(METRICS_EVAL) + 1)
    header = "Model & " + " & ".join(METRICS_EVAL) + r" & Global \\"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule", header, r"\midrule",
    ]
    for model in models:
        cells = [model]
        for metric in METRICS_EVAL:
            val = pivot.at[model, metric] if metric in pivot.columns else np.nan
            cell = str(int(val)) if not np.isnan(val) else "--"
            if not np.isnan(val) and val == best[metric]:
                cell = bold(cell)
            if metric == "MRE" and mre_rank.get(model) != global_rank.get(model):
                cell += r"$^\dagger$"
            cells.append(cell)
        g_val = borda_global.loc[borda_global["model_type"] == model, "borda_total_all"]
        g_str = str(int(g_val.values[0])) if len(g_val) else "--"
        cells.append(g_str)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    save_tex(lines, os.path.join(out_dir, "t2_rank_disagree.tex"))
