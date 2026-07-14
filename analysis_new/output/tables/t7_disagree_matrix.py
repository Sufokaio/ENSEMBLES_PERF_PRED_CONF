# T7: Metric Rank Disagreement Matrix (RQ1 / C2).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t7")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    sub = df_singles_best[df_singles_best["metric"].isin(METRICS_EVAL)]

    med = (
        sub.groupby(["model_type", "dataset", "sample_size", "metric"])["value"]
        .median().reset_index()
    )
    pivot = med.pivot(
        index=["model_type", "dataset", "sample_size"],
        columns="metric", values="value"
    ).reset_index()

    records = []
    for (ds, ss), grp in pivot.groupby(["dataset", "sample_size"]):
        g = grp.copy()
        for metric in METRICS_EVAL:
            if metric in g.columns:
                g[f"rank_{metric}"] = g[metric].rank(method="min")
        records.append(g)
    ranked = pd.concat(records, ignore_index=True)
    n_total = len(ranked)

    mat_count = np.zeros((4, 4), dtype=int)
    for i, ma in enumerate(METRICS_EVAL):
        for j, mb in enumerate(METRICS_EVAL):
            if i == j:
                continue
            ca, cb = f"rank_{ma}", f"rank_{mb}"
            if ca in ranked.columns and cb in ranked.columns:
                mat_count[i, j] = int((ranked[ca] != ranked[cb]).sum())

    mat_pct = mat_count / n_total * 100

    off_diag = mat_count.copy(); np.fill_diagonal(off_diag, 0)
    max_val  = off_diag.max()

    def _build_table(mat, label, fmt):
        col_spec = "l" + "r" * len(METRICS_EVAL)
        header = " & " + " & ".join(METRICS_EVAL) + r" \\"
        lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            r" & \multicolumn{4}{c}{" + label + r"} \\",
            r"\cmidrule{2-5}",
            header,
            r"\midrule",
        ]
        for i, ma in enumerate(METRICS_EVAL):
            cells = [ma]
            for j, mb in enumerate(METRICS_EVAL):
                if i == j:
                    cells.append("0")
                else:
                    cell = fmt(mat[i, j])
                    if mat_count[i, j] == max_val:
                        cell = bold(cell)
                    cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")
        lines += [
            r"\bottomrule",
            r"\multicolumn{5}{l}{\footnotesize N = " + str(n_total) + r" (model $\times$ dataset $\times$ sample\_size) triples per metric pair.}",
            r"\end{tabular}",
        ]
        return lines

    os.makedirs(out_dir, exist_ok=True)
    save_tex(
        _build_table(mat_count, r"\# triples where rank$_A \neq$ rank$_B$", str),
        os.path.join(out_dir, "t7_disagree_count.tex"),
    )
    save_tex(
        _build_table(mat_pct, r"\% triples where rank$_A \neq$ rank$_B$",
                     lambda v: f"{v:.1f}\\%"),
        os.path.join(out_dir, "t7_disagree_pct.tex"),
    )
