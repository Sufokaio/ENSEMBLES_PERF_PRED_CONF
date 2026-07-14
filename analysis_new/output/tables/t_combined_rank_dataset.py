# T_COMBINED_RANK_DATASET: Per-dataset breakdown of the mixed 16-competitor SK ranking (RQ2).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

def generate(sk_mixed, latex_dir, model_order=None, dataset_order=None):
    out_dir = os.path.join(latex_dir, "t_combined_rank_dataset")

    sub = sk_mixed[sk_mixed["metric"] == "MRE"].copy()

    datasets = dataset_order or sorted(sub["dataset"].unique())

    per_ds = (
        sub.groupby(["competitor", "dataset"])["sk_rank"]
        .mean().unstack("dataset")
        .reindex(columns=datasets)
    )
    per_ds["Overall"] = sub.groupby("competitor")["sk_rank"].mean()

    per_ds = per_ds.sort_values("Overall")
    competitors_sorted = per_ds.index.tolist()

    all_cols = datasets + ["Overall"]
    best = {c: per_ds[c].min() for c in all_cols}

    short_ds = [d[:8] for d in datasets]

    col_spec = "l" + "c" * len(all_cols)
    header = "Competitor & " + " & ".join(short_ds + ["Overall"]) + r" \\"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    singles_done = False
    for comp in competitors_sorted:
        kind = sk_mixed[sk_mixed["competitor"] == comp]["kind"].iloc[0]
        if not singles_done and kind == "ensemble":
            lines.append(r"\midrule")
            singles_done = True

        label = comp.replace("_S", r"\_S").replace("_E", r"\_E")
        cells = [label]
        for col in all_cols:
            v = per_ds.at[comp, col]
            if np.isnan(v):
                cells.append("--")
            else:
                cell = f"{v:.1f}"
                if abs(v - best[col]) < 1e-9:
                    cell = bold(cell)
                cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    n_cols = len(all_cols) + 1
    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(n_cols) + r"}{l}{\footnotesize "
        r"Cell = mean SK rank in the 16-competitor mixed ranking (MRE), "
        r"averaged over 5 sample sizes. \_S = single, \_E = best ensemble. "
        r"Bold = best in column.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_combined_rank_dataset.tex"))
