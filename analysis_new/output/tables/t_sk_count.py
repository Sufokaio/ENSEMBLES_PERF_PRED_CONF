# T_SK_COUNT: Scott-Knott cluster membership counts (RQ1).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(sk_singles, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t_sk_count")
    models = model_order or sorted(sk_singles["model_type"].unique())
    N = sk_singles[["dataset", "sample_size"]].drop_duplicates().shape[0]

    _variant_a(sk_singles, models, N, out_dir)
    _variant_b(sk_singles, models, N, out_dir)
    _variant_c(sk_singles, models, N, out_dir)

def _variant_a(sk, models, N, out_dir):
    lines = [
        r"\begin{tabular}{l" + "c" * len(METRICS_EVAL) + "}",
        r"\toprule",
        "Model & " + " & ".join(METRICS_EVAL) + r" \\",
        r"\midrule",
    ]
    counts = {}
    for metric in METRICS_EVAL:
        sub = sk[(sk["metric"] == metric) & (sk["sk_rank"] == 1)]
        counts[metric] = sub.groupby("model_type").size().reindex(models, fill_value=0)

    best_per_metric = {m: counts[m].max() for m in METRICS_EVAL}
    for model in models:
        cells = [model]
        for metric in METRICS_EVAL:
            v = int(counts[metric][model])
            cell = str(v)
            if v == best_per_metric[metric]:
                cell = bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{" + str(len(METRICS_EVAL)+1) + r"}{l}{\footnotesize $N=" + str(N) + r"$ evaluation scenarios.} \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "tsk_count_best.tex"))

def _variant_b(sk, models, N, out_dir):
    lines = [
        r"\begin{tabular}{l" + "c" * len(METRICS_EVAL) + "}",
        r"\toprule",
        "Model & " + " & ".join(METRICS_EVAL) + r" \\",
        r"\midrule",
    ]
    counts = {}
    for metric in METRICS_EVAL:
        sub = sk[(sk["metric"] == metric) & (sk["sk_rank"] <= 2)]
        counts[metric] = sub.groupby("model_type").size().reindex(models, fill_value=0)

    best_per_metric = {m: counts[m].max() for m in METRICS_EVAL}
    for model in models:
        cells = [model]
        for metric in METRICS_EVAL:
            v = int(counts[metric][model])
            cell = str(v)
            if v == best_per_metric[metric]:
                cell = bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{" + str(len(METRICS_EVAL)+1) + r"}{l}{\footnotesize Top-2 clusters. $N=" + str(N) + r"$ evaluation scenarios.} \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "tsk_count_top2.tex"))

def _variant_c(sk, models, N, out_dir):
    max_rank = int(sk["sk_rank"].max())
    rank_labels = [f"R{r}" for r in range(1, max_rank + 1)]

    for metric in METRICS_EVAL:
        sub = sk[sk["metric"] == metric]
        col_spec = "l" + "r" * max_rank
        lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            "Model & " + " & ".join(rank_labels) + r" \\",
            r"\midrule",
        ]
        dist = (
            sub.groupby(["model_type", "sk_rank"])
            .size().reset_index(name="count")
            .pivot(index="model_type", columns="sk_rank", values="count")
            .reindex(index=models, columns=range(1, max_rank+1))
            .fillna(0).astype(int)
        )
        for model in models:
            cells = [model]
            for r in range(1, max_rank + 1):
                v = int(dist.at[model, r]) if r in dist.columns else 0
                cells.append(str(v))
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save_tex(lines, os.path.join(out_dir, f"tsk_count_dist_{metric.lower()}.tex"))
