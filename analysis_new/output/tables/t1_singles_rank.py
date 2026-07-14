# T1: Rank-Annotated Multi-Metric Table for Single Models (RQ1 primary).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex, fmt_cell

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_singles_best, sk_df, borda_per_metric, borda_global,
             latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t1")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    _one_variant(df_singles_best, sk_df, borda_global,
                 models, "all", "mean", out_dir)

def _filter_scope(df, scope):
    if scope == "s1":
        mins = df.groupby("dataset")["sample_size"].min().reset_index()
        mins.columns = ["dataset", "_min"]
        df = df.merge(mins, on="dataset")
        df = df[df["sample_size"] == df["_min"]].drop(columns="_min")
    return df

def _one_variant(df_all, sk_all, borda_global, models, scope, agg, out_dir):
    df   = _filter_scope(df_all, scope)
    sk   = _filter_scope(sk_all, scope)

    rows = []
    for model in models:
        row = {"Model": model}
        for metric in METRICS_EVAL:
            vals = df[(df["model_type"] == model) & (df["metric"] == metric)]["value"].values
            sk_v = sk[(sk["model_type"] == model) & (sk["metric"] == metric)]["sk_rank"].values
            if len(vals) == 0:
                row[metric] = ("--", np.nan, "--", np.nan)
                continue
            if agg == "mean":
                c, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            else:
                c = float(np.median(vals))
                s = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            mean_sk = float(np.mean(sk_v)) if len(sk_v) > 0 else np.nan
            rank_disp = f"{mean_sk:.1f}" if not np.isnan(mean_sk) else "--"
            row[metric] = (c, s, rank_disp, mean_sk)

        sa_vals = df[(df["model_type"] == model) & (df["metric"] == "SA")]["value"].values
        d_vals  = df[(df["model_type"] == model) & (df["metric"] == "D")]["value"].values
        sa_mean = float(np.mean(sa_vals)) if len(sa_vals) > 0 else np.nan
        d_mean  = float(np.mean(d_vals))  if len(d_vals)  > 0 else np.nan
        row["SA"] = (sa_mean, d_mean)

        bg_row = borda_global[borda_global["model_type"] == model]
        row["borda_rank"] = int(bg_row["borda_rank"].values[0]) if len(bg_row) else "--"
        rows.append(row)

    best = {}
    for metric in METRICS_EVAL:
        vals = [r[metric][0] for r in rows
                if r[metric][0] != "--" and not np.isnan(float(r[metric][0]))]
        best[metric] = min(vals) if vals else np.nan
    best_sa = max((r["SA"][0] for r in rows if not np.isnan(r["SA"][0])), default=np.nan)
    best_borda = min((r["borda_rank"] for r in rows if isinstance(r["borda_rank"], int)), default=None)

    col_spec = "l" + "c" * (len(METRICS_EVAL) + 1) + "c"
    header = "Model & " + " & ".join(METRICS_EVAL) + r" & SA (mean D) & Borda \\"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule", header, r"\midrule",
    ]
    for row in rows:
        cells = [row["Model"]]
        for metric in METRICS_EVAL:
            c, s, rank_disp, mean_sk_num = row[metric]
            if c == "--":
                cells.append("--"); continue
            cell = fmt_cell(c, s, sk_rank=rank_disp)
            if not np.isnan(c) and abs(c - best[metric]) < 1e-9:
                cell = bold(cell)
            cells.append(cell)
        sa_c, d_c = row["SA"]
        sa_cell = f"{sa_c:.3f} ({d_c:.2f})" if not np.isnan(sa_c) else "--"
        if not np.isnan(sa_c) and abs(sa_c - best_sa) < 1e-9:
            sa_cell = bold(sa_cell)
        cells.append(sa_cell)
        br = row["borda_rank"]
        br_cell = str(br) if br != best_borda else bold(str(br))
        cells.append(br_cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t1_{agg}_{scope}.tex"
    save_tex(lines, os.path.join(out_dir, fname))
