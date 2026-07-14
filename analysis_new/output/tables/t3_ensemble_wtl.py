# T3: Best-Ensemble vs. Single Comparison (RQ2 primary).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_DISPLAY = ["MRE", "MAE", "MBRE", "MIBRE", "SA"]

def generate(wtl_df, latex_dir, model_order=None, agg_label="mean"):
    out_dir = os.path.join(latex_dir, "t3")
    models  = model_order or sorted(wtl_df["base_type"].unique())
    _one_variant(wtl_df, models, agg_label, out_dir)

def _one_variant(df, models, agg_label, out_dir):
    metric_headers = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{m}}}" for m in METRICS_DISPLAY
    )
    sub_header = (" & ".join([r"W & T & L & imp\%"] * len(METRICS_DISPLAY)))
    col_spec = "l" + "rrrr" * len(METRICS_DISPLAY)

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Model & " + metric_headers + r" \\",
        "& " + sub_header + r" \\",
        r"\midrule",
    ]

    for model in models:
        cells = [model]
        for metric in METRICS_DISPLAY:
            row = df[(df["base_type"] == model) & (df["metric"] == metric)]
            if row.empty:
                cells.extend(["--"] * 4)
                continue
            r = row.iloc[0]
            w, t, l_ = int(r["W"]), int(r["T"]), int(r["L"])
            imp = float(r["imp_pct_mean"])
            imp_str = f"{imp:+.1f}\\%"
            cells += [str(w), str(t), str(l_), imp_str]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    fname = f"t3_wtl_{agg_label}.tex"
    save_tex(lines, os.path.join(out_dir, fname))

    _one_variant_winrate(df, models, agg_label, out_dir)

def _one_variant_winrate(df, models, agg_label, out_dir):
    col_spec = "l" + "c" * len(METRICS_DISPLAY)
    metric_headers = " & ".join(METRICS_DISPLAY)

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Model & " + metric_headers + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for metric in METRICS_DISPLAY:
            row = df[(df["base_type"] == model) & (df["metric"] == metric)]
            if row.empty:
                cells.append("--")
                continue
            r = row.iloc[0]
            wr  = float(r["win_rate"]) * 100
            lo  = float(r["win_rate_lo"]) * 100
            hi  = float(r["win_rate_hi"]) * 100
            imp = float(r["imp_pct_mean"])
            cells.append(
                f"{wr:.0f}\\% [{lo:.0f}--{hi:.0f}\\%] ({imp:+.1f}\\%)"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    fname = f"t3_winrate_{agg_label}.tex"
    save_tex(lines, os.path.join(out_dir, fname))

def generate_sk_diff(sk_mixed, latex_dir, model_order=None):
    import numpy as np

    METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]
    out_dir = os.path.join(latex_dir, "t3")
    base_types = model_order or sorted(
        sk_mixed[sk_mixed["kind"] == "single"]["base_type"].unique()
    )

    rows = []
    for bt in base_types:
        row = {"Model": bt}
        for metric in METRICS_EVAL:
            sub = sk_mixed[sk_mixed["metric"] == metric]
            s_ranks = sub[(sub["base_type"] == bt) & (sub["kind"] == "single")]["sk_rank"]
            e_ranks = sub[(sub["base_type"] == bt) & (sub["kind"] == "ensemble")]["sk_rank"]
            if s_ranks.empty or e_ranks.empty:
                row[metric] = np.nan
            else:
                row[metric] = float(e_ranks.mean()) - float(s_ranks.mean())
        rows.append(row)

    best = {}
    for metric in METRICS_EVAL:
        vals = [r[metric] for r in rows if not np.isnan(r[metric])]
        best[metric] = min(vals) if vals else np.nan

    col_spec = "l" + "c" * len(METRICS_EVAL)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Model & " + " & ".join(METRICS_EVAL) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["Model"]]
        for metric in METRICS_EVAL:
            v = row[metric]
            if np.isnan(v):
                cells.append("--")
            else:
                cell = f"{v:+.2f}"
                if not np.isnan(best[metric]) and abs(v - best[metric]) < 1e-9:
                    cell = bold(cell)
                cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(len(METRICS_EVAL)+1) + r"}{l}{\footnotesize $\Delta$ SK rank = mean SK rank (ensemble) $-$ mean SK rank (single) in mixed ranking. Negative = ensemble improved.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t3_sk_rank_diff.tex"))
