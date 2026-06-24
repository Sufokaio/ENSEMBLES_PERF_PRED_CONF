"""
T3: Best-Ensemble vs. Single Comparison (RQ2 primary).

8 rows (base model types) × 5 metric groups (MRE, MAE, MBRE, MIBRE, SA).
Per group: W/T/L | win-rate [95% CI] | imp%
Two variants: mean-based and median-based.
"""
import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

METRICS_DISPLAY = ["MRE", "MAE", "MBRE", "MIBRE", "SA"]


def generate(wtl_df, latex_dir, model_order=None):
    """
    Parameters
    ----------
    wtl_df : output of comparisons.compute_wtl()
             [base_type, metric, W, T, L, N, imp_pct_mean, win_rate, win_rate_lo, win_rate_hi]
    """
    out_dir = os.path.join(latex_dir, "t3")
    models  = model_order or sorted(wtl_df["base_type"].unique())

    for agg_label in ("mean", "median"):
        # wtl_df already computed for one agg; caller passes one per variant
        _one_variant(wtl_df, models, agg_label, out_dir)


def _one_variant(df, models, agg_label, out_dir):
    # Metric group header spans 4 cols: W | T | L | imp%
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

    # Also write a compact version with win-rate + CI
    _one_variant_winrate(df, models, agg_label, out_dir)


def _one_variant_winrate(df, models, agg_label, out_dir):
    """Alternative format: win-rate [CI] + imp% (TOSEM style)."""
    metric_headers = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in METRICS_DISPLAY
    )
    sub_header = " & ".join([r"win\% [95\%CI] & imp\%"] * len(METRICS_DISPLAY))
    col_spec   = "l" + "cc" * len(METRICS_DISPLAY)

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
                cells.extend(["--", "--"])
                continue
            r = row.iloc[0]
            wr   = float(r["win_rate"]) * 100
            lo   = float(r["win_rate_lo"]) * 100
            hi   = float(r["win_rate_hi"]) * 100
            imp  = float(r["imp_pct_mean"])
            cells.append(f"{wr:.0f}\\% [{lo:.0f}--{hi:.0f}\\%]")
            cells.append(f"{imp:+.1f}\\%")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    fname = f"t3_winrate_{agg_label}.tex"
    save_tex(lines, os.path.join(out_dir, fname))
