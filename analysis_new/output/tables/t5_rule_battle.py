# T5: Combination Rule Battle Royale (RQ3.3).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex, fmt_cell

RULES        = ["MEAN", "IRWM", "NN"]
METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

def generate(df_ens_rq33, sk_rq33, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t5")
    models  = model_order or sorted(df_ens_rq33["base_type"].unique())

    for agg in ("mean", "median"):
        _option_a(df_ens_rq33, sk_rq33, models, agg, out_dir)
    _option_c(df_ens_rq33, models, out_dir)

def _option_a(df, sk, models, agg, out_dir):
    has_rule = "rule" in sk.columns

    rule_headers = " & ".join(
        rf"\multicolumn{{{len(METRICS_EVAL)}}}{{c}}{{{r}}}" for r in RULES
    )
    sub_header = " & ".join(METRICS_EVAL * len(RULES))
    col_spec = "l" + "c" * (len(RULES) * len(METRICS_EVAL))

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Model & {rule_headers}" + r" \\",
        f"& {sub_header}" + r" \\",
        r"\midrule",
    ]

    all_cells = {}
    for rule in RULES:
        for metric in METRICS_EVAL:
            vals_list = []
            for model in models:
                sub = df[(df["base_type"] == model) & (df["rule"] == rule) & (df["metric"] == metric)]["value"].values
                c = float(np.mean(sub)) if agg == "mean" and len(sub) > 0 \
                    else float(np.median(sub)) if len(sub) > 0 else np.nan
                vals_list.append((model, c))
            all_cells[(metric, rule)] = vals_list
    best = {k: min(c for _, c in v if not np.isnan(c)) for k, v in all_cells.items()}

    for model in models:
        cells = [model]
        for rule in RULES:
            for metric in METRICS_EVAL:
                sub = df[(df["base_type"] == model) & (df["rule"] == rule) & (df["metric"] == metric)]["value"].values
                if len(sub) == 0:
                    cells.append("--"); continue
                if agg == "mean":
                    c = float(np.mean(sub)); s = float(np.std(sub, ddof=1))
                else:
                    c = float(np.median(sub))
                    s = float(np.percentile(sub, 75) - np.percentile(sub, 25))

                if has_rule:
                    sk_v = sk[(sk["base_type"] == model) & (sk["rule"] == rule) & (sk["metric"] == metric)]["sk_rank"].values
                else:
                    sk_v = np.array([])
                mean_sk = float(np.mean(sk_v)) if len(sk_v) > 0 else np.nan
                rank_disp = f"{mean_sk:.1f}" if not np.isnan(mean_sk) else "--"
                cell = fmt_cell(c, s, sk_rank=rank_disp)
                if not np.isnan(c) and abs(c - best[(metric, rule)]) < 1e-9:
                    cell = bold(cell)
                cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    save_tex(lines, os.path.join(out_dir, f"t5a_rule_battle_{agg}.tex"))

def _option_c(df, models, out_dir):
    pairs = [("MEAN", "IRWM"), ("MEAN", "NN"), ("IRWM", "NN")]
    metric = "MRE"

    pair_headers = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{a} vs {b}}}" for a, b in pairs
    )
    sub_hdr = " & ".join(["W & T & L"] * len(pairs))
    col_spec = "l" + "rrr" * len(pairs)

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Model & {pair_headers}" + r" \\",
        f"& {sub_hdr}" + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for (r1, r2) in pairs:
            sub1 = df[(df["base_type"] == model) & (df["rule"] == r1) & (df["metric"] == metric)]
            sub2 = df[(df["base_type"] == model) & (df["rule"] == r2) & (df["metric"] == metric)]
            idx_cols = ["dataset", "sample_size", "run"]
            m = sub1.merge(sub2, on=idx_cols, suffixes=("_1", "_2"))
            if m.empty:
                cells.extend(["--", "--", "--"]); continue
            wins  = int((m["value_1"] < m["value_2"]).sum())
            losses = int((m["value_1"] > m["value_2"]).sum())
            ties  = len(m) - wins - losses
            cells += [str(wins), str(ties), str(losses)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    save_tex(lines, os.path.join(out_dir, "t5c_rule_wtl.tex"))
