# T_K_VS_BASELINE: % of scenarios where k is statistically better than k=2 (RQ3.2) — Idea 3.

import os
import numpy as np
import pandas as pd

from output.utils import save_tex

RULES = ["MEAN", "IRWM", "NN"]

def _s1_filter(df):
    min_ss = df.groupby("dataset")["sample_size"].transform("min")
    return df[df["sample_size"] == min_ss]

def _compute_pct_better(k_sk_ranks, base_types, ks_compare):
    result = {}
    for (bt, rule, ds, ss), grp in k_sk_ranks.groupby(
            ["base_type", "rule", "dataset", "sample_size"]):
        rank_k2 = grp[grp["k"] == 2]["sk_rank"]
        if rank_k2.empty:
            continue
        r2 = int(rank_k2.values[0])
        for k in ks_compare:
            rank_k = grp[grp["k"] == k]["sk_rank"]
            if rank_k.empty:
                continue
            rk = int(rank_k.values[0])
            key = (bt, rule, k)
            result.setdefault(key, []).append(1 if rk < r2 else 0)

    return {key: float(np.mean(vals)) * 100 for key, vals in result.items()}

def _build_table(pct_dict, base_types, ks_compare, rule, note=""):
    rows = []
    for bt in base_types:
        cells = [bt]
        for k in ks_compare:
            v = pct_dict.get((bt, rule, k), np.nan)
            if np.isnan(v):
                cells.append("--")
            elif v == 0.0:
                cells.append("0")
            else:
                cells.append(f"{v:.0f}")
        rows.append(" & ".join(cells) + r" \\")
    return rows

def generate(k_sk_ranks, latex_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(latex_dir, "t_k_vs_baseline")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    all_ks     = sorted(k_sk_ranks["k"].unique())
    ks_compare = [k for k in all_ks if k > 2]

    pct = _compute_pct_better(k_sk_ranks, base_types, ks_compare)

    k_header = " & ".join([f"$k$={k}" for k in ks_compare])
    n_cols   = 1 + len(ks_compare)

    for rule in RULES:
        lines = [
            r"\begin{tabular}{l" + "c" * len(ks_compare) + "}",
            r"\toprule",
            r"Base type & " + k_header + r" \\",
            r"\midrule",
        ]
        lines += _build_table(pct, base_types, ks_compare, rule)
        lines += [
            r"\bottomrule",
            rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
            r"Values = \% of 40 evaluation scenarios where $k$ is in a statistically "
            r"better SK group than $k$=2. "
            r"0 = $k$ never beats $k$=2 statistically; "
            r"100 = always beats $k$=2.}} \\",
            r"\end{tabular}",
        ]
        fname = f"t_k_vs_baseline_{rule.lower()}{suffix}.tex"
        save_tex(lines, os.path.join(out_dir, fname))

    _generate_merged(pct, base_types, ks_compare, out_dir,
                     fname=f"t_k_vs_baseline_merged{suffix}.tex",
                     n_scenarios=40)

def generate_s1(k_sk_ranks, latex_dir, model_order=None, suffix=""):
    out_dir    = os.path.join(latex_dir, "t_k_vs_baseline")
    base_types = model_order or sorted(k_sk_ranks["base_type"].unique())
    sub_s1     = _s1_filter(k_sk_ranks)
    all_ks     = sorted(sub_s1["k"].unique())
    ks_compare = [k for k in all_ks if k > 2]

    pct = _compute_pct_better(sub_s1, base_types, ks_compare)

    k_header = " & ".join([f"$k$={k}" for k in ks_compare])
    n_cols   = 1 + len(ks_compare)

    for rule in RULES:
        lines = [
            r"\begin{tabular}{l" + "c" * len(ks_compare) + "}",
            r"\toprule",
            r"Base type & " + k_header + r" \\",
            r"\midrule",
        ]
        lines += _build_table(pct, base_types, ks_compare, rule)
        lines += [
            r"\bottomrule",
            rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
            r"Values = \% of S1 scenarios (1 per dataset) where $k$ is in a statistically "
            r"better SK group than $k$=2.}} \\",
            r"\end{tabular}",
        ]
        save_tex(lines, os.path.join(out_dir, f"t_k_vs_baseline_s1_{rule.lower()}{suffix}.tex"))

    _generate_merged(pct, base_types, ks_compare, out_dir,
                     fname=f"t_k_vs_baseline_s1_merged{suffix}.tex",
                     n_scenarios=8)

def _generate_merged(pct, base_types, ks_compare, out_dir, fname, n_scenarios):
    k_head = " & ".join([f"$k$={k}" for k in ks_compare])
    rule_span = len(ks_compare)
    n_cols = 1 + rule_span * len(RULES)

    rule_headers = " & ".join(
        [rf"\multicolumn{{{rule_span}}}{{c}}{{\textbf{{{r}}}}}" for r in RULES]
    )
    col_spec = "l" + ("c" * rule_span + "|") * (len(RULES) - 1) + "c" * rule_span

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Base type & " + rule_headers + r" \\",
        r"& " + " & ".join([k_head] * len(RULES)) + r" \\",
        r"\midrule",
    ]

    for bt in base_types:
        cells = [bt]
        for rule in RULES:
            for k in ks_compare:
                v = pct.get((bt, rule, k), np.nan)
                cells.append("--" if np.isnan(v) else ("0" if v == 0.0 else f"{v:.0f}"))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
        rf"Values = \% of {n_scenarios} evaluation scenarios where $k$ is in a statistically "
        r"better SK group than $k$=2 (lower SK rank = better). "
        r"0 = never beats $k$=2; 100 = always beats $k$=2.}} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, fname))
