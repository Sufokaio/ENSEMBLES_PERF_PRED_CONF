"""
T8: Ensemble Size Sensitivity — k=2 vs k=best vs k=10 (RQ3.2).

For each (base_type, rule): median MRE at three ensemble sizes:
  k=2 (minimum), k=best (globally optimal per rule), k=10 (maximum tested).
Plus two delta columns:
  Δ(best−2)  = MRE(best) − MRE(2)    [negative = tuning pays off over k=2]
  Δ(10−best) = MRE(10)   − MRE(best) [positive = using max k over-shoots]

Practical use:
  - Small |Δ(best−2)|  → k=2 is nearly as good; no tuning needed.
  - Negative Δ(10−best) → larger k still helps; recommendation: push to 10.
  - Positive Δ(10−best) → there is an elbow before k=10; tune k.

Bold = k=best cell (the recommended value).
The footnote reports the global median best k per rule.
"""
import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

RULES = ["MEAN", "IRWM", "NN"]


def generate(df_ens_raw, latex_dir, model_order=None, sel_agg="median"):
    """
    Parameters
    ----------
    df_ens_raw  : full ensemble DataFrame (all k, all rules)
    sel_agg     : "median" or "mean"
    """
    out_dir = os.path.join(latex_dir, "t8")
    models  = model_order or sorted(df_ens_raw["base_type"].unique())
    fn      = np.median if sel_agg == "median" else np.mean

    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"]

    # Best k per (base_type, rule) aggregated across ALL datasets + sample_sizes
    agg = (
        sub.groupby(["base_type", "rule", "k"])["value"]
        .agg(fn).reset_index().rename(columns={"value": "agg_val"})
    )
    best_k_idx = agg.groupby(["base_type", "rule"])["agg_val"].idxmin()
    best_k_map = agg.loc[best_k_idx].set_index(["base_type", "rule"])["k"].to_dict()

    # Global median best k per rule (for footnote)
    rule_best_k_median = {}
    for rule in RULES:
        ks = [best_k_map.get((m, rule)) for m in models if best_k_map.get((m, rule)) is not None]
        rule_best_k_median[rule] = int(np.median(ks)) if ks else "?"

    rule_headers = " & ".join(
        rf"\multicolumn{{5}}{{c}}{{{r}}}" for r in RULES
    )
    sub_header = " & ".join(
        [r"$k$=2 & $k^*$ & $k$=10 & $\Delta_{2}$ & $\Delta_{10}$"] * len(RULES)
    )
    col_spec = "l" + "ccccc" * len(RULES)

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Model & {rule_headers}" + r" \\",
        r"\cmidrule{2-6}\cmidrule{7-11}\cmidrule{12-16}",
        f"& {sub_header}" + r" \\",
        r"\midrule",
    ]

    for model in models:
        cells = [model]
        for rule in RULES:
            best_k = best_k_map.get((model, rule))
            vals = {}
            for k_label, k_val in [("k2", 2), ("kbest", best_k), ("k10", 10)]:
                if k_val is None:
                    vals[k_label] = np.nan
                    continue
                v = sub[(sub["base_type"] == model) & (sub["rule"] == rule) &
                        (sub["k"] == k_val)]["value"].values
                vals[k_label] = float(fn(v)) if len(v) > 0 else np.nan

            row_cells = []
            for k_label, k_val in [("k2", 2), ("kbest", best_k), ("k10", 10)]:
                v = vals.get(k_label, np.nan)
                cell = f"{v:.4f}" if not np.isnan(v) else "--"
                if k_label == "kbest":
                    cell = bold(cell)
                    if best_k is not None:
                        cell = bold(f"{v:.4f} $[{best_k}]$") if not np.isnan(v) else "--"
                row_cells.append(cell)

            d1 = vals.get("kbest", np.nan) - vals.get("k2",    np.nan)
            d2 = vals.get("k10",   np.nan) - vals.get("kbest", np.nan)
            row_cells.append(f"{d1:+.4f}" if not np.isnan(d1) else "--")
            row_cells.append(f"{d2:+.4f}" if not np.isnan(d2) else "--")
            cells.extend(row_cells)

        lines.append(" & ".join(cells) + r" \\")

    footnote_parts = "; ".join(
        f"{r}: median $k^*$ = {rule_best_k_median[r]}" for r in RULES
    )
    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(1 + 5 * len(RULES)) + r"}{l}{\footnotesize " +
        r"$k^*$ = globally selected best $k$ [shown in brackets]. " +
        r"$\Delta_2 = $ MRE($k^*$) $-$ MRE(2). $\Delta_{10} = $ MRE(10) $-$ MRE($k^*$). " +
        r"Negative $\Delta$ = smaller MRE. " + footnote_parts + r".}",
        r"\end{tabular}",
    ]
    os.makedirs(out_dir, exist_ok=True)
    save_tex(lines, os.path.join(out_dir, "t8_k_compare.tex"))
