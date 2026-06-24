"""
T4: Base-Learner Type Effect Within Ensembles (RQ3.1).

Two sub-tables:
  T4a — Lift table: Borda rank as single | Borda rank as ensemble base | shift
  T4b — Multi-metric table for ensemble-base performance (like T1 but for ensembles)
"""
import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex, fmt_cell

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]


def generate(df_singles_best, df_ens_best_rq31,
             sk_singles, borda_global_singles,
             sk_ens, borda_global_ens,
             latex_dir, model_order=None):
    """
    Parameters
    ----------
    df_ens_best_rq31 : ensemble df after selecting best (k, rule) per
                       (base_type, dataset, sample_size) — same as RQ2 best.
    sk_singles / borda_global_singles : from singles analysis.
    sk_ens / borda_global_ens         : run SK on ens_best using base_type as group.
    """
    out_dir = os.path.join(latex_dir, "t4")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    _lift_table(borda_global_singles, borda_global_ens, models, out_dir)
    _ens_rank_table(df_ens_best_rq31, sk_ens, borda_global_ens, models, out_dir)


def _lift_table(borda_single, borda_ens, models, out_dir):
    """T4a: rank-as-single | rank-as-ensemble | shift."""
    s_rank = borda_single.set_index("model_type")["borda_rank"].to_dict()
    e_rank = borda_ens.set_index("base_type")["borda_rank"].to_dict() \
             if "base_type" in borda_ens.columns \
             else borda_ens.set_index("model_type")["borda_rank"].to_dict()

    col_spec = "lccc"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Model & Rank (single) & Rank (ensemble base) & Shift \\",
        r"\midrule",
    ]
    for model in models:
        sr = s_rank.get(model, "--")
        er = e_rank.get(model, "--")
        if isinstance(sr, int) and isinstance(er, int):
            shift = sr - er  # positive = improved as ensemble base
            shift_str = f"{shift:+d}"
        else:
            shift_str = "--"
        lines.append(f"{model} & {sr} & {er} & {shift_str} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    save_tex(lines, os.path.join(out_dir, "t4a_lift.tex"))


def _ens_rank_table(df_ens, sk_ens, borda_global_ens, models, out_dir):
    """T4b: multi-metric rank table for ensemble-base performance."""
    group_col = "base_type"
    # Rename base_type → model_type for uniform processing
    df = df_ens.rename(columns={"base_type": "model_type"})
    sk = sk_ens.rename(columns={"base_type": "model_type"}) \
         if "base_type" in sk_ens.columns else sk_ens

    for agg in ("mean", "median"):
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
                    c = float(np.mean(vals)); s = float(np.std(vals, ddof=1))
                else:
                    c = float(np.median(vals))
                    s = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                mean_sk = float(np.mean(sk_v)) if len(sk_v) > 0 else np.nan
                row[metric] = (c, s, f"{mean_sk:.1f}" if not np.isnan(mean_sk) else "--", mean_sk)
            rows.append(row)

        best = {m: min(r[m][0] for r in rows if isinstance(r[m][0], float)) for m in METRICS_EVAL}
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
                c, s, rank_disp, mean_sk_num = row[metric]
                if c == "--":
                    cells.append("--"); continue
                cell = fmt_cell(c, s, sk_rank=rank_disp)
                if not np.isnan(c) and abs(c - best[metric]) < 1e-9:
                    cell = bold(cell)
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save_tex(lines, os.path.join(out_dir, f"t4b_ens_rank_{agg}.tex"))
