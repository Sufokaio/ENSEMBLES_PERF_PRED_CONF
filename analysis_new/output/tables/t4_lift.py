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
             latex_dir, model_order=None,
             df_baseline=None, sk_mixed=None):
    """
    Parameters
    ----------
    df_ens_best_rq31 : ensemble df after selecting best (k, rule) per
                       (base_type, dataset, sample_size) — same as RQ2 best.
    sk_singles / borda_global_singles : from singles analysis.
    sk_ens / borda_global_ens         : run SK on ens_best using base_type as group.
    df_baseline  : optional — if provided, SA and D columns are added to T4b.
    sk_mixed     : optional — if provided, Borda rank in the 16-competitor mixed
                   ranking is added as a column to T4b.
    """
    out_dir = os.path.join(latex_dir, "t4")
    models  = model_order or sorted(df_singles_best["model_type"].unique())

    _lift_table(borda_global_singles, borda_global_ens, models, out_dir)
    _ens_rank_table(df_ens_best_rq31, sk_ens, borda_global_ens, models, out_dir,
                    df_baseline=df_baseline, sk_mixed=sk_mixed)


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


def _ens_rank_table(df_ens, sk_ens, borda_global_ens, models, out_dir,
                    df_baseline=None, sk_mixed=None):
    """T4b: multi-metric rank table for ensemble-base performance."""
    # Rename base_type → model_type for uniform processing
    df = df_ens.rename(columns={"base_type": "model_type"})
    sk = sk_ens.rename(columns={"base_type": "model_type"}) \
         if "base_type" in sk_ens.columns else sk_ens

    # SA/D augmented df (uses MAE rows from df_ens)
    df_aug = None
    if df_baseline is not None:
        from aggregators.comparisons import add_ensemble_sa_d
        df_aug_raw = add_ensemble_sa_d(df_ens, df_baseline)
        df_aug = df_aug_raw.rename(columns={"base_type": "model_type"})

    # Mixed Borda rank — rank among all 16 competitors (8 singles + 8 ensembles)
    mixed_borda = {}
    if sk_mixed is not None:
        totals = (sk_mixed.groupby("competitor")["sk_rank"]
                  .sum()
                  .reset_index())
        totals["borda_rank"] = totals["sk_rank"].rank(method="min").astype(int)
        for model in models:
            row = totals[totals["competitor"] == f"{model}_E"]
            mixed_borda[model] = int(row["borda_rank"].values[0]) if not row.empty else "--"

    has_sa    = df_aug is not None
    has_borda = bool(mixed_borda)

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

            if has_sa:
                sa_v = df_aug[(df_aug["model_type"] == model) & (df_aug["metric"] == "SA")]["value"].values
                d_v  = df_aug[(df_aug["model_type"] == model) & (df_aug["metric"] == "D")]["value"].values
                row["sa_d"] = (
                    float(np.mean(sa_v)) if len(sa_v) > 0 else np.nan,
                    float(np.mean(d_v))  if len(d_v)  > 0 else np.nan,
                )

            row["borda_mixed"] = mixed_borda.get(model, "--")
            rows.append(row)

        best = {m: min(r[m][0] for r in rows if not isinstance(r[m][0], str)) for m in METRICS_EVAL}
        if has_sa:
            best_sa = max(r["sa_d"][0] for r in rows if not np.isnan(r["sa_d"][0]))

        extra_cols = (["SA (D)"] if has_sa else []) + (["Borda"] if has_borda else [])
        col_spec = "l" + "c" * (len(METRICS_EVAL) + len(extra_cols))
        lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            "Model & " + " & ".join(METRICS_EVAL + extra_cols) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            cells = [row["Model"]]
            for metric in METRICS_EVAL:
                c, s, rank_disp, mean_sk_num = row[metric]
                if isinstance(c, str):
                    cells.append("--"); continue
                cell = fmt_cell(c, s, sk_rank=rank_disp)
                if abs(c - best[metric]) < 1e-9:
                    cell = bold(cell)
                cells.append(cell)
            if has_sa:
                sa_c, d_c = row["sa_d"]
                sa_str = f"{sa_c:.3f} ({d_c:+.2f})" if not np.isnan(sa_c) else "--"
                if not np.isnan(sa_c) and abs(sa_c - best_sa) < 1e-9:
                    sa_str = bold(sa_str)
                cells.append(sa_str)
            if has_borda:
                cells.append(str(row["borda_mixed"]))
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save_tex(lines, os.path.join(out_dir, f"t4b_ens_rank_{agg}.tex"))
