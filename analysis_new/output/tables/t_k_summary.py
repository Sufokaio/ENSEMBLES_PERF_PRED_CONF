"""
T_K_SUMMARY: Tabular summary of optimal k analysis (RQ3.2).

24 rows (8 base types x 3 rules). Columns:
  k*(med)   — k with lowest global median MRE across all scenarios
  k*(mean)  — k with lowest global mean MRE across all scenarios
  k*(rank)  — k with lowest mean Friedman rank across scenarios
               (per scenario rank k values 1..N by median MRE; average those ranks)
  MRE(k=2)  — median MRE at k=2 (baseline)
  MRE(k*)   — median MRE at k*(med) (best achievable)
  Gain%     — (MRE(k=2) - MRE(k*)) / MRE(k=2) * 100
  k*>5 %    — % of 40 scenarios where per-scenario best k > 5
"""
import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

RULES = ["MEAN", "IRWM", "NN"]


def _best_k_by_agg(sub, agg_fn):
    agg = sub.groupby(["base_type", "rule", "k"])["value"].agg(agg_fn).reset_index()
    result = {}
    for (bt, rule), grp in agg.groupby(["base_type", "rule"]):
        if not grp.empty:
            result[(bt, rule)] = int(grp.loc[grp["value"].idxmin(), "k"])
    return result


def _best_k_by_sk_rank(sub):
    """k with lowest mean SK rank across scenarios (actual Scott-Knott test per scenario)."""
    from aggregators.sk_impl import scott_knott

    SK_KW = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}
    rows = []
    for (bt, rule, ds, ss), grp in sub.groupby(["base_type", "rule", "dataset", "sample_size"]):
        treatments = {int(k): v["value"].tolist() for k, v in grp.groupby("k")}
        if len(treatments) < 2:
            for k in treatments:
                rows.append({"base_type": bt, "rule": rule, "k": k, "sk_rank": 1})
            continue
        try:
            for rank, k, *_ in scott_knott([(k, v) for k, v in treatments.items()], **SK_KW):
                rows.append({"base_type": bt, "rule": rule, "k": k, "sk_rank": int(rank)})
        except Exception:
            pass

    if not rows:
        return {}
    df_sk = pd.DataFrame(rows)
    mean_sk = df_sk.groupby(["base_type", "rule", "k"])["sk_rank"].mean().reset_index()
    result = {}
    for (bt, rule), grp in mean_sk.groupby(["base_type", "rule"]):
        if not grp.empty:
            result[(bt, rule)] = int(grp.loc[grp["sk_rank"].idxmin(), "k"])
    return result


def generate(df_ens_raw, latex_dir, model_order=None):
    out_dir    = os.path.join(latex_dir, "t_k_summary")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())

    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"].copy()

    k_med  = _best_k_by_agg(sub, "median")
    k_mean = _best_k_by_agg(sub, "mean")
    k_rank = _best_k_by_sk_rank(sub)

    # Global aggregation per (base_type, rule, k)
    global_med = sub.groupby(["base_type", "rule", "k"])["value"].median().reset_index(name="med_mre")

    # Per-scenario best k for %k>5
    per_scen = sub.groupby(["base_type", "rule", "dataset", "sample_size", "k"])["value"].median().reset_index(name="med_mre")
    idx_best = per_scen.groupby(["base_type", "rule", "dataset", "sample_size"])["med_mre"].idxmin()
    best_k_per_scen = per_scen.loc[idx_best, ["base_type", "rule", "dataset", "sample_size", "k"]]

    rows = []
    for bt in base_types:
        for rule in RULES:
            g = global_med[(global_med["base_type"] == bt) & (global_med["rule"] == rule)]

            km  = k_med.get((bt, rule), "--")
            kmn = k_mean.get((bt, rule), "--")
            kr  = k_rank.get((bt, rule), "--")

            row_k2   = g[g["k"] == 2]
            mre_k2   = float(row_k2["med_mre"].values[0]) if not row_k2.empty else np.nan

            mre_best = np.nan
            if isinstance(km, int):
                row_best = g[g["k"] == km]
                mre_best = float(row_best["med_mre"].values[0]) if not row_best.empty else np.nan

            gain_pct = (mre_k2 - mre_best) / mre_k2 * 100 \
                       if not np.isnan(mre_k2) and not np.isnan(mre_best) and mre_k2 > 0 else np.nan

            bk = best_k_per_scen[
                (best_k_per_scen["base_type"] == bt) & (best_k_per_scen["rule"] == rule)
            ]
            pct_gt5 = float((bk["k"] > 5).mean() * 100) if not bk.empty else np.nan

            rows.append({"base_type": bt, "rule": rule,
                         "k_med": km, "k_mean": kmn, "k_rank": kr,
                         "mre_k2": mre_k2, "mre_best": mre_best,
                         "gain_pct": gain_pct, "pct_k_gt5": pct_gt5})

    df = pd.DataFrame(rows)

    col_spec = "ll" + "c" * 7
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Base type & Rule & $k^*_{\text{med}}$ & $k^*_{\text{mean}}$ & $k^*_{\text{rank}}$ & "
        r"MRE($k$=2) & MRE($k^*$) & Gain\,(\%) & $k^*>5$\,(\%) \\",
        r"\midrule",
    ]

    best_gain = df["gain_pct"].max()

    for bt in base_types:
        sub_bt = df[df["base_type"] == bt]
        first  = True
        for _, row in sub_bt.iterrows():
            bt_label = bt if first else ""
            first    = False

            gain_str = f"{row['gain_pct']:.1f}" if not np.isnan(row["gain_pct"]) else "--"
            pct_str  = f"{row['pct_k_gt5']:.0f}" if not np.isnan(row["pct_k_gt5"]) else "--"
            mre_k2   = f"{row['mre_k2']:.3f}"  if not np.isnan(row["mre_k2"])  else "--"
            mre_best = f"{row['mre_best']:.3f}" if not np.isnan(row["mre_best"]) else "--"

            if not np.isnan(row["gain_pct"]) and abs(row["gain_pct"] - best_gain) < 1e-9:
                gain_str = bold(gain_str)

            cells = [
                bt_label, str(row["rule"]),
                str(row["k_med"]), str(row["k_mean"]), str(row["k_rank"]),
                mre_k2, mre_best, gain_str, pct_str,
            ]
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines += [
        r"\multicolumn{9}{l}{\footnotesize "
        r"$k^*_{\text{med}}$: best $k$ by global median MRE. "
        r"$k^*_{\text{mean}}$: best $k$ by global mean MRE. "
        r"$k^*_{\text{rank}}$: best $k$ by mean SK rank (Scott-Knott per scenario). "
        r"Gain: \% MRE improvement from $k$=2 to $k^*_{\text{med}}$. "
        r"$k^*>5$: \% of 40 scenarios where per-scenario best $k > 5$.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_k_summary.tex"))


def generate_by_base(df_ens_raw, latex_dir, model_order=None):
    """Compact 8-row table: per base type, rules aggregated.

    Rows = base type (8). Columns:
      k*(med)   — k with lowest global median MRE across all rules & datasets
      MRE(k=2)  — median MRE at k=2 (agg. across rules & datasets)
      MRE(k*)   — median MRE at k* (agg. across rules & datasets)
      Gain%     — (MRE(k=2) − MRE(k*)) / MRE(k=2) × 100
      k*>5 %    — % of 40×3=120 (scenario, rule) combos where per-combo best k > 5
    """
    out_dir    = os.path.join(latex_dir, "t_k_summary")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())

    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"].copy()

    # Global aggregation across rules AND datasets
    global_med = (
        sub.groupby(["base_type", "k"])["value"]
        .median().reset_index(name="med_mre")
    )

    # Per-scenario-rule best k (for %k>5)
    per_combo = (
        sub.groupby(["base_type", "rule", "dataset", "sample_size", "k"])["value"]
        .median().reset_index(name="med_mre")
    )
    idx_best = per_combo.groupby(["base_type", "rule", "dataset", "sample_size"])["med_mre"].idxmin()
    best_k_pc = per_combo.loc[idx_best, ["base_type", "rule", "dataset", "sample_size", "k"]]

    rows = []
    for bt in base_types:
        g = global_med[global_med["base_type"] == bt]

        km  = int(g.loc[g["med_mre"].idxmin(), "k"]) if not g.empty else "--"

        row_k2   = g[g["k"] == 2]
        mre_k2   = float(row_k2["med_mre"].values[0]) if not row_k2.empty else np.nan

        mre_best = np.nan
        if isinstance(km, int):
            row_best = g[g["k"] == km]
            mre_best = float(row_best["med_mre"].values[0]) if not row_best.empty else np.nan

        gain_pct = (mre_k2 - mre_best) / mre_k2 * 100 \
                   if not np.isnan(mre_k2) and not np.isnan(mre_best) and mre_k2 > 0 else np.nan

        bk = best_k_pc[best_k_pc["base_type"] == bt]
        pct_gt5 = float((bk["k"] > 5).mean() * 100) if not bk.empty else np.nan

        rows.append({"base_type": bt, "k_med": km,
                     "mre_k2": mre_k2, "mre_best": mre_best,
                     "gain_pct": gain_pct, "pct_k_gt5": pct_gt5})

    df = pd.DataFrame(rows)
    best_gain = df["gain_pct"].max()

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Base type & $k^*$ & MRE($k$=2) & MRE($k^*$) & Gain\,(\%) & $k^*>5$\,(\%) \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        gain_str = f"{row['gain_pct']:.1f}" if not np.isnan(row["gain_pct"]) else "--"
        pct_str  = f"{row['pct_k_gt5']:.0f}" if not np.isnan(row["pct_k_gt5"]) else "--"
        mre_k2   = f"{row['mre_k2']:.3f}"  if not np.isnan(row["mre_k2"])  else "--"
        mre_best = f"{row['mre_best']:.3f}" if not np.isnan(row["mre_best"]) else "--"
        if not np.isnan(row["gain_pct"]) and abs(row["gain_pct"] - best_gain) < 1e-9:
            gain_str = bold(gain_str)
        cells = [str(row["base_type"]), str(row["k_med"]), mre_k2, mre_best, gain_str, pct_str]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize "
        r"$k^*$: $k$ with lowest global median MRE aggregated across all rules and datasets. "
        r"Gain: \% MRE improvement from $k$=2 to $k^*$. "
        r"$k^*>5$: \% of 120 (scenario$\times$rule) combinations where per-combination best $k > 5$.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_k_summary_bybase.tex"))
