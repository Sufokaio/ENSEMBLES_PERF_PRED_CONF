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


def generate_threshold(df_ens_raw, latex_dir, model_order=None, threshold=0.90):
    """Table idea A: k-threshold table (RQ3.2).

    For each (base_type, rule): the smallest k that captures ≥`threshold` of the
    total MRE gain from k=2 to k=10.

    Answers: 'what is the practical minimum k?' — k_thresh=2 means gains plateau
    immediately; k_thresh=9 means you need many learners to extract value.

    Rows: base types. Column groups: MEAN | IRWM | NN, each showing k_thresh.
    """
    out_dir    = os.path.join(latex_dir, "t_k_summary")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())

    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"].copy()

    # Global median MRE per (base_type, rule, k)
    global_med = (
        sub.groupby(["base_type", "rule", "k"])["value"]
        .median().reset_index(name="med_mre")
    )

    def _threshold_k(g):
        """Smallest k where gain from k=2 to k >= threshold * (gain from k=2 to k=10)."""
        ks  = sorted(g["k"].unique())
        mre = {int(row["k"]): row["med_mre"] for _, row in g.iterrows()}
        mre2  = mre.get(2,  np.nan)
        mre10 = mre.get(10, np.nan)
        if np.isnan(mre2) or np.isnan(mre10):
            return "--"
        total_gain = mre2 - mre10
        if total_gain <= 0:
            return 2  # no gain at all — k=2 already achieves everything
        target = mre2 - threshold * total_gain  # MRE level that corresponds to `threshold` of gain
        for k in ks:
            if mre.get(k, np.nan) <= target:
                return k
        return ks[-1]  # need all learners

    records = {}
    for (bt, rule), grp in global_med.groupby(["base_type", "rule"]):
        records[(bt, rule)] = _threshold_k(grp)

    pct_label = f"{int(threshold * 100)}\\%"
    col_spec  = "l" + "c" * len(RULES)
    rule_header = " & ".join([f"\\textbf{{{r}}}" for r in RULES])

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Base type & " + rule_header + r" \\",
        rf"\multicolumn{{{1 + len(RULES)}}}{{l}}"
        rf"{{\footnotesize Min $k$ to capture {pct_label} of MRE gain from $k$=2 to $k$=10}} \\",
        r"\midrule",
    ]

    for bt in base_types:
        cells = [bt] + [str(records.get((bt, rule), "--")) for rule in RULES]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        rf"\multicolumn{{{1 + len(RULES)}}}{{l}}{{\footnotesize "
        rf"$k_{{\text{{thresh}}}}$: smallest $k$ where $\geq${pct_label} of the total MRE "
        r"reduction from $k$=2 to $k$=10 is achieved (median MRE, aggregated across all datasets). "
        r"Low value = gains plateau early; high value = more learners keep helping.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_k_threshold.tex"))


def generate_fixed_k(df_ens_raw, latex_dir, model_order=None, fixed_ks=(2, 5)):
    """Table idea B: Fixed-k sensitivity table (RQ3.2).

    For each base type (rules aggregated): MRE at k=optimal, and the % extra error
    incurred by using each fixed k instead.

    Rows: base types. Columns: k*(optimal) | MRE(k*) | %extra(k=2) | %extra(k=5) | %extra(k=10).

    Answers: 'if I always use k=5, how much do I sacrifice per model type?'
    A value near 0% means k=5 is fine; a larger value means you should tune k.
    """
    out_dir    = os.path.join(latex_dir, "t_k_summary")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())

    sub = df_ens_raw[df_ens_raw["metric"] == "MRE"].copy()

    # Global median across rules AND datasets
    global_med = (
        sub.groupby(["base_type", "k"])["value"]
        .median().reset_index(name="med_mre")
    )

    rows = []
    for bt in base_types:
        g    = global_med[global_med["base_type"] == bt]
        k_opt = int(g.loc[g["med_mre"].idxmin(), "k"])
        mre_opt = float(g[g["k"] == k_opt]["med_mre"].values[0])

        extra = {}
        for fk in list(fixed_ks) + [10]:
            row = g[g["k"] == fk]
            if row.empty:
                extra[fk] = np.nan
            else:
                mre_fk = float(row["med_mre"].values[0])
                extra[fk] = (mre_fk - mre_opt) / mre_opt * 100 if mre_opt > 0 else np.nan

        rows.append({"base_type": bt, "k_opt": k_opt, "mre_opt": mre_opt, **{f"extra_k{fk}": extra[fk] for fk in list(fixed_ks) + [10]}})

    df = pd.DataFrame(rows)

    fk_cols  = [f"extra_k{fk}" for fk in list(fixed_ks) + [10]]
    fk_heads = [f"\\%extra($k$={fk})" for fk in list(fixed_ks) + [10]]

    col_spec = "l" + "c" * (2 + len(fk_cols))
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Base type & $k^*$ & MRE($k^*$) & " + " & ".join(fk_heads) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        cells = [str(row["base_type"]), str(row["k_opt"]), f"{row['mre_opt']:.3f}"]
        for fk_col in fk_cols:
            v = row[fk_col]
            cells.append(f"{v:.1f}" if not np.isnan(v) else "--")
        lines.append(" & ".join(cells) + r" \\")

    n_cols = 3 + len(fk_cols)
    lines += [
        r"\bottomrule",
        rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
        r"$k^*$: optimal $k$ (lowest global median MRE, aggregated across all rules and datasets). "
        r"\%extra($k$=X): percentage MRE increase if $k$ is fixed at X instead of $k^*$ "
        r"(0\% = no cost; larger = tuning $k$ matters more).} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_k_fixed.tex"))
