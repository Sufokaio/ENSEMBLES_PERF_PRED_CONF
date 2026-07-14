# T_LIFT_SUMMARY: Compact ensemble-vs-single lift summary per base type (RQ2).

import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex

def generate(sk_mixed, latex_dir, model_order=None):
    out_dir = os.path.join(latex_dir, "t_lift_summary")

    sub = sk_mixed[sk_mixed["metric"] == "MRE"].copy()

    base_types = model_order or sorted(
        sk_mixed[sk_mixed["kind"] == "single"]["base_type"].dropna().unique()
    )

    rows = []
    for bt in base_types:
        comp_s = f"{bt}_S"
        comp_e = f"{bt}_E"

        s_rank = sub[sub["competitor"] == comp_s]["sk_rank"].mean()
        e_rank = sub[sub["competitor"] == comp_e]["sk_rank"].mean()
        delta  = e_rank - s_rank

        overall = sub.groupby("competitor")["sk_rank"].mean()
        borda   = overall.rank(method="average")
        s_borda = borda.get(comp_s, np.nan)
        e_borda = borda.get(comp_e, np.nan)
        d_borda = e_borda - s_borda

        ds_ranks = sub.groupby(["competitor", "dataset"])["sk_rank"].mean().unstack("dataset")
        datasets = ds_ranks.columns.tolist()
        wins = 0
        for ds in datasets:
            sv = ds_ranks.at[comp_s, ds] if comp_s in ds_ranks.index else np.nan
            ev = ds_ranks.at[comp_e, ds] if comp_e in ds_ranks.index else np.nan
            if not np.isnan(sv) and not np.isnan(ev) and ev < sv:
                wins += 1
        ds_win_pct = wins / len(datasets) * 100 if datasets else np.nan

        rows.append({
            "base_type": bt,
            "S_rank": s_rank, "E_rank": e_rank, "Delta": delta,
            "S_Borda": s_borda, "E_Borda": e_borda, "D_Borda": d_borda,
            "DS_wins": ds_win_pct,
        })

    df = pd.DataFrame(rows).sort_values("Delta")

    col_spec = "l" + "c" * 7
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Model & $\bar{r}_S$ & $\bar{r}_E$ & $\Delta r$ "
        r"& $b_S$ & $b_E$ & $\Delta b$ & DS wins (\%) \\",
        r"\midrule",
    ]

    best_delta  = df["Delta"].min()
    best_dborda = df["D_Borda"].min()

    for _, row in df.iterrows():
        delta_str  = f"{row['Delta']:+.2f}"
        dborda_str = f"{row['D_Borda']:+.1f}"
        if abs(row["Delta"]  - best_delta)  < 1e-9:
            delta_str  = bold(delta_str)
        if abs(row["D_Borda"] - best_dborda) < 1e-9:
            dborda_str = bold(dborda_str)

        cells = [
            row["base_type"],
            f"{row['S_rank']:.2f}",
            f"{row['E_rank']:.2f}",
            delta_str,
            f"{row['S_Borda']:.0f}",
            f"{row['E_Borda']:.0f}",
            dborda_str,
            f"{row['DS_wins']:.0f}\\%",
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{\footnotesize "
        r"$\bar{r}_S$/$\bar{r}_E$: mean SK rank (MRE) of single/ensemble in the "
        r"mixed 16-competitor ranking. "
        r"$\Delta r = \bar{r}_E - \bar{r}_S$ (negative = ensemble improved). "
        r"$b_S$/$b_E$: Borda rank among 16 competitors. "
        r"DS wins: \% of 8 datasets where ensemble has lower mean SK rank than single. "
        r"Sorted by $\Delta r$.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_lift_summary.tex"))
