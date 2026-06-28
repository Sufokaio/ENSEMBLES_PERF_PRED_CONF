"""
T_CROSS_WIN: Cross-level win matrix (RQ2).

8×8 table. Cell (i,j) = % of 40 scenarios where
ensemble of base_type i beats single of base_type j on MRE.

Diagonal (i==j) is the standard RQ2 comparison; bold there.
"""
import os
import numpy as np
import pandas as pd

from output.utils import bold, save_tex


def generate(cross_win_df, latex_dir, model_order=None):
    """
    Parameters
    ----------
    cross_win_df : 8×8 DataFrame (reset_index from parquet) or already indexed
                   Rows = ensemble base type, Cols = single base type.
                   If loaded from parquet (reset_index'd), the row labels are in
                   the first column.
    """
    out_dir = os.path.join(latex_dir, "t_cross_win")

    # Handle the _row_label column added when saving to parquet
    if "_row_label" in cross_win_df.columns:
        cwm = cross_win_df.set_index("_row_label")
        cwm.index.name = None
    elif "index" in cross_win_df.columns:
        cwm = cross_win_df.set_index("index")
    else:
        cwm = cross_win_df.copy()

    base_types = model_order or sorted(cwm.index.tolist())
    cwm = cwm.reindex(index=base_types, columns=base_types)

    col_spec = "l" + "c" * len(base_types)
    # Abbreviated column headers to fit
    short = {m: m[:3] for m in base_types}
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Ens $\backslash$ Sing & " + " & ".join(base_types) + r" \\",
        r"\midrule",
    ]

    for bt_i in base_types:
        cells = [bt_i]
        for bt_j in base_types:
            v = cwm.at[bt_i, bt_j]
            if np.isnan(v):
                cell = "--"
            else:
                cell = f"{v:.0f}\\%"
                if bt_i == bt_j:
                    cell = bold(cell)
                elif v >= 75:
                    cell = bold(cell)
        cells.append(cell if not np.isnan(v) else "--")
        # Rebuild properly:
        cells = [bt_i]
        for bt_j in base_types:
            v = cwm.at[bt_i, bt_j]
            if np.isnan(v):
                cells.append("--")
            else:
                cell = f"{v:.0f}\\%"
                if bt_i == bt_j or v >= 75:
                    cell = bold(cell)
                cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    n_cols = len(base_types) + 1
    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(n_cols) + r"}{l}{\footnotesize Cell $(i,j)$: \% of 40 scenarios where ensemble of row $i$ beats single of column $j$ on MRE. Bold diagonal = standard RQ2 pair. Bold off-diagonal $\geq$75\%.} \\",
        r"\end{tabular}",
    ]
    save_tex(lines, os.path.join(out_dir, "t_cross_win.tex"))
