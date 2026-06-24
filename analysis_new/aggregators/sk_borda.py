"""
Scott-Knott + Borda count over long-format result DataFrames.

Borda convention (Day 2 catalog):
  Borda total = sum of SK ranks across all (metric, dataset, sample_size).
  Lower total = better model.
"""
import pandas as pd
from .sk_impl import scott_knott

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

_SK_KWARGS = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}


def run_sk_on_df(df, group_col, metrics=None, sk_kwargs=None):
    """
    Run Scott-Knott per (metric, dataset, sample_size).

    Parameters
    ----------
    df        : long-format DataFrame with [dataset, sample_size, <group_col>, metric, value]
    group_col : column that identifies the treatment, e.g. "model_type" or "base_type"
    metrics   : list of metrics to run SK on (default: METRICS_EVAL)
    sk_kwargs : kwargs forwarded to scott_knott (default: project params)

    Returns
    -------
    DataFrame: [dataset, sample_size, metric, <group_col>, sk_rank]
    """
    if metrics is None:
        metrics = METRICS_EVAL
    if sk_kwargs is None:
        sk_kwargs = _SK_KWARGS

    rows = []
    for metric in metrics:
        sub_m = df[df["metric"] == metric]
        for (dataset, sample_size), grp in sub_m.groupby(["dataset", "sample_size"]):
            results_dict = {
                name: group["value"].tolist()
                for name, group in grp.groupby(group_col)
            }
            if not results_dict:
                continue
            if len(results_dict) == 1:
                for name in results_dict:
                    rows.append({
                        "dataset": dataset, "sample_size": sample_size,
                        "metric": metric, group_col: name, "sk_rank": 1,
                    })
                continue
            try:
                sk_result = scott_knott(
                    [(n, v) for n, v in results_dict.items()], **sk_kwargs
                )
                for rank, name, *_ in sk_result:
                    rows.append({
                        "dataset": dataset, "sample_size": sample_size,
                        "metric": metric, group_col: name, "sk_rank": int(rank),
                    })
            except Exception:
                for name in results_dict:
                    rows.append({
                        "dataset": dataset, "sample_size": sample_size,
                        "metric": metric, group_col: name, "sk_rank": 1,
                    })
    return pd.DataFrame(rows)


def compute_borda_global(sk_df, group_col, metrics=None):
    """
    Compute Borda totals from SK ranks.

    Returns
    -------
    borda_per_metric : DataFrame [metric, <group_col>, borda_total]
        Per-metric sum of SK ranks across all (dataset, sample_size). Lower = better.

    borda_global : DataFrame [<group_col>, borda_total_all]
        Sum across all metrics and scenarios. Sorted ascending (rank 1 = best).
    """
    if metrics is None:
        metrics = METRICS_EVAL

    sub = sk_df[sk_df["metric"].isin(metrics)]

    borda_per_metric = (
        sub.groupby(["metric", group_col])["sk_rank"]
        .sum()
        .reset_index()
        .rename(columns={"sk_rank": "borda_total"})
    )
    borda_global = (
        sub.groupby(group_col)["sk_rank"]
        .sum()
        .reset_index()
        .rename(columns={"sk_rank": "borda_total_all"})
        .sort_values("borda_total_all")
        .reset_index(drop=True)
    )
    # Add ordinal Borda rank column
    borda_global["borda_rank"] = range(1, len(borda_global) + 1)
    return borda_per_metric, borda_global


def compute_borda_per_dataset(sk_df, group_col, metrics=None):
    """
    Borda total per (dataset, group) by summing SK ranks across sample sizes.
    Used for F1 per-dataset rank heatmap.
    """
    if metrics is None:
        metrics = METRICS_EVAL
    sub = sk_df[sk_df["metric"].isin(metrics)]
    borda = (
        sub.groupby(["dataset", group_col])["sk_rank"]
        .sum()
        .reset_index()
        .rename(columns={"sk_rank": "borda_total"})
    )
    # Rank within each dataset: lower borda_total = better = lower rank
    borda["borda_rank"] = borda.groupby("dataset")["borda_total"].rank(method="min").astype(int)
    return borda
