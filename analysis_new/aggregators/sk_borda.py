# Scott-Knott + Borda count over long-format result DataFrames.

import pandas as pd
from .sk_impl import scott_knott

METRICS_EVAL = ["MRE", "MAE", "MBRE", "MIBRE"]

_SK_KWARGS = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}

def run_sk_on_df(df, group_col, metrics=None, sk_kwargs=None):
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
    borda_global["borda_rank"] = range(1, len(borda_global) + 1)
    return borda_per_metric, borda_global

def compute_borda_per_dataset(sk_df, group_col, metrics=None):
    if metrics is None:
        metrics = METRICS_EVAL
    sub = sk_df[sk_df["metric"].isin(metrics)]
    borda = (
        sub.groupby(["dataset", group_col])["sk_rank"]
        .sum()
        .reset_index()
        .rename(columns={"sk_rank": "borda_total"})
    )
    borda["borda_rank"] = borda.groupby("dataset")["borda_total"].rank(method="min").astype(int)
    return borda
