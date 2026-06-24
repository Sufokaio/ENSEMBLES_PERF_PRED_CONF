"""
Best-variant selection — mirrors the conf-repo's select_best_per_scenario logic.

Parameters
----------
sel_metric : str, default "MRE"
    Which metric to use for selection (any metric in the DataFrame).
sel_agg    : str, default "median"
    How to aggregate the 30 runs before comparing configs: "median" or "mean".
    The conf-repo uses median; mean is available for sensitivity checks.

For singles:  per (model_type, dataset, sample_size), pick the config_id with the
  lowest sel_agg(sel_metric) across 30 runs.

For ensembles (RQ2): per (base_type, dataset, sample_size), pick the (k, rule)
  with the lowest sel_agg(sel_metric).  Rule and k treated as joint hyper-param.

For ensembles (RQ3.3): per (base_type, rule, dataset, sample_size), pick the k
  with the lowest sel_agg(sel_metric).  Rule is fixed; k is the only free parameter.
"""
import numpy as np
import pandas as pd


def _agg_fn(sel_agg):
    if sel_agg == "mean":
        return np.mean
    if sel_agg == "median":
        return np.median
    raise ValueError(f"sel_agg must be 'mean' or 'median', got {sel_agg!r}")


def select_best_singles(df, sel_metric="MRE", sel_agg="median"):
    """
    Input:  long-format singles DataFrame (all configs).
    Output: same schema but only the best config per (model_type, dataset, sample_size).
            config_id column is dropped.
    """
    mdf = df[df["metric"] == sel_metric].copy()
    agg = (
        mdf.groupby(["model_type", "dataset", "sample_size", "config_id"])["value"]
        .agg(_agg_fn(sel_agg))
        .reset_index()
        .rename(columns={"value": "_agg"})
    )
    idx  = agg.groupby(["model_type", "dataset", "sample_size"])["_agg"].idxmin()
    best = agg.loc[idx, ["model_type", "dataset", "sample_size", "config_id"]]

    out = df.merge(best, on=["model_type", "dataset", "sample_size", "config_id"])
    return out.drop(columns=["config_id"]).reset_index(drop=True)


def select_best_ensembles_rq2(df, sel_metric="MRE", sel_agg="median"):
    """
    Input:  long-format ensembles DataFrame (all k and all rules).
    Output: best (k, rule) per (base_type, dataset, sample_size).
    """
    mdf = df[df["metric"] == sel_metric].copy()
    agg = (
        mdf.groupby(["base_type", "dataset", "sample_size", "k", "rule"])["value"]
        .agg(_agg_fn(sel_agg))
        .reset_index()
        .rename(columns={"value": "_agg"})
    )
    idx  = agg.groupby(["base_type", "dataset", "sample_size"])["_agg"].idxmin()
    best = agg.loc[idx, ["base_type", "dataset", "sample_size", "k", "rule"]]

    return df.merge(best, on=["base_type", "dataset", "sample_size", "k", "rule"]).reset_index(drop=True)


def select_best_ensembles_rq33(df, sel_metric="MRE", sel_agg="median"):
    """
    Per (base_type, rule, dataset, sample_size): pick the k with lowest sel_agg(sel_metric).
    Rule is fixed; k is the tunable hyperparameter.  Used for T5 / RQ3.3 comparison.
    """
    mdf = df[df["metric"] == sel_metric].copy()
    agg = (
        mdf.groupby(["base_type", "rule", "dataset", "sample_size", "k"])["value"]
        .agg(_agg_fn(sel_agg))
        .reset_index()
        .rename(columns={"value": "_agg"})
    )
    idx  = agg.groupby(["base_type", "rule", "dataset", "sample_size"])["_agg"].idxmin()
    best = agg.loc[idx, ["base_type", "rule", "dataset", "sample_size", "k"]]

    return df.merge(best, on=["base_type", "rule", "dataset", "sample_size", "k"]).reset_index(drop=True)
