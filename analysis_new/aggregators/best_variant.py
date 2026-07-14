# Best-variant selection per (model/base type, dataset, sample size) scenario.

import numpy as np
import pandas as pd

def _agg_fn(sel_agg):
    if sel_agg == "mean":
        return np.mean
    if sel_agg == "median":
        return np.median
    raise ValueError(f"sel_agg must be 'mean' or 'median', got {sel_agg!r}")

def select_best_singles(df, sel_metric="MRE", sel_agg="median"):
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
