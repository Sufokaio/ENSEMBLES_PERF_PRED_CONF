# Derived quantities: SA/D for ensembles, W/T/L, imp%, central tendency, mixed SK analysis, and cross-win matrix.

import numpy as np
import pandas as pd

METRICS_ERROR = ["MRE", "MAE", "MBRE", "MIBRE"]
METRICS_SA    = ["SA"]

def add_ensemble_sa_d(df_ens, df_baseline):
    mae = df_ens[df_ens["metric"] == "MAE"].copy()
    m = mae.merge(df_baseline[["dataset", "sample_size", "MAEp0", "Sp0"]],
                  on=["dataset", "sample_size"])

    sa = m.copy()
    sa["metric"] = "SA"
    sa["value"]  = 1.0 - sa["value"] / sa["MAEp0"]

    d = m.copy()
    d["metric"] = "D"
    d["value"]  = (d["value"] - d["MAEp0"]) / d["Sp0"]

    extra = pd.concat([sa, d], ignore_index=True).drop(columns=["MAEp0", "Sp0"])
    return pd.concat([df_ens, extra], ignore_index=True)

def compute_central(df, group_cols, agg="median"):
    fn = np.median if agg == "median" else np.mean
    return (
        df.groupby(group_cols)["value"]
        .agg(fn)
        .reset_index()
        .rename(columns={"value": "central"})
    )

def compute_wtl(df_singles_best, df_ens_best, df_baseline,
                metrics=None, agg="median"):
    if metrics is None:
        metrics = METRICS_ERROR + METRICS_SA

    fn = np.median if agg == "median" else np.mean

    ens_aug = add_ensemble_sa_d(df_ens_best, df_baseline)

    sc = compute_central(
        df_singles_best[df_singles_best["metric"].isin(metrics)],
        ["model_type", "dataset", "sample_size", "metric"], agg
    ).rename(columns={"model_type": "base_type"})

    ec = compute_central(
        ens_aug[ens_aug["metric"].isin(metrics)],
        ["base_type", "dataset", "sample_size", "metric"], agg
    )

    merged = sc.merge(ec, on=["base_type", "dataset", "sample_size", "metric"],
                      suffixes=("_single", "_ens"))

    rows = []
    for (bt, metric), grp in merged.groupby(["base_type", "metric"]):
        lower_better = metric in METRICS_ERROR
        s = grp["central_single"]
        e = grp["central_ens"]
        n = len(grp)

        if lower_better:
            wins  = int((e < s).sum())
            losses = int((e > s).sum())
            imp   = ((s - e) / s.abs() * 100)
        else:
            wins  = int((e > s).sum())
            losses = int((e < s).sum())
            imp   = ((e - s) / s.abs() * 100)
        ties = n - wins - losses

        win_rate, lo, hi = _wilson_ci(wins, n)
        rows.append({
            "base_type":     bt,
            "metric":        metric,
            "W":             wins,
            "T":             ties,
            "L":             losses,
            "N":             n,
            "imp_pct_mean":  float(imp.mean()),
            "win_rate":      float(win_rate),
            "win_rate_lo":   float(lo),
            "win_rate_hi":   float(hi),
        })
    return pd.DataFrame(rows)

def compute_mixed_sk(df_singles_best, df_ens_best_rq2):
    from .sk_borda import run_sk_on_df

    METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]

    s = df_singles_best[df_singles_best["metric"].isin(METRICS)].copy()
    s["competitor"] = s["model_type"].astype(str) + "_S"
    s["base_type"]  = s["model_type"]
    s["kind"]       = "single"

    e = df_ens_best_rq2[df_ens_best_rq2["metric"].isin(METRICS)].copy()
    e["competitor"] = e["base_type"].astype(str) + "_E"
    e["kind"]       = "ensemble"

    combined = pd.concat([
        s[["dataset", "sample_size", "metric", "competitor", "base_type", "kind", "value", "run"]],
        e[["dataset", "sample_size", "metric", "competitor", "base_type", "kind", "value", "run"]],
    ], ignore_index=True)

    sk = run_sk_on_df(combined, group_col="competitor", metrics=METRICS)

    meta = combined[["competitor", "base_type", "kind"]].drop_duplicates()
    sk = sk.merge(meta, on="competitor", how="left")
    return sk

def compute_cross_win_matrix(df_singles_best, df_ens_best_rq2,
                              metric="MRE", agg="median"):
    fn = np.median if agg == "median" else np.mean
    base_types = sorted(df_singles_best["model_type"].unique())

    s_c = (
        df_singles_best[df_singles_best["metric"] == metric]
        .groupby(["model_type", "dataset", "sample_size"])["value"]
        .agg(fn)
        .reset_index()
        .rename(columns={"value": "single_val"})
    )
    e_c = (
        df_ens_best_rq2[df_ens_best_rq2["metric"] == metric]
        .groupby(["base_type", "dataset", "sample_size"])["value"]
        .agg(fn)
        .reset_index()
        .rename(columns={"value": "ens_val"})
    )

    mat = {}
    for base_i in base_types:
        row = {}
        ens_sub = e_c[e_c["base_type"] == base_i]
        for base_j in base_types:
            sing_sub = s_c[s_c["model_type"] == base_j]
            merged = ens_sub.merge(sing_sub, on=["dataset", "sample_size"])
            if len(merged) == 0:
                row[base_j] = np.nan
            else:
                row[base_j] = float((merged["ens_val"] < merged["single_val"]).mean() * 100)
        mat[base_i] = row

    return pd.DataFrame(mat).T.reindex(index=base_types, columns=base_types)

def compute_k_sk_ranks(df_ens_raw, metric="MRE"):
    from .sk_impl import scott_knott

    SK_KW = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}

    sub  = df_ens_raw[df_ens_raw["metric"] == metric].copy()
    rows = []

    for (bt, rule, ds, ss), grp in sub.groupby(
            ["base_type", "rule", "dataset", "sample_size"]):
        treatments = {
            int(k): vals["value"].tolist()
            for k, vals in grp.groupby("k")
        }
        if len(treatments) < 2:
            for k in treatments:
                rows.append({"base_type": bt, "rule": rule,
                             "dataset": ds, "sample_size": ss,
                             "k": k, "sk_rank": 1})
            continue
        try:
            for rank, k, *_ in scott_knott(
                    [(k, v) for k, v in treatments.items()], **SK_KW):
                rows.append({"base_type": bt, "rule": rule,
                             "dataset": ds, "sample_size": ss,
                             "k": int(k), "sk_rank": int(rank)})
        except Exception:
            pass

    return pd.DataFrame(rows)

def compute_k_sk_ranks_perrule(df_ens_raw, metric="MRE"):
    from .sk_borda import run_sk_on_df

    sub = df_ens_raw[df_ens_raw["metric"] == metric].copy()
    rows = []
    for rule in sub["rule"].unique():
        rule_sub = sub[sub["rule"] == rule].copy()
        rule_sub["competitor"] = (
            rule_sub["base_type"].astype(str) + "_k" + rule_sub["k"].astype(str)
        )
        sk = run_sk_on_df(rule_sub, group_col="competitor", metrics=[metric])
        meta = rule_sub[["competitor", "base_type", "k"]].drop_duplicates()
        sk = sk.merge(meta, on="competitor")
        sk["rule"] = rule
        rows.append(sk[["base_type", "rule", "dataset", "sample_size", "k", "sk_rank"]])

    return pd.concat(rows, ignore_index=True)

def compute_k_sk_ranks_global(df_ens_raw, metric="MRE"):
    from .sk_borda import run_sk_on_df

    sub = df_ens_raw[df_ens_raw["metric"] == metric].copy()
    sub["competitor"] = (
        sub["base_type"].astype(str) + "_" +
        sub["rule"].astype(str) + "_k" +
        sub["k"].astype(str)
    )

    sk = run_sk_on_df(sub, group_col="competitor", metrics=[metric])

    meta = sub[["competitor", "base_type", "rule", "k"]].drop_duplicates()
    sk   = sk.merge(meta, on="competitor")

    return (
        sk[["base_type", "rule", "dataset", "sample_size", "k", "sk_rank"]]
        .reset_index(drop=True)
    )

def _wilson_ci(successes, total, alpha=0.05):
    if total == 0:
        return 0.0, 0.0, 1.0
    from scipy.stats import norm
    z    = norm.ppf(1 - alpha / 2)
    p    = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z / denom * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return p, float(max(0.0, center - margin)), float(min(1.0, center + margin))
