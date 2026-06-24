"""
Derived quantities: SA/D for ensembles, W/T/L, imp%, central tendency.
"""
import numpy as np
import pandas as pd

METRICS_ERROR = ["MRE", "MAE", "MBRE", "MIBRE"]  # lower is better
METRICS_SA    = ["SA"]                              # higher is better


# ---------------------------------------------------------------------------
# SA / D computation for ensembles
# ---------------------------------------------------------------------------

def add_ensemble_sa_d(df_ens, df_baseline):
    """
    Append SA and D rows to an ensemble DataFrame using baseline statistics.
      SA_run = 1 - MAE_run / MAEp0
      D_run  = (MAE_run - MAEp0) / Sp0

    Parameters
    ----------
    df_ens      : ensemble long-format DataFrame (must include metric="MAE" rows)
    df_baseline : DataFrame with [dataset, sample_size, MAEp0, Sp0]

    Returns
    -------
    df_ens augmented with SA and D rows (same schema, metric column = "SA" or "D").
    """
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


# ---------------------------------------------------------------------------
# Central tendency
# ---------------------------------------------------------------------------

def compute_central(df, group_cols, agg="median"):
    """
    Aggregate 30-run values to a central tendency.

    Parameters
    ----------
    df         : long-format DataFrame with 'value' and 'run' columns
    group_cols : list of columns to group by (e.g. ["model_type", "dataset", "sample_size", "metric"])
    agg        : "median" or "mean"

    Returns
    -------
    DataFrame with group_cols + ["central"]
    """
    fn = np.median if agg == "median" else np.mean
    return (
        df.groupby(group_cols)["value"]
        .agg(fn)
        .reset_index()
        .rename(columns={"value": "central"})
    )


# ---------------------------------------------------------------------------
# W/T/L and imp%
# ---------------------------------------------------------------------------

def compute_wtl(df_singles_best, df_ens_best, df_baseline,
                metrics=None, agg="median"):
    """
    W/T/L and imp% for best ensemble vs. best single, per (model/base_type, metric).

    - For error metrics (MRE, MAE, MBRE, MIBRE): win = ensemble < single.
    - For SA: win = ensemble > single.
    - imp% = positive means ensemble is better.

    Returns
    -------
    DataFrame:
      base_type | metric | W | T | L | N | imp_pct_mean | win_rate
                | win_rate_lo | win_rate_hi
    """
    if metrics is None:
        metrics = METRICS_ERROR + METRICS_SA

    fn = np.median if agg == "median" else np.mean

    # Add SA/D to ensembles
    ens_aug = add_ensemble_sa_d(df_ens_best, df_baseline)

    # Central tendency for singles: group by model_type
    sc = compute_central(
        df_singles_best[df_singles_best["metric"].isin(metrics)],
        ["model_type", "dataset", "sample_size", "metric"], agg
    ).rename(columns={"model_type": "base_type"})

    # Central tendency for ensembles: group by base_type
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


def _wilson_ci(successes, total, alpha=0.05):
    """Wilson score interval. Returns (p_hat, lo, hi)."""
    if total == 0:
        return 0.0, 0.0, 1.0
    from scipy.stats import norm
    z    = norm.ppf(1 - alpha / 2)
    p    = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z / denom * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return p, float(max(0.0, center - margin)), float(min(1.0, center + margin))
