# F_K_HEATMAP: Best k per (base_type, rule) — 8x3 heatmap (RQ3.2).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_utils import save_figure

RULES = ["MEAN", "IRWM", "NN"]

def _filter_s1(df):
    mins = df.groupby("dataset")["sample_size"].min().reset_index(name="_min")
    return df.merge(mins, on="dataset").query("sample_size == _min").drop(columns="_min")

def _best_k_by_agg(sub, agg_fn):
    agg = sub.groupby(["base_type", "rule", "k"])["value"].agg(agg_fn).reset_index()
    result = {}
    for (bt, rule), grp in agg.groupby(["base_type", "rule"]):
        if not grp.empty:
            result[(bt, rule)] = int(grp.loc[grp["value"].idxmin(), "k"])
    return result

def _best_k_by_sk_rank(sub):
    from aggregators.sk_impl import scott_knott

    SK_KW = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}

    rows = []
    for (bt, rule, ds, ss), grp in sub.groupby(["base_type", "rule", "dataset", "sample_size"]):
        treatments = {
            int(k): vals["value"].tolist()
            for k, vals in grp.groupby("k")
        }
        if len(treatments) < 2:
            for k, _ in treatments.items():
                rows.append({"base_type": bt, "rule": rule, "k": k, "sk_rank": 1})
            continue
        try:
            result = scott_knott([(k, v) for k, v in treatments.items()], **SK_KW)
            for rank, k, *_ in result:
                rows.append({"base_type": bt, "rule": rule, "k": k, "sk_rank": int(rank)})
        except Exception:
            pass

    if not rows:
        return {}

    df_sk = pd.DataFrame(rows)
    mean_sk = df_sk.groupby(["base_type", "rule", "k"])["sk_rank"].mean().reset_index()
    result_dict = {}
    for (bt, rule), grp in mean_sk.groupby(["base_type", "rule"]):
        if not grp.empty:
            result_dict[(bt, rule)] = int(grp.loc[grp["sk_rank"].idxmin(), "k"])
    return result_dict

def _draw_heatmap(best_k_dict, base_types, title, out_dir, fname):
    mat = np.full((len(base_types), len(RULES)), np.nan)
    for i, bt in enumerate(base_types):
        for j, rule in enumerate(RULES):
            v = best_k_dict.get((bt, rule))
            if v is not None:
                mat[i, j] = v

    k_min = int(np.nanmin(mat)); k_max = int(np.nanmax(mat))
    cmap  = matplotlib.cm.get_cmap("Blues", k_max - k_min + 1)

    fig, ax = plt.subplots(figsize=(4.0, 4.5))
    im = ax.imshow(mat, cmap=cmap, vmin=k_min - 0.5, vmax=k_max + 0.5, aspect="auto")
    ax.set_xticks(range(len(RULES)));    ax.set_xticklabels(RULES)
    ax.set_yticks(range(len(base_types))); ax.set_yticklabels(base_types)
    ax.set_xlabel("Combination rule"); ax.set_ylabel("Base model type")
    ax.set_title(title)

    for i in range(len(base_types)):
        for j in range(len(RULES)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if (v - k_min) / max(k_max - k_min, 1) > 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, ticks=range(k_min, k_max + 1))
    cbar.set_label("Best $k$")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, fname))

def generate(df_ens_raw, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_k_heatmap")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = df_ens_raw[df_ens_raw["metric"] == "MRE"]

    _draw_heatmap(
        _best_k_by_agg(sub, "median"), base_types,
        "Best $k$ by median MRE — all scenarios",
        out_dir, "f_k_heatmap_median_mre_all.pdf"
    )
    _draw_heatmap(
        _best_k_by_agg(sub, "mean"), base_types,
        "Best $k$ by mean MRE — all scenarios",
        out_dir, "f_k_heatmap_mean_mre_all.pdf"
    )
    _draw_heatmap(
        _best_k_by_sk_rank(sub), base_types,
        "Best $k$ by mean SK rank — all scenarios",
        out_dir, "f_k_heatmap_mean_sk_rank_all.pdf"
    )
    _draw_heatmap(
        _best_k_by_agg(sub, "median"), base_types,
        "Best $k$ by median MRE — all scenarios",
        out_dir, "f_k_heatmap.pdf"
    )

def generate_s1(df_ens_raw, figures_dir, model_order=None):
    out_dir    = os.path.join(figures_dir, "f_k_heatmap")
    base_types = model_order or sorted(df_ens_raw["base_type"].unique())
    sub        = _filter_s1(df_ens_raw[df_ens_raw["metric"] == "MRE"].copy())

    _draw_heatmap(
        _best_k_by_agg(sub, "median"), base_types,
        "Best $k$ by median MRE — S1 only",
        out_dir, "f_k_heatmap_median_mre_s1.pdf"
    )
    _draw_heatmap(
        _best_k_by_agg(sub, "mean"), base_types,
        "Best $k$ by mean MRE — S1 only",
        out_dir, "f_k_heatmap_mean_mre_s1.pdf"
    )
    _draw_heatmap(
        _best_k_by_sk_rank(sub), base_types,
        "Best $k$ by mean SK rank — S1 only",
        out_dir, "f_k_heatmap_mean_sk_rank_s1.pdf"
    )
    _draw_heatmap(
        _best_k_by_agg(sub, "median"), base_types,
        "Best $k$ by median MRE — S1 only",
        out_dir, "f_k_heatmap_s1.pdf"
    )
