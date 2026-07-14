# Pipeline entry point for the analysis: load, aggregate, tables, figures.

import argparse
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

import config as cfg

os.makedirs(cfg.CACHE_DIR,   exist_ok=True)
os.makedirs(cfg.LATEX_DIR,   exist_ok=True)
os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

_FORCE      = False
_SEL_SUFFIX = ""

def _cache(name):
    return os.path.join(cfg.CACHE_DIR, f"{name}{_SEL_SUFFIX}.parquet")

def _out_dir(base_dir):
    if _SEL_SUFFIX:
        d = os.path.join(base_dir + _SEL_SUFFIX)
        os.makedirs(d, exist_ok=True)
        return d
    return base_dir

def _save(df, name):
    df.to_parquet(_cache(name), index=False)
    print(f"  cached {name}{_SEL_SUFFIX} ({len(df):,} rows)")

_RAW_CACHES = {"singles_raw", "ensembles_raw", "baseline",
               "k_sk_ranks", "k_sk_ranks_perrule"}

def _load(name):
    suffix = "" if name in _RAW_CACHES else _SEL_SUFFIX
    p = os.path.join(cfg.CACHE_DIR, f"{name}{suffix}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Cache file missing: {p}\n"
            "Run --load (and --aggregate if needed) first."
        )
    return pd.read_parquet(p)

def _skip(name):
    if not _FORCE and os.path.exists(_cache(name)):
        print(f"  skip {name}{_SEL_SUFFIX} (already cached; use --force to recompute)")
        return True
    return False

def stage_load():
    print("\n=== STAGE: load ===")
    from loaders import load_singles, load_ensembles, load_baseline

    print("Loading baseline files …")
    df_base = load_baseline(cfg.RESULTS_DIR)
    _save(df_base, "baseline")

    print("Loading single-model result files …")
    df_singles = load_singles(cfg.RESULTS_DIR)
    _save(df_singles, "singles_raw")

    print("Loading ensemble prediction files …")
    df_ens = load_ensembles(cfg.RESULTS_DIR)
    _save(df_ens, "ensembles_raw")

    print(f"  singles_raw: {len(df_singles):,} rows across "
          f"{df_singles['model_type'].nunique()} model types, "
          f"{df_singles['dataset'].nunique()} datasets")
    print(f"  ensembles_raw: {len(df_ens):,} rows")

def stage_aggregate(sel_metric="MRE", sel_agg="median"):
    print(f"\n=== STAGE: aggregate (sel_metric={sel_metric}, sel_agg={sel_agg}) ===")
    from aggregators import (
        select_best_singles, select_best_ensembles_rq2, select_best_ensembles_rq33,
        run_sk_on_df, compute_borda_global, compute_borda_per_dataset,
        compute_wtl,
    )

    df_singles_raw = _load("singles_raw")
    df_ens_raw     = _load("ensembles_raw")
    df_base        = _load("baseline")

    if not _skip("singles_best"):
        print(f"Selecting best single configs by {sel_agg}({sel_metric}) …")
        df_singles_best = select_best_singles(df_singles_raw, sel_metric=sel_metric, sel_agg=sel_agg)
        _save(df_singles_best, "singles_best")
    df_singles_best = _load("singles_best")

    if not _skip("ensembles_best_rq2"):
        print(f"Selecting best ensemble per scenario (RQ2) by {sel_agg}({sel_metric}) …")
        df_ens_best_rq2 = select_best_ensembles_rq2(df_ens_raw, sel_metric=sel_metric, sel_agg=sel_agg)
        _save(df_ens_best_rq2, "ensembles_best_rq2")
    df_ens_best_rq2 = _load("ensembles_best_rq2")

    if not _skip("ensembles_best_rq33"):
        print(f"Selecting best k per rule (RQ3.3) by {sel_agg}({sel_metric}) …")
        df_ens_rq33 = select_best_ensembles_rq33(df_ens_raw, sel_metric=sel_metric, sel_agg=sel_agg)
        _save(df_ens_rq33, "ensembles_best_rq33")
    df_ens_rq33 = _load("ensembles_best_rq33")

    if not _skip("sk_singles"):
        print("Running Scott-Knott on singles (all 4 metrics) …")
        sk_singles = run_sk_on_df(
            df_singles_best[df_singles_best["metric"].isin(cfg.METRICS)],
            group_col="model_type"
        )
        _save(sk_singles, "sk_singles")
    sk_singles = _load("sk_singles")

    if not _skip("borda_per_metric_singles"):
        borda_pm_singles, borda_gl_singles = compute_borda_global(sk_singles, "model_type")
        _save(borda_pm_singles, "borda_per_metric_singles")
        _save(borda_gl_singles,  "borda_global_singles")

    if not _skip("borda_per_dataset_singles"):
        sk_singles = _load("sk_singles")
        borda_ds_singles = compute_borda_per_dataset(sk_singles, "model_type")
        _save(borda_ds_singles, "borda_per_dataset_singles")

    if not _skip("sk_ens_rq31"):
        print("Running Scott-Knott on ensemble-base types (RQ3.1) …")
        sk_ens_rq31 = run_sk_on_df(
            df_ens_best_rq2[df_ens_best_rq2["metric"].isin(cfg.METRICS)].rename(
                columns={"base_type": "model_type"}
            ),
            group_col="model_type"
        )
        _save(sk_ens_rq31, "sk_ens_rq31")
    sk_ens_rq31 = _load("sk_ens_rq31")

    if not _skip("borda_global_ens_rq31"):
        _, borda_gl_ens_rq31 = compute_borda_global(sk_ens_rq31, "model_type")
        borda_gl_ens_rq31 = borda_gl_ens_rq31.rename(columns={"model_type": "base_type"})
        _save(borda_gl_ens_rq31, "borda_global_ens_rq31")

    if not _skip("sk_rq33"):
        print("Running Scott-Knott on ensembles per rule (RQ3.3) …")
        sk_rq33_rows = []
        for rule in cfg.RULES:
            sub = df_ens_rq33[df_ens_rq33["rule"] == rule]
            sk_r = run_sk_on_df(
                sub[sub["metric"].isin(cfg.METRICS)],
                group_col="base_type"
            )
            sk_r["rule"] = rule
            sk_rq33_rows.append(sk_r)
        sk_rq33 = pd.concat(sk_rq33_rows, ignore_index=True)
        _save(sk_rq33, "sk_rq33")

    if not _skip("wtl_median"):
        print("Computing W/T/L (median) …")
        wtl_median = compute_wtl(df_singles_best, df_ens_best_rq2, df_base, agg="median")
        _save(wtl_median, "wtl_median")

    if not _skip("wtl_mean"):
        print("Computing W/T/L (mean) …")
        wtl_mean = compute_wtl(df_singles_best, df_ens_best_rq2, df_base, agg="mean")
        _save(wtl_mean, "wtl_mean")

    if not _skip("sk_mixed"):
        print("Running mixed SK (singles + ensembles, 16 competitors) …")
        from aggregators.comparisons import compute_mixed_sk
        sk_mixed = compute_mixed_sk(df_singles_best, df_ens_best_rq2)
        _save(sk_mixed, "sk_mixed")

    if not _skip("k_sk_ranks"):
        print("Running global SK for all ensembles (RQ3.2) — 216 competitors per scenario …")
        from aggregators.comparisons import compute_k_sk_ranks_global
        k_sk_ranks = compute_k_sk_ranks_global(_load("ensembles_raw"))
        _save(k_sk_ranks, "k_sk_ranks")

    if not _skip("k_sk_ranks_perrule"):
        print("Running per-rule SK for RQ3.2 — 72 competitors (8 base × 9 k) per rule …")
        from aggregators.comparisons import compute_k_sk_ranks_perrule
        k_sk_ranks_pr = compute_k_sk_ranks_perrule(_load("ensembles_raw"))
        _save(k_sk_ranks_pr, "k_sk_ranks_perrule")

    if not _skip("cross_win_matrix"):
        print("Computing cross-win matrix (8x8) …")
        from aggregators.comparisons import compute_cross_win_matrix
        cwm = compute_cross_win_matrix(df_singles_best, df_ens_best_rq2)
        _save(cwm.reset_index().rename(columns={"index": "_row_label"}), "cross_win_matrix")

    print("Aggregation done.")

def stage_tables(sel_agg="median"):
    print(f"\n=== STAGE: tables (sel_agg={sel_agg}, suffix='{_SEL_SUFFIX}') ===")
    from output.tables import (
        t1_singles_rank, t3_ensemble_wtl, t4_lift,
    )

    df_singles_best   = _load("singles_best")
    sk_singles        = _load("sk_singles")
    borda_pm_singles  = _load("borda_per_metric_singles")
    borda_gl_singles  = _load("borda_global_singles")
    df_ens_best_rq2   = _load("ensembles_best_rq2")
    sk_ens_rq31_raw   = _load("sk_ens_rq31")
    borda_gl_ens_rq31 = _load("borda_global_ens_rq31")
    wtl_mean          = _load("wtl_mean")
    sk_mixed          = _load("sk_mixed")
    df_base           = _load("baseline")

    model_order = cfg.MODEL_TYPES

    print("T1 (mean, all): singles rank table …")
    t1_singles_rank.generate(
        df_singles_best, sk_singles, borda_pm_singles, borda_gl_singles,
        _out_dir(cfg.LATEX_DIR), model_order=model_order
    )

    print("T3 (winrate, mean): ensemble W/T/L …")
    t3_ensemble_wtl.generate(wtl_mean, _out_dir(cfg.LATEX_DIR), model_order=model_order, agg_label="mean")

    print("T4b (mean): ensemble base-type rank table …")
    borda_gl_ens_for_t4 = borda_gl_ens_rq31.rename(columns={"base_type": "model_type"})
    _, borda_gl_ens_full = _compute_borda_from_sk(sk_ens_rq31_raw, "model_type")
    t4_lift.generate(
        df_singles_best, df_ens_best_rq2,
        sk_singles, borda_gl_singles,
        sk_ens_rq31_raw, borda_gl_ens_for_t4,
        _out_dir(cfg.LATEX_DIR), model_order=model_order,
        df_baseline=df_base, sk_mixed=sk_mixed
    )

    print("All tables done.")

def _compute_borda_from_sk(sk_df, group_col):
    from aggregators.sk_borda import compute_borda_global
    return compute_borda_global(sk_df, group_col)

def stage_figures(sel_agg="median"):
    print(f"\n=== STAGE: figures (display_agg={sel_agg}) ===")
    from output.figures import (
        f1_singles_heatmap, f14_d_forest_rq2,
        f_gap_close, f_mre_abs_heatmap,
        f_k_sk_rank_curve,
        f_rq33_combined_heatmap,
    )

    df_singles_best  = _load("singles_best")
    df_ens_best_rq2  = _load("ensembles_best_rq2")
    df_ens_rq33      = _load("ensembles_best_rq33")
    sk_rq33_fig      = _load("sk_rq33")
    df_base          = _load("baseline")
    sk_singles       = _load("sk_singles")
    sk_mixed         = _load("sk_mixed")

    model_order   = cfg.MODEL_TYPES
    dataset_order = sorted(df_singles_best["dataset"].unique())

    print("F1-S1: rank heatmap S1 …")
    f1_singles_heatmap.generate_s1(
        sk_singles, _out_dir(cfg.FIGURES_DIR),
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_MRE_ABS_HEATMAP (mean, all + S1) …")
    f_mre_abs_heatmap.generate(
        df_singles_best, _out_dir(cfg.FIGURES_DIR),
        sk_singles=sk_singles,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F14: Δ forest single vs ensemble …")
    f14_d_forest_rq2.generate(
        df_singles_best, df_ens_best_rq2, df_base, _out_dir(cfg.FIGURES_DIR),
        model_order=model_order
    )

    print("F_GAP_CLOSE: SK rank gap heatmap …")
    f_gap_close.generate(
        sk_mixed, _out_dir(cfg.FIGURES_DIR),
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_K_SK_RANK_CURVE (perrule, byrule): mean SK rank vs k per rule …")
    k_sk_ranks_pr = _load("k_sk_ranks_perrule")
    f_k_sk_rank_curve.generate_byrule(
        k_sk_ranks_pr, _out_dir(cfg.FIGURES_DIR), model_order=model_order, suffix="_perrule"
    )

    print("F_RQ33_COMBINED_HEATMAP (all + S1) …")
    f_rq33_combined_heatmap.generate(
        sk_rq33_fig, df_ens_rq33, _out_dir(cfg.FIGURES_DIR), model_order=model_order
    )
    f_rq33_combined_heatmap.generate_s1(
        sk_rq33_fig, df_ens_rq33, _out_dir(cfg.FIGURES_DIR), model_order=model_order
    )

    print("All figures done.")

def main():
    parser = argparse.ArgumentParser(description="Analysis pipeline for the paper")
    parser.add_argument("--load",       action="store_true", help="Load JSON results → parquet cache")
    parser.add_argument("--aggregate",  action="store_true", help="Best-variant + SK + Borda")
    parser.add_argument("--tables",     action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--figures",    action="store_true", help="Generate PDF figures")
    parser.add_argument("--all",        action="store_true", help="Run all stages")
    parser.add_argument("--sel-metric", default="MRE",
                        help="Metric used for best-variant selection (default: MRE)")
    parser.add_argument("--sel-agg",    default="median", choices=["median", "mean"],
                        help="Aggregation for best-variant selection AND display in tables/figures "
                             "(default: median)")
    parser.add_argument("--force",      action="store_true",
                        help="Recompute and overwrite existing cache files")
    args = parser.parse_args()

    global _FORCE, _SEL_SUFFIX
    _FORCE      = args.force
    _SEL_SUFFIX = "" if args.sel_agg == "median" else f"_{args.sel_agg}"

    if not any([args.load, args.aggregate, args.tables, args.figures, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.all or args.load:
        stage_load()
    if args.all or args.aggregate:
        stage_aggregate(sel_metric=args.sel_metric, sel_agg=args.sel_agg)
    if args.all or args.tables:
        stage_tables(sel_agg=args.sel_agg)
    if args.all or args.figures:
        stage_figures(sel_agg=args.sel_agg)

    print("\nDone.")

if __name__ == "__main__":
    main()
