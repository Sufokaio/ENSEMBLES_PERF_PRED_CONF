"""
Pipeline entry point for the CSI resubmission analysis.

Stages (run independently or all together):
  --load       Read all JSON result files; cache to parquet.
  --aggregate  Best-variant selection + SK + Borda; cache intermediates.
  --tables     Generate all LaTeX table fragments.
  --figures    Generate all PDF figures.
  --all        Run all stages end to end.

Usage examples:
  python run_all.py --all
  python run_all.py --load
  python run_all.py --aggregate                          # default: median MRE selection
  python run_all.py --aggregate --sel-metric MAE --sel-agg mean
  python run_all.py --tables --figures

The cache lives in output_artifacts/cache/ (parquet files).
Outputs live in output_artifacts/latex/ and output_artifacts/figures/.
"""
import argparse
import os
import sys

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
import config as cfg

os.makedirs(cfg.CACHE_DIR,   exist_ok=True)
os.makedirs(cfg.LATEX_DIR,   exist_ok=True)
os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

# ── cache helpers ───────────────────────────────────────────────────────────
def _cache(name):
    return os.path.join(cfg.CACHE_DIR, f"{name}.parquet")

_FORCE = False  # set by --force flag at startup


def _save(df, name):
    df.to_parquet(_cache(name), index=False)
    print(f"  cached {name} ({len(df):,} rows)")


def _load(name):
    p = _cache(name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Cache file missing: {p}\n"
            "Run --load (and --aggregate if needed) first."
        )
    return pd.read_parquet(p)


def _skip(name):
    """Return True (and print a message) if the cache exists and --force not set."""
    if not _FORCE and os.path.exists(_cache(name)):
        print(f"  skip {name} (already cached; use --force to recompute)")
        return True
    return False


# ── stage: load ─────────────────────────────────────────────────────────────
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


# ── stage: aggregate ─────────────────────────────────────────────────────────
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

    # ── singles best-variant ────────────────────────────────────────────
    if not _skip("singles_best"):
        print(f"Selecting best single configs by {sel_agg}({sel_metric}) …")
        df_singles_best = select_best_singles(df_singles_raw, sel_metric=sel_metric, sel_agg=sel_agg)
        _save(df_singles_best, "singles_best")
    df_singles_best = _load("singles_best")

    # ── ensembles best-variant (RQ2) ────────────────────────────────────
    if not _skip("ensembles_best_rq2"):
        print(f"Selecting best ensemble per scenario (RQ2) by {sel_agg}({sel_metric}) …")
        df_ens_best_rq2 = select_best_ensembles_rq2(df_ens_raw, sel_metric=sel_metric, sel_agg=sel_agg)
        _save(df_ens_best_rq2, "ensembles_best_rq2")
    df_ens_best_rq2 = _load("ensembles_best_rq2")

    # ── ensembles best-k per rule (RQ3.3) ───────────────────────────────
    if not _skip("ensembles_best_rq33"):
        print(f"Selecting best k per rule (RQ3.3) by {sel_agg}({sel_metric}) …")
        df_ens_rq33 = select_best_ensembles_rq33(df_ens_raw, sel_metric=sel_metric, sel_agg=sel_agg)
        _save(df_ens_rq33, "ensembles_best_rq33")
    df_ens_rq33 = _load("ensembles_best_rq33")

    # ── SK + Borda on singles ────────────────────────────────────────────
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

    # ── SK + Borda on ensemble-base (RQ3.1) ─────────────────────────────
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

    # ── SK on RQ3.3 (per rule) ───────────────────────────────────────────
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

    # ── W/T/L ───────────────────────────────────────────────────────────
    if not _skip("wtl_median"):
        print("Computing W/T/L (median) …")
        wtl_median = compute_wtl(df_singles_best, df_ens_best_rq2, df_base, agg="median")
        _save(wtl_median, "wtl_median")

    if not _skip("wtl_mean"):
        print("Computing W/T/L (mean) …")
        wtl_mean = compute_wtl(df_singles_best, df_ens_best_rq2, df_base, agg="mean")
        _save(wtl_mean, "wtl_mean")

    print("Aggregation done.")


# ── stage: tables ─────────────────────────────────────────────────────────────
def stage_tables():
    print("\n=== STAGE: tables ===")
    from output.tables import (
        t1_singles_rank, t2_rank_disagree, t3_ensemble_wtl,
        t4_lift, t5_rule_battle,
    )

    df_singles_best   = _load("singles_best")
    sk_singles        = _load("sk_singles")
    borda_pm_singles  = _load("borda_per_metric_singles")
    borda_gl_singles  = _load("borda_global_singles")
    df_ens_best_rq2   = _load("ensembles_best_rq2")
    df_ens_rq33       = _load("ensembles_best_rq33")
    sk_ens_rq31_raw   = _load("sk_ens_rq31")
    sk_rq33           = _load("sk_rq33")
    borda_gl_ens_rq31 = _load("borda_global_ens_rq31")
    wtl_median        = _load("wtl_median")
    wtl_mean          = _load("wtl_mean")

    model_order = cfg.MODEL_TYPES

    print("T1: singles rank table …")
    t1_singles_rank.generate(
        df_singles_best, sk_singles, borda_pm_singles, borda_gl_singles,
        cfg.LATEX_DIR, model_order=model_order
    )

    print("T2: metric rank disagreement …")
    t2_rank_disagree.generate(
        borda_pm_singles, borda_gl_singles,
        cfg.LATEX_DIR, model_order=model_order
    )

    print("T3: ensemble W/T/L …")
    t3_ensemble_wtl.generate(wtl_median, cfg.LATEX_DIR, model_order=model_order)
    t3_ensemble_wtl.generate(wtl_mean,   cfg.LATEX_DIR, model_order=model_order)

    print("T4: lift table …")
    # Re-build sk_ens_rq31 with original base_type column for t4
    sk_ens_rq31 = sk_ens_rq31_raw.rename(columns={"model_type": "base_type"})
    # Also need borda for ensembles (rename back to model_type for t4 internals)
    borda_gl_ens_for_t4 = borda_gl_ens_rq31.rename(columns={"base_type": "model_type"})
    _, borda_gl_ens_full = _compute_borda_from_sk(sk_ens_rq31_raw, "model_type")
    borda_gl_ens_full_bt = borda_gl_ens_full.rename(columns={"model_type": "base_type"})

    t4_lift.generate(
        df_singles_best,
        df_ens_best_rq2,
        sk_singles,
        borda_gl_singles,
        sk_ens_rq31_raw,
        borda_gl_ens_for_t4,
        cfg.LATEX_DIR, model_order=model_order
    )

    print("T5: rule battle royale …")
    t5_rule_battle.generate(df_ens_rq33, sk_rq33, cfg.LATEX_DIR, model_order=model_order)

    print("All tables done.")


def _compute_borda_from_sk(sk_df, group_col):
    from aggregators.sk_borda import compute_borda_global
    return compute_borda_global(sk_df, group_col)


# ── stage: figures ─────────────────────────────────────────────────────────────
def stage_figures():
    print("\n=== STAGE: figures ===")
    from output.figures import (
        f1_singles_heatmap, f2_ensemble_gain, f3_k_trend,
        f4_wtl_bars, f5_sa_bars, f6_mibre_mre_scatter,
        f7_forest_plot, f8_cd_diagram, f9_parallel_coords,
    )

    df_singles_best    = _load("singles_best")
    df_ens_best_rq2    = _load("ensembles_best_rq2")
    df_ens_rq33        = _load("ensembles_best_rq33")
    df_ens_raw         = _load("ensembles_raw")
    df_base            = _load("baseline")
    borda_ds_singles   = _load("borda_per_dataset_singles")
    wtl_median         = _load("wtl_median")
    sk_rq33            = _load("sk_rq33")

    model_order   = cfg.MODEL_TYPES
    dataset_order = sorted(df_singles_best["dataset"].unique())

    print("F1: per-dataset rank heatmap …")
    f1_singles_heatmap.generate(
        borda_ds_singles, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F2: ensemble gain heatmap …")
    f2_ensemble_gain.generate(
        df_singles_best, df_ens_best_rq2, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F3: k-trend …")
    f3_k_trend.generate(df_ens_raw, cfg.FIGURES_DIR, model_order=model_order)

    print("F4: W/T/L bars …")
    f4_wtl_bars.generate_f4a(wtl_median, cfg.FIGURES_DIR, model_order=model_order)
    f4_wtl_bars.generate_f4b(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)

    print("F5: SA bars …")
    f5_sa_bars.generate(
        df_singles_best, df_ens_best_rq2, df_base, cfg.FIGURES_DIR,
        model_order=model_order
    )

    print("F6: MIBRE vs. MRE scatter …")
    f6_mibre_mre_scatter.generate(
        df_singles_best, cfg.FIGURES_DIR, model_order=model_order
    )

    print("F7: forest plot …")
    f7_forest_plot.generate(df_singles_best, cfg.FIGURES_DIR, model_order=model_order)

    print("F8: CD diagrams …")
    f8_cd_diagram.generate(df_singles_best, cfg.FIGURES_DIR, model_order=model_order)

    print("F9: parallel coordinates …")
    f9_parallel_coords.generate(df_singles_best, cfg.FIGURES_DIR, model_order=model_order)

    print("All figures done.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSI resubmission analysis pipeline")
    parser.add_argument("--load",       action="store_true", help="Load JSON results → parquet cache")
    parser.add_argument("--aggregate",  action="store_true", help="Best-variant + SK + Borda")
    parser.add_argument("--tables",     action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--figures",    action="store_true", help="Generate PDF figures")
    parser.add_argument("--all",        action="store_true", help="Run all stages")
    parser.add_argument("--sel-metric", default="MRE",
                        help="Metric used for best-variant selection (default: MRE)")
    parser.add_argument("--sel-agg",    default="median", choices=["median", "mean"],
                        help="Aggregation for best-variant selection (default: median)")
    parser.add_argument("--force",      action="store_true",
                        help="Recompute and overwrite existing cache files")
    args = parser.parse_args()

    global _FORCE
    _FORCE = args.force

    if not any([args.load, args.aggregate, args.tables, args.figures, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.all or args.load:
        stage_load()
    if args.all or args.aggregate:
        stage_aggregate(sel_metric=args.sel_metric, sel_agg=args.sel_agg)
    if args.all or args.tables:
        stage_tables()
    if args.all or args.figures:
        stage_figures()

    print("\nDone.")


if __name__ == "__main__":
    main()
