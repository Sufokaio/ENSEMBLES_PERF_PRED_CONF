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

# Force UTF-8 output so Unicode characters in print() work on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
import config as cfg

os.makedirs(cfg.CACHE_DIR,   exist_ok=True)
os.makedirs(cfg.LATEX_DIR,   exist_ok=True)
os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

# ── cache helpers ───────────────────────────────────────────────────────────
# _SEL_SUFFIX is set at startup from --sel-agg.
# "median" (default) → no suffix (backwards compatible existing cache files).
# "mean"             → "_mean" appended to every cache name and output file.
_FORCE      = False
_SEL_SUFFIX = ""   # "" for median (default), "_mean" for mean


def _cache(name):
    return os.path.join(cfg.CACHE_DIR, f"{name}{_SEL_SUFFIX}.parquet")


def _out_dir(base_dir):
    """Return output directory, adding sel-suffix sub-folder when non-default."""
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
    # Raw input caches and RQ3.2 SK caches never carry a sel suffix
    suffix = "" if name in _RAW_CACHES else _SEL_SUFFIX
    p = os.path.join(cfg.CACHE_DIR, f"{name}{suffix}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Cache file missing: {p}\n"
            "Run --load (and --aggregate if needed) first."
        )
    return pd.read_parquet(p)


def _skip(name):
    """Return True (and print a message) if the cache exists and --force not set."""
    if not _FORCE and os.path.exists(_cache(name)):
        print(f"  skip {name}{_SEL_SUFFIX} (already cached; use --force to recompute)")
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
    """Best-variant selection uses sel_agg(sel_metric) — default median MRE,
    matching the conf-repo.  Tables/figures then produce median AND mean display
    variants from the same cached best-variant data."""
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

    # ── W/T/L for both display agg flavours ─────────────────────────────
    if not _skip("wtl_median"):
        print("Computing W/T/L (median) …")
        wtl_median = compute_wtl(df_singles_best, df_ens_best_rq2, df_base, agg="median")
        _save(wtl_median, "wtl_median")

    if not _skip("wtl_mean"):
        print("Computing W/T/L (mean) …")
        wtl_mean = compute_wtl(df_singles_best, df_ens_best_rq2, df_base, agg="mean")
        _save(wtl_mean, "wtl_mean")

    # ── Mixed SK (singles + ensembles together) ───────────────────────────
    if not _skip("sk_mixed"):
        print("Running mixed SK (singles + ensembles, 16 competitors) …")
        from aggregators.comparisons import compute_mixed_sk
        sk_mixed = compute_mixed_sk(df_singles_best, df_ens_best_rq2)
        _save(sk_mixed, "sk_mixed")

    # ── SK ranks for all k values (RQ3.2 statistical plots) ─────────────
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

    # ── Cross-win matrix ──────────────────────────────────────────────────
    if not _skip("cross_win_matrix"):
        print("Computing cross-win matrix (8x8) …")
        from aggregators.comparisons import compute_cross_win_matrix
        cwm = compute_cross_win_matrix(df_singles_best, df_ens_best_rq2)
        _save(cwm.reset_index().rename(columns={"index": "_row_label"}), "cross_win_matrix")

    print("Aggregation done.")


# ── stage: tables ─────────────────────────────────────────────────────────────
def stage_tables(sel_agg="median"):
    """sel_agg controls both which caches are loaded (via _SEL_SUFFIX) and
    the display aggregation in table cells."""
    print(f"\n=== STAGE: tables (sel_agg={sel_agg}, suffix='{_SEL_SUFFIX}') ===")
    # FINAL SET ONLY — imports trimmed to what is actually generated
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

    # ── FINAL: RQ1 ──────────────────────────────────────────────────────────
    print("T1 (mean, all): singles rank table …")
    t1_singles_rank.generate(
        df_singles_best, sk_singles, borda_pm_singles, borda_gl_singles,
        _out_dir(cfg.LATEX_DIR), model_order=model_order
    )

    # ── FINAL: RQ2 ──────────────────────────────────────────────────────────
    print("T3 (winrate, mean): ensemble W/T/L …")
    t3_ensemble_wtl.generate(wtl_mean, _out_dir(cfg.LATEX_DIR), model_order=model_order, agg_label="mean")
    # t3_ensemble_wtl.generate(wtl_median, ..., agg_label="median")  # not in final paper
    # t3_ensemble_wtl.generate_sk_diff(...)  # not in final paper

    # ── FINAL: RQ3.1 ────────────────────────────────────────────────────────
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

    # ── NOT IN FINAL PAPER (commented out) ──────────────────────────────────
    # t5_rule_battle.generate(...)
    # t7_disagree_matrix.generate(...)
    # t8_k_compare.generate(...)
    # t_sk_count.generate(...)
    # t_combined_rank.generate(...)
    # t_cross_win.generate(...)
    # t_combined_rank_dataset.generate(...)
    # t_lift_summary.generate(...)
    # t_k_summary.generate(...)
    # t_k_summary.generate_by_base(...)
    # t_k_vs_baseline.generate(...)   (global and per-rule, all/s1)
    # t_k_summary.generate_threshold(...)
    # t_k_summary.generate_fixed_k(...)

    print("All tables done.")


def _compute_borda_from_sk(sk_df, group_col):
    from aggregators.sk_borda import compute_borda_global
    return compute_borda_global(sk_df, group_col)


# ── stage: figures ─────────────────────────────────────────────────────────────
def stage_figures(sel_agg="median"):
    """sel_agg is passed through to individual figure generators that produce
    both median and mean display variants (mirrors conf-repo _t5/_t6/_t7 loops)."""
    print(f"\n=== STAGE: figures (display_agg={sel_agg}) ===")
    # FINAL SET ONLY — imports trimmed to what is actually generated
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

    # ── FINAL: RQ1 ──────────────────────────────────────────────────────────
    # f1_singles_heatmap.generate(...)  # not in final paper (all-scenarios Borda heatmap)
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

    # ── FINAL: RQ2 ──────────────────────────────────────────────────────────
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

    # ── FINAL: RQ3.2 ────────────────────────────────────────────────────────
    print("F_K_SK_RANK_CURVE (perrule, byrule): mean SK rank vs k per rule …")
    k_sk_ranks_pr = _load("k_sk_ranks_perrule")
    f_k_sk_rank_curve.generate_byrule(
        k_sk_ranks_pr, _out_dir(cfg.FIGURES_DIR), model_order=model_order, suffix="_perrule"
    )

    # ── FINAL: RQ3.3 ────────────────────────────────────────────────────────
    print("F_RQ33_COMBINED_HEATMAP (all + S1) …")
    f_rq33_combined_heatmap.generate(
        sk_rq33_fig, df_ens_rq33, _out_dir(cfg.FIGURES_DIR), model_order=model_order
    )
    f_rq33_combined_heatmap.generate_s1(
        sk_rq33_fig, df_ens_rq33, _out_dir(cfg.FIGURES_DIR), model_order=model_order
    )

    # ── NOT IN FINAL PAPER (commented out) ──────────────────────────────────
    # f2_ensemble_gain.generate / generate_s1
    # f4_wtl_bars.generate_f4a
    # f6_mibre_mre_scatter.generate
    # f7_forest_plot.generate
    # f8_cd_diagram.generate
    # f10_samplesize_trend.generate
    # f12_gain_4metric.generate / generate_s1
    # f13_sa_lift_heatmap.generate
    # f15_winrate_heatmap.generate
    # f17_type_rule_heatmap.generate / generate_s1 / generate_4metric
    # f18_k_marginal.generate
    # f21_rule_samplesize.generate / generate_per_base
    # f22_nn_scatter.generate
    # f_rq33_slope.generate / generate_s1
    # f_rq33_winner_map.generate / generate_s1 / generate_sk / generate_sk_s1
    # f_rq33_rule_rank_heatmap.generate / generate_s1
    # f_rq33_rule_bar.generate / generate_s1
    # f_rq33_rule_mre_heatmap.generate / generate_s1
    # f_lift_scatter.generate
    # f_gap_close2.generate
    # f_borda_mixed.generate
    # f_k_heatmap.generate / generate_s1
    # f_k_box.generate
    # f_k_elbow.generate / generate_mean / generate_s1_mean / generate_mean_byrule / generate_s1_mean_byrule
    # f_k_elbow_dataset.generate / generate_by_base / generate_by_base_s1 / generate_by_base_agg / generate_by_base_agg_s1
    # f_k_sk_rank_curve.generate / generate_s1 / generate_s1_byrule  (global and other perrule variants)
    # f_k_rank1_heatmap.generate / generate_s1
    # f_k_pct_rank1_curve.generate / generate_s1
    # f_k_vs_baseline_compare.generate / generate_s1
    # f_rule_violin.generate
    # f_rule_metric_heatmap.generate / generate_per_base / generate_s1
    # f_rank_stability.generate
    # f_dataset_champion.generate
    # f_metric_disagree_dataset.generate
    # f_profile_lines.generate
    # f_model_dataset_rank.generate
    # f_rank_flip_heatmap.generate
    # f_4metric_heatmap.generate
    # f_dataset_rank_profiles.generate
    # f_bump_chart.generate
    # f_metric_consistency.generate

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
                        help="Aggregation for best-variant selection AND display in tables/figures "
                             "(default: median, matching conf-repo)")
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
