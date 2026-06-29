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
        print("Running SK per scenario across k values (RQ3.2) …")
        from aggregators.comparisons import compute_k_sk_ranks
        k_sk_ranks = compute_k_sk_ranks(_load("ensembles_raw"))
        _save(k_sk_ranks, "k_sk_ranks")

    # ── Cross-win matrix ──────────────────────────────────────────────────
    if not _skip("cross_win_matrix"):
        print("Computing cross-win matrix (8x8) …")
        from aggregators.comparisons import compute_cross_win_matrix
        cwm = compute_cross_win_matrix(df_singles_best, df_ens_best_rq2)
        _save(cwm.reset_index().rename(columns={"index": "_row_label"}), "cross_win_matrix")

    print("Aggregation done.")


# ── stage: tables ─────────────────────────────────────────────────────────────
def stage_tables(sel_agg="median"):
    """sel_agg controls the display aggregation in tables (median or mean of the 30
    runs shown in each cell).  Best-variant selection is fixed in the cache."""
    print(f"\n=== STAGE: tables (display_agg={sel_agg}) ===")
    from output.tables import (
        t1_singles_rank, t3_ensemble_wtl,
        t4_lift, t5_rule_battle,
        t7_disagree_matrix, t8_k_compare,
        t_sk_count, t_combined_rank, t_cross_win,
        t_combined_rank_dataset, t_lift_summary,
        t_k_summary,
        t_k_vs_baseline,
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
    sk_mixed          = _load("sk_mixed")
    cross_win_raw     = _load("cross_win_matrix")

    model_order = cfg.MODEL_TYPES

    print("T1: singles rank table …")
    t1_singles_rank.generate(
        df_singles_best, sk_singles, borda_pm_singles, borda_gl_singles,
        cfg.LATEX_DIR, model_order=model_order
    )

    print("T3: ensemble W/T/L …")
    t3_ensemble_wtl.generate(wtl_median, cfg.LATEX_DIR, model_order=model_order, agg_label="median")
    t3_ensemble_wtl.generate(wtl_mean,   cfg.LATEX_DIR, model_order=model_order, agg_label="mean")

    print("T3-SK: SK rank difference (mixed) …")
    t3_ensemble_wtl.generate_sk_diff(sk_mixed, cfg.LATEX_DIR, model_order=model_order)

    print("T4: ensemble rank table (T4b only) …")
    sk_ens_rq31 = sk_ens_rq31_raw.rename(columns={"model_type": "base_type"})
    borda_gl_ens_for_t4 = borda_gl_ens_rq31.rename(columns={"base_type": "model_type"})
    _, borda_gl_ens_full = _compute_borda_from_sk(sk_ens_rq31_raw, "model_type")
    t4_lift.generate(
        df_singles_best, df_ens_best_rq2,
        sk_singles, borda_gl_singles,
        sk_ens_rq31_raw, borda_gl_ens_for_t4,
        cfg.LATEX_DIR, model_order=model_order
    )

    print("T5: rule battle royale …")
    t5_rule_battle.generate(df_ens_rq33, sk_rq33, cfg.LATEX_DIR, model_order=model_order)

    print("T7: metric rank disagreement matrix …")
    t7_disagree_matrix.generate(df_singles_best, cfg.LATEX_DIR, model_order=model_order)

    print("T8: k=2 vs k=best vs k=10 sensitivity …")
    t8_k_compare.generate(
        _load("ensembles_raw"), cfg.LATEX_DIR, model_order=model_order, sel_agg=sel_agg
    )

    print("T_SK_COUNT: SK cluster membership counts …")
    t_sk_count.generate(sk_singles, cfg.LATEX_DIR, model_order=model_order)

    print("T_COMBINED_RANK: mixed singles+ensembles SK rank table …")
    t_combined_rank.generate(sk_mixed, cfg.LATEX_DIR, model_order=model_order)

    print("T_CROSS_WIN: cross-level win matrix …")
    t_cross_win.generate(cross_win_raw, cfg.LATEX_DIR, model_order=model_order)

    print("T_COMBINED_RANK_DATASET: per-dataset breakdown of mixed SK ranking …")
    t_combined_rank_dataset.generate(
        sk_mixed, cfg.LATEX_DIR,
        model_order=model_order, dataset_order=sorted(df_singles_best["dataset"].unique())
    )

    print("T_LIFT_SUMMARY: ensemble vs single lift summary …")
    t_lift_summary.generate(sk_mixed, cfg.LATEX_DIR, model_order=model_order)

    print("T_K_SUMMARY: optimal k tabular summary (by base type × rule) …")
    t_k_summary.generate(_load("ensembles_raw"), cfg.LATEX_DIR, model_order=model_order)
    print("T_K_SUMMARY by base type (rules aggregated) …")
    t_k_summary.generate_by_base(_load("ensembles_raw"), cfg.LATEX_DIR, model_order=model_order)
    print("T_K_VS_BASELINE: %% scenarios statistically better than k=2 …")
    k_sk_ranks = _load("k_sk_ranks")
    t_k_vs_baseline.generate(k_sk_ranks, cfg.LATEX_DIR, model_order=model_order)
    t_k_vs_baseline.generate_s1(k_sk_ranks, cfg.LATEX_DIR, model_order=model_order)

    print("T_K_THRESHOLD: min k to capture 90%% of gain …")
    t_k_summary.generate_threshold(_load("ensembles_raw"), cfg.LATEX_DIR, model_order=model_order)
    print("T_K_FIXED: % extra error at fixed k vs optimal …")
    t_k_summary.generate_fixed_k(_load("ensembles_raw"), cfg.LATEX_DIR, model_order=model_order)

    print("All tables done.")


def _compute_borda_from_sk(sk_df, group_col):
    from aggregators.sk_borda import compute_borda_global
    return compute_borda_global(sk_df, group_col)


# ── stage: figures ─────────────────────────────────────────────────────────────
def stage_figures(sel_agg="median"):
    """sel_agg is passed through to individual figure generators that produce
    both median and mean display variants (mirrors conf-repo _t5/_t6/_t7 loops)."""
    print(f"\n=== STAGE: figures (display_agg={sel_agg}) ===")
    from output.figures import (
        f1_singles_heatmap, f2_ensemble_gain,
        f4_wtl_bars, f6_mibre_mre_scatter,
        f7_forest_plot, f8_cd_diagram,
        f10_samplesize_trend, f12_gain_4metric,
        f13_sa_lift_heatmap, f14_d_forest_rq2, f15_winrate_heatmap,
        f17_type_rule_heatmap, f18_k_marginal,
        f21_rule_samplesize, f22_nn_scatter,
        f_lift_scatter, f_gap_close, f_borda_mixed,
        f_k_heatmap, f_k_box, f_k_elbow,
        f_rule_violin, f_rule_metric_heatmap,
        f_k_elbow_dataset,
        f_rank_stability, f_mre_abs_heatmap, f_dataset_champion,
        f_metric_disagree_dataset, f_profile_lines,
        f_model_dataset_rank, f_rank_flip_heatmap,
        f_4metric_heatmap, f_dataset_rank_profiles,
        f_bump_chart, f_metric_consistency,
        f_gap_close2,
        f_k_sk_rank_curve, f_k_rank1_heatmap,
    )

    df_singles_best    = _load("singles_best")
    df_ens_best_rq2    = _load("ensembles_best_rq2")
    df_ens_rq33        = _load("ensembles_best_rq33")
    df_ens_raw         = _load("ensembles_raw")
    df_base            = _load("baseline")
    borda_ds_singles   = _load("borda_per_dataset_singles")
    sk_singles         = _load("sk_singles")
    wtl_median         = _load("wtl_median")
    sk_mixed           = _load("sk_mixed")

    model_order   = cfg.MODEL_TYPES
    dataset_order = sorted(df_singles_best["dataset"].unique())

    print("F1: per-dataset rank heatmap …")
    f1_singles_heatmap.generate(
        borda_ds_singles, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )
    print("F1-S1: S1-only heatmap …")
    f1_singles_heatmap.generate_s1(
        sk_singles, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F2: ensemble gain heatmap (MRE) …")
    f2_ensemble_gain.generate(
        df_singles_best, df_ens_best_rq2, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )
    print("F2-S1: S1-only gain heatmap …")
    f2_ensemble_gain.generate_s1(
        df_singles_best, df_ens_best_rq2, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F4a: W/T/L bars …")
    f4_wtl_bars.generate_f4a(wtl_median, cfg.FIGURES_DIR, model_order=model_order)

    print("F6: MIBRE vs. MRE scatter …")
    f6_mibre_mre_scatter.generate(
        df_singles_best, cfg.FIGURES_DIR, model_order=model_order
    )

    print("F7: forest plot …")
    f7_forest_plot.generate(df_singles_best, cfg.FIGURES_DIR, model_order=model_order)

    print("F8: CD diagrams (3 variants) …")
    f8_cd_diagram.generate(
        df_singles_best, df_ens_best_rq2, df_ens_rq33,
        cfg.FIGURES_DIR, model_order=model_order
    )

    print("F10: sample-size trend …")
    f10_samplesize_trend.generate(
        df_singles_best, df_ens_best_rq2, cfg.FIGURES_DIR, model_order=model_order
    )

    print("F12: 4-metric gain heatmap …")
    f12_gain_4metric.generate(
        df_singles_best, df_ens_best_rq2, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )
    print("F12-S1: 4-metric gain heatmap S1 …")
    f12_gain_4metric.generate_s1(
        df_singles_best, df_ens_best_rq2, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F13: SA lift heatmap …")
    f13_sa_lift_heatmap.generate(
        df_singles_best, df_ens_best_rq2, df_base, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F14: D forest single vs ensemble ...")
    f14_d_forest_rq2.generate(
        df_singles_best, df_ens_best_rq2, df_base, cfg.FIGURES_DIR,
        model_order=model_order
    )

    print("F15: win rate heatmap …")
    f15_winrate_heatmap.generate(wtl_median, cfg.FIGURES_DIR, model_order=model_order)

    print("F17: type x rule interaction heatmap …")
    f17_type_rule_heatmap.generate(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)
    print("F17-S1: type x rule interaction S1 only …")
    f17_type_rule_heatmap.generate_s1(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)
    print("F17-4metric: type x rule across all 4 metrics …")
    f17_type_rule_heatmap.generate_4metric(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)

    print("F18: optimal k histogram + marginal gain …")
    f18_k_marginal.generate(df_ens_raw, cfg.FIGURES_DIR)

    print("F21: rule by sample size tier …")
    f21_rule_samplesize.generate(df_ens_rq33, df_base, cfg.FIGURES_DIR)
    print("F21-per-base: rule by sample size tier per base type …")
    f21_rule_samplesize.generate_per_base(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)

    print("F22: NN vs MEAN scatter (aggregated) …")
    f22_nn_scatter.generate(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)

    print("F_LIFT_SCATTER: SK rank lift single→ensemble …")
    f_lift_scatter.generate(sk_mixed, cfg.FIGURES_DIR, model_order=model_order)

    print("F_GAP_CLOSE: SK rank gap heatmap …")
    f_gap_close.generate(
        sk_mixed, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )
    print("F_GAP_CLOSE2: single-dataset zoom …")
    f_gap_close2.generate(sk_mixed, cfg.FIGURES_DIR, model_order=model_order)

    print("F_BORDA_MIXED: Borda score singles vs ensembles …")
    f_borda_mixed.generate(sk_mixed, cfg.FIGURES_DIR, model_order=model_order)

    print("F_K_HEATMAP: best k per base type × rule …")
    f_k_heatmap.generate(df_ens_raw, cfg.FIGURES_DIR, model_order=model_order)
    print("F_K_HEATMAP-S1: best k S1 only …")
    f_k_heatmap.generate_s1(df_ens_raw, cfg.FIGURES_DIR, model_order=model_order)

    print("F_K_BOX: MRE distribution per k …")
    f_k_box.generate(df_ens_raw, cfg.FIGURES_DIR)

    print("F_K_ELBOW: MRE vs k per base type (median, all scenarios) …")
    f_k_elbow.generate(df_ens_raw, cfg.FIGURES_DIR, model_order=model_order)
    print("F_K_ELBOW mean (all scenarios) …")
    f_k_elbow.generate_mean(df_ens_raw, cfg.FIGURES_DIR, model_order=model_order)
    print("F_K_ELBOW mean S1 (per-dataset min sample size) …")
    f_k_elbow.generate_s1_mean(df_ens_raw, cfg.FIGURES_DIR, model_order=model_order)
    print("F_K_ELBOW_DATASET: MRE vs k per dataset (by rule) …")
    f_k_elbow_dataset.generate(
        df_ens_raw, cfg.FIGURES_DIR, dataset_order=dataset_order
    )
    print("F_K_ELBOW_DATASET by base type (all datasets) …")
    f_k_elbow_dataset.generate_by_base(
        df_ens_raw, cfg.FIGURES_DIR,
        dataset_order=dataset_order, model_order=model_order
    )
    print("F_K_ELBOW_DATASET by base type (S1 only) …")
    f_k_elbow_dataset.generate_by_base_s1(
        df_ens_raw, cfg.FIGURES_DIR,
        dataset_order=dataset_order, model_order=model_order
    )
    print("F_K_SK_RANK_CURVE: mean SK rank vs k …")
    k_sk_ranks = _load("k_sk_ranks")
    f_k_sk_rank_curve.generate(k_sk_ranks, cfg.FIGURES_DIR, model_order=model_order)
    f_k_sk_rank_curve.generate_s1(k_sk_ranks, cfg.FIGURES_DIR, model_order=model_order)

    print("F_K_RANK1_HEATMAP: %% rank-1 per (base_type, rule, k) …")
    f_k_rank1_heatmap.generate(k_sk_ranks, cfg.FIGURES_DIR, model_order=model_order)
    f_k_rank1_heatmap.generate_s1(k_sk_ranks, cfg.FIGURES_DIR, model_order=model_order)

    print("F_K_ELBOW_DATASET by base type (aggregate, all scenarios) …")
    f_k_elbow_dataset.generate_by_base_agg(
        df_ens_raw, cfg.FIGURES_DIR, model_order=model_order
    )
    print("F_K_ELBOW_DATASET by base type (aggregate, S1 per dataset) …")
    f_k_elbow_dataset.generate_by_base_agg_s1(
        df_ens_raw, cfg.FIGURES_DIR, model_order=model_order
    )

    print("F_RULE_VIOLIN: MRE distribution per rule …")
    f_rule_violin.generate(df_ens_rq33, cfg.FIGURES_DIR)

    print("F_RULE_METRIC_HEATMAP: rule x metric …")
    f_rule_metric_heatmap.generate(df_ens_rq33, cfg.FIGURES_DIR)
    print("F_RULE_METRIC_HEATMAP per base type …")
    f_rule_metric_heatmap.generate_per_base(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)
    print("F_RULE_METRIC_HEATMAP S1 only …")
    f_rule_metric_heatmap.generate_s1(df_ens_rq33, cfg.FIGURES_DIR, model_order=model_order)

    print("F_RANK_STABILITY: per-model rank stability across datasets …")
    f_rank_stability.generate(
        borda_ds_singles, cfg.FIGURES_DIR, model_order=model_order
    )

    print("F_MRE_ABS_HEATMAP: absolute MRE heatmap models x datasets …")
    f_mre_abs_heatmap.generate(
        df_singles_best, cfg.FIGURES_DIR,
        sk_singles=sk_singles,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_DATASET_CHAMPION: per-dataset model dominance …")
    f_dataset_champion.generate(
        df_singles_best, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_METRIC_DISAGREE_DATASET: per-dataset metric disagreement …")
    f_metric_disagree_dataset.generate(
        df_singles_best, cfg.FIGURES_DIR, dataset_order=dataset_order
    )

    print("F_PROFILE_LINES: model MRE profiles across datasets …")
    f_profile_lines.generate(
        df_singles_best, cfg.FIGURES_DIR, model_order=model_order
    )

    print("F_MODEL_DATASET_RANK: HINNPerf vs DeepPerf dataset x tier rank …")
    f_model_dataset_rank.generate(
        df_singles_best, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_RANK_FLIP_HEATMAP: metric rank-flip rate per dataset …")
    f_rank_flip_heatmap.generate(
        df_singles_best, cfg.FIGURES_DIR, dataset_order=dataset_order
    )

    print("F_4METRIC_HEATMAP: 2x2 heatmap one panel per metric …")
    f_4metric_heatmap.generate(
        df_singles_best, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_DATASET_RANK_PROFILES: per-dataset models x metrics rank heatmap …")
    f_dataset_rank_profiles.generate(
        df_singles_best, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_BUMP_CHART: rank bump chart across metrics per dataset …")
    f_bump_chart.generate(
        df_singles_best, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

    print("F_METRIC_CONSISTENCY: metric-consistency heatmap …")
    f_metric_consistency.generate(
        df_singles_best, cfg.FIGURES_DIR,
        model_order=model_order, dataset_order=dataset_order
    )

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
        stage_tables(sel_agg=args.sel_agg)
    if args.all or args.figures:
        stage_figures(sel_agg=args.sel_agg)

    print("\nDone.")


if __name__ == "__main__":
    main()
