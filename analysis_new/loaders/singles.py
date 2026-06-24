"""
Load all single-model metric result files.

File layout: {results_dir}/{dataset}/{sample_size}/{model_type}_metrics_results.json

Each JSON is an array of hyperparameter-config objects, each with:
  Rank, Params, Metrics -> {MRE, MAE, MBRE, MIBRE, SA, D, SA_5, LSD} (30 values per run)

SA_5 and LSD are skipped here; SA_5 is loaded via baseline.py.
"""
import os
import json
import pandas as pd

_SKIP_METRICS = {"SA_5", "LSD"}


def load_singles(results_dir):
    """
    Returns long-format DataFrame:
      dataset | sample_size | model_type | config_id | run | metric | value

    Metrics included: MRE, MAE, MBRE, MIBRE, SA, D
    config_id = 0-based index into the JSON array (hyperparameter config index).
    run = 0-based index into the 30-run array.
    """
    rows = []
    for ds_entry in _scan_dirs(results_dir):
        dataset = ds_entry.name
        for sz_entry in _scan_dirs(ds_entry.path):
            try:
                sample_size = int(sz_entry.name)
            except ValueError:
                continue
            for fname in sorted(os.listdir(sz_entry.path)):
                if fname == "baseline_metrics_results.json":
                    continue
                if not fname.endswith("_metrics_results.json"):
                    continue
                model_type = fname[: -len("_metrics_results.json")]
                fpath = os.path.join(sz_entry.path, fname)
                with open(fpath) as f:
                    configs = json.load(f)
                for config_id, cfg in enumerate(configs):
                    for metric, vals in cfg["Metrics"].items():
                        if metric in _SKIP_METRICS:
                            continue
                        if not isinstance(vals, list):
                            continue
                        for run_idx, val in enumerate(vals):
                            rows.append({
                                "dataset":     dataset,
                                "sample_size": sample_size,
                                "model_type":  model_type,
                                "config_id":   config_id,
                                "run":         run_idx,
                                "metric":      metric,
                                "value":       float(val),
                            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sample_size"] = df["sample_size"].astype(int)
    return df


def _scan_dirs(path):
    return [e for e in os.scandir(path) if e.is_dir()]
