# Load all single-model metric result files.

import os
import json
import pandas as pd

_SKIP_METRICS = {"SA_5", "LSD"}

def load_singles(results_dir):
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
