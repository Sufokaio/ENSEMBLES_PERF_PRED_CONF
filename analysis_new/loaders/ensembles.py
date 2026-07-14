# Load ensemble prediction metric files.

import os
import json
import pandas as pd

_METRICS_ENS = ["MRE", "MAE", "MBRE", "MIBRE"]

_SUFFIX_TO_RULE = {"": "MEAN", "_irwm": "IRWM", "_nn": "NN"}

def load_ensembles(results_dir):
    rows = []
    for ds_entry in _scan_dirs(results_dir):
        dataset = ds_entry.name
        for sz_entry in _scan_dirs(ds_entry.path):
            try:
                sample_size = int(sz_entry.name)
            except ValueError:
                continue
            for fname in sorted(os.listdir(sz_entry.path)):
                if not fname.endswith("_predictions.json"):
                    continue
                parsed = _parse_filename(fname)
                if parsed is None:
                    continue
                base_type, k, rule = parsed
                fpath = os.path.join(sz_entry.path, fname)
                with open(fpath) as f:
                    obj = json.load(f)
                metrics = obj.get("Metrics", {})
                for metric in _METRICS_ENS:
                    vals = metrics.get(metric)
                    if vals is None:
                        continue
                    for run_idx, val in enumerate(vals):
                        rows.append({
                            "dataset":     dataset,
                            "sample_size": sample_size,
                            "base_type":   base_type,
                            "k":           k,
                            "rule":        rule,
                            "run":         run_idx,
                            "metric":      metric,
                            "value":       float(val),
                        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sample_size"] = df["sample_size"].astype(int)
        df["k"] = df["k"].astype(int)
    return df

def _parse_filename(fname):
    stem = fname[: -len("_predictions.json")]
    if "_top" not in stem:
        return None
    base_type, rest = stem.split("_top", 1)
    parts = rest.split("_", 1)
    try:
        k = int(parts[0])
    except ValueError:
        return None
    rule_suffix = ("_" + parts[1]) if len(parts) > 1 else ""
    rule = _SUFFIX_TO_RULE.get(rule_suffix)
    if rule is None:
        return None
    return base_type, k, rule

def _scan_dirs(path):
    return [e for e in os.scandir(path) if e.is_dir()]
