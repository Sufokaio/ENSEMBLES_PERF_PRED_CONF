# Load baseline metrics (random-guesser anchor).

import os
import json
import pandas as pd

def load_baseline(results_dir):
    rows = []
    for ds_entry in _scan_dirs(results_dir):
        dataset = ds_entry.name
        for sz_entry in _scan_dirs(ds_entry.path):
            try:
                sample_size = int(sz_entry.name)
            except ValueError:
                continue
            fpath = os.path.join(sz_entry.path, "baseline_metrics_results.json")
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                d = json.load(f)
            rows.append({
                "dataset":     dataset,
                "sample_size": sample_size,
                "MAEp0":       float(d["MAEp0"]),
                "Sp0":         float(d["Sp0"]),
                "Q5p0":        float(d["Q5p0"]),
                "SA_5":        float(d["SA_5"]),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sample_size"] = df["sample_size"].astype(int)
    return df

def _scan_dirs(path):
    return [e for e in os.scandir(path) if e.is_dir()]
