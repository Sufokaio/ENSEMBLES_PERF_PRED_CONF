# Shared helpers for table and figure emitters.

import os
import numpy as np

def bold(s):
    return r"\textbf{" + str(s) + r"}"

def save_tex(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(path)}")

def fmt_cell(central, spread, sk_rank=None, fmt=".2f"):
    cell = f"{central:{fmt}} ({spread:{fmt}})"
    if sk_rank is not None:
        cell += f"$_{{{sk_rank}}}$"
    return cell

def mean_std(values):
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1))

def median_iqr(values):
    a = np.asarray(values, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))
