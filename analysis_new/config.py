import os

_HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results_new"))
OUTPUT_DIR  = os.path.join(_HERE, "output_artifacts")
CACHE_DIR   = os.path.join(OUTPUT_DIR, "cache")
LATEX_DIR   = os.path.join(OUTPUT_DIR, "latex")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

MODEL_TYPES = ["LR", "SVR", "RT", "RF", "KNN", "KRR", "DeepPerf", "HINNPerf"]
METRICS     = ["MRE", "MAE", "MBRE", "MIBRE"]
RULES       = ["MEAN", "IRWM", "NN"]
K_RANGE     = list(range(2, 11))

# File suffix for each combination rule in ensemble filenames
RULE_FILE_SUFFIX = {"MEAN": "", "IRWM": "_irwm", "NN": "_nn"}

MODEL_COLORS = {
    "LR":       "#1f77b4",
    "SVR":      "#ff7f0e",
    "RT":       "#2ca02c",
    "RF":       "#d62728",
    "KNN":      "#9467bd",
    "KRR":      "#8c564b",
    "DeepPerf": "#e377c2",
    "HINNPerf": "#7f7f7f",
}

# Nemenyi q_{alpha=0.05} critical values (Demsar 2006, Table 5) keyed by # of classifiers
NEMENYI_Q = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}
