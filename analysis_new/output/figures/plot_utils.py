# Shared matplotlib utilities for all figure emitters.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

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

RULE_COLORS = {
    "MEAN": "#2166ac",
    "IRWM": "#4dac26",
    "NN":   "#d01c8b",
}

RULE_MARKERS = {"MEAN": "o", "IRWM": "s", "NN": "^"}

DS_DISPLAY = {
    "apache": "APACHE", "bdbc": "BDBC", "dune": "DUNE",
    "hipacc": "HIPACC", "hsmgp": "HSMGP", "kanzi": "KANZI",
    "lrzip": "LRZIP", "x264": "X264",
}

def ds_label(name):
    return DS_DISPLAY.get(name.lower(), name.upper())

def get_model_color(name):
    return MODEL_COLORS.get(name, "#333333")

def save_figure(fig, path, fmt="pdf"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format=fmt)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
