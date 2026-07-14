# Homogeneous Ensembles for Configurable Software Performance Prediction: A Multi-Metric Empirical Study

This repository contains the source code, the data used, and the raw
results for each experiment from the corresponding paper.

## Documents

### /data

This folder contains the data used for all experiments. There is one CSV
file for each subject system (eight in total). Each column represents a
configuration option, and the last column is the performance value to be
predicted.

### /results_paper
This folder does not contain the raw results directly. Due to their large size, separate download links are provided for each dataset in `results_paper/results_paper.txt`. Download and extract them here to reproduce the paper's original results. The folder/file structure is explained below.

Any new execution of `main.py` writes to `/results_new` instead (see `main.py`), so `/results_paper` is never overwritten. `main.py` creates `/results_new` automatically on first run.

The format of the results is as follows:

One folder per dataset, and within each dataset folder, one folder per
sample size (five per dataset).

In each sample size subfolder, there are eight files named
`{Model-Name}_metrics_results.json`. These files represent the results
of the single variants of the corresponding model type (eight model
types).

Each file has the following format:

A JSON file containing one list with ten dict entries. Each dict
represents one variant of the corresponding model with different
hyperparameters (visible in the `"Params"` entry). Each dict contains
the original `"Rank"` entry from the previous hyperparameter tuning, the
new `"Borda_Rank"` entry based on our evaluation method, and the mean
values across 30 runs for each evaluation metric (`Mean_MAE`,
`Mean_MRE`, `Mean_MBRE`, `Mean_MIBRE`). Additionally, `"Metrics"` and
`"Runs"` contain the evaluation metric values of each run and the raw
predictions of each run for each element in the corresponding test sets.
Each entry includes its SA comparison to assess whether the selected
learners perform meaningfully better than chance.

Our experiments include 27 different combinations of ensemble variants
for each learning technique: 3 combination rules (Mean, IRWM, and
NN) and 9 different numbers of single learners (from 2 to 10).
Similar to the single variants, we have:

-   `{Model-Name}_top{k}_predictions.json` for the Mean combination rule
-   `{Model-Name}_top{k}_irwm_predictions.json` for the IRWM combination
    rule
-   `{Model-Name}_top{k}_nn_predictions.json` for the NN combination
    rule

with `k` representing the number of single learners.

These files contain a single dict with the mean values across 30 runs
for each evaluation metric (`Mean_MAE`, `Mean_MRE`, `Mean_MBRE`,
`Mean_MIBRE`). `"Metrics"` and `"Runs"` contain the evaluation metric
values of each run and the raw predictions of each run for each element
in the corresponding test sets.

### /DEEPPERF and /HINNPERF

These directories contain the implementations of both state-of-the-art
approaches, using their publicly available resources:
https://github.com/DeepPerf/DeepPerf and https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh

### ML Techniques

Each ML technique has its own class based on the `Base` class in
`base.py`. It contains the general logic described in the paper for
constructing the single variants.

### main.py

This file contains the general logic described in the paper for
evaluating the single variants from the previous step and constructing
and evaluating the ensemble variants.

### /analysis_new

Analysis code (statistics, tables, figures) for the paper. It only reads
result JSON files already produced by `main.py` (it never re-runs any
model or experiment). See "How to Run the Analysis" below.

## Prerequisites and Installation

Use `requirements.txt` and follow any runtime messages. The experiments were executed with Python 3.9.x.

## How to Run

Comment out the datasets and models you want to exclude. Otherwise, it
will run all experiments by default, as described in the paper.

To reproduce the experiments from the paper:

    python main.py --model baseline
    python main.py

Note: Results from the next execution will be saved in `/results_new`.

## How to Run the Analysis

The code under `/analysis_new` only reads existing result JSON files (from
`/results_paper` or `/results_new`) and never re-runs any experiment.

1. (If needed) Install its dependencies (on top of `requirements.txt`):

       pip install -r analysis_new/requirements_analysis.txt

2. Set `RESULTS_DIR` in `analysis_new/config.py`:

       RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results_new"))

   Change `results_new` to `results_paper` (or any other results folder) to
   analyze a different set of results.

3. Run the pipeline from `analysis_new/`:

       cd analysis_new
       python run_all.py --all --sel-agg mean

   `--sel-agg mean` is the aggregation used for the tables/figures in the
   paper; it writes to `output_artifacts/latex_mean/` and
   `output_artifacts/figures_mean/` (the default `--sel-agg median` writes
   to `latex/`/`figures/`).

   Or run individual stages (each caches its output, so later stages reuse
   it): `--load`, `--aggregate`, `--tables`, `--figures`. Add `--force` to
   ignore the cache and recompute. Outputs are written to
   `analysis_new/output_artifacts/` (`cache/`, `latex/`, `figures/`).
