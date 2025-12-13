import argparse
import os
import pickle
import json

import numpy as np
from deepperf import DeepPerf
from rHINNPERF import HINNPerf

from KNN import KNN
from RT import RT
from KRR import KRR
from LR import LR
from RF import RF

from SVRWrapper import SVRWrapper as SVR
from baseline import Baseline

import time
import copy
import tensorflow as tf

import time
import json
import argparse
import numpy as np



import multiprocessing

import os

cpu_count = os.cpu_count()




MODEL_REGISTRY = {
    "baseline": Baseline,

    "LR": LR,
    "SVR": SVR,
    "KNN": KNN,
    "RF": RF,
    "RT": RT,
    "KRR": KRR,

    "DeepPerf": DeepPerf,
    "HINNPerf": HINNPerf


 }
 # Dataset name mapped to its corresponding sample sizes
DATASET_REGISTRY = {
    "apache": np.array( [9,18,27,36,45]),
    "bdbc": np.array([18,36,54,72,90]),
    "kanzi": np.array([31,62,93,124,155]),
    "x264": np.array([16,32,48,64,80]),


    "lrzip": np.array([127,295,386,485,907]),
    "dune": np.array([224,692,1000,1365,1612]),
    "hipacc": np.array([261,528,736,1281,2631]),
    "hsmgp": np.array([77,173,384,480,864]),

}

# Evaluate and rank top-10 models based on multiple metrics
# Assignment of Borda scores and ranks for ensemble construction
def eval_top10(metrics_results):
    for entry in metrics_results:
        mean_sa = np.mean(entry["Metrics"]["SA"])
        mean_sa_5 = np.mean(entry["Metrics"]["SA_5"])
        entry["Mean_SA"] = mean_sa
        entry["SA_vs_SA_5"] = f"SA: {mean_sa:.4f} | SA_5: {mean_sa_5:.4f}"

    metrics_results.sort(key=lambda x: x["Mean_SA"], reverse=True)

    for entry in metrics_results:
        entry["Mean_MAE"] = np.mean(entry["Metrics"]["MAE"])
        entry["Mean_MRE"] = np.mean(entry["Metrics"]["MRE"])
        entry["Mean_MBRE"] = np.mean(entry["Metrics"]["MBRE"])
        entry["Mean_MIBRE"] = np.mean(entry["Metrics"]["MIBRE"])

    def rank_entries(metric_name):
        sorted_entries = sorted(metrics_results, key=lambda x: x[f"Mean_{metric_name}"])
        for rank, entry in enumerate(sorted_entries, start=1):
            entry[f"Rank_{metric_name}"] = rank

    for metric in ["MAE", "MRE", "MBRE", "MIBRE"]:
        rank_entries(metric)

    for entry in metrics_results:
        entry["Borda_Score"] = (
            entry["Rank_MAE"] +
            entry["Rank_MRE"] +
            entry["Rank_MBRE"] +
            entry["Rank_MIBRE"]
        )

    sorted_by_borda = sorted(metrics_results, key=lambda x: (x["Borda_Score"], x["Mean_MAE"]))

    for rank, entry in enumerate(sorted_by_borda, start=1):
        entry["Borda_Rank"] = rank

    for i, entry in enumerate(metrics_results):
        print(
            f"Entry {i}: "
            f"MAE={entry['Mean_MAE']:.4f} (Rank {entry['Rank_MAE']}), "
            f"MRE={entry['Mean_MRE']:.4f} (Rank {entry['Rank_MRE']}), "
            f"MBRE={entry['Mean_MBRE']:.4f} (Rank {entry['Rank_MBRE']}), "
            f"MIBRE={entry['Mean_MIBRE']:.4f} (Rank {entry['Rank_MIBRE']}), "
            f"Borda Score={entry['Borda_Score']}, Borda Rank={entry['Borda_Rank']}"
        )
    return sorted_by_borda


# Mean Combination Rule for top-n models
def aggregate_top_n_results(all_topn_runs, n):
    n = min(n, len(all_topn_runs))
    k = len(all_topn_runs[0]["results"])

    results = []
    for run_idx in range(k):
        np.random.seed(25 + run_idx)

        y_preds = [np.array(all_topn_runs[param_idx]["results"][run_idx]["y_pred"]) for param_idx in range(n)]

        y_test = all_topn_runs[0]["results"][run_idx]["y_test"]
        # X_test = all_topn_runs[0]["results"][run_idx]["X_test"]

        # Aggregate y_pred by mean across n param sets (element-wise)
        y_pred_mean = np.mean(y_preds, axis=0).tolist()

        # Calculate MAE and MRE
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred_mean)
        mae = np.mean(np.abs(y_test_arr - y_pred_arr))
        mre = np.mean(np.abs(y_test_arr - y_pred_arr) / (np.abs(y_test_arr))) * 100

        results.append({
            "y_test": y_test,
            "y_pred": y_pred_mean,
            "mae": mae,
            "mre": mre
        })

    # Compute mean MAE and MRE across all runs
    mean_mae = np.mean([r["mae"] for r in results])
    mean_mre = np.mean([r["mre"] for r in results])
    median_mae = np.median([r["mae"] for r in results])
    median_mre = np.median([r["mre"] for r in results])

    print(f"Aggregated Top-{n} Mean MAE: {mean_mae:.4f}, Mean MRE: {mean_mre:.4f}")

    return results, mean_mae, mean_mre, median_mae, median_mre

# NN Combination Rule for top-n models
def aggregate_top_n_nn_results(
    all_topn_runs, 
    n, 
    epochs=600, 
    activation='relu', 
    loss='mae', 
    early_stopping=True, 
    patience=20
):
    """
    Trains a 1-layer NN to combine top-n predictions for each test set.
    """
    n = min(n, len(all_topn_runs))
    k = len(all_topn_runs[0]["results"])

    results = []
    for run_idx in range(k):
        tf.keras.backend.clear_session()
        tf.random.set_seed(25 + run_idx)

        
        # Prepare training features (top-n y_train_pred) and targets (y_train)
        y_train_preds = [np.array(all_topn_runs[param_idx]["results"][run_idx]["y_train_pred"]) for param_idx in range(n)]
        y_train_preds = np.stack(y_train_preds, axis=1)  # shape: (num_train_samples, n)
        y_train = np.array(all_topn_runs[0]["results"][run_idx]["y_train"])

        # Prepare test features (top-n y_pred) and targets (y_test)
        y_preds = [np.array(all_topn_runs[param_idx]["results"][run_idx]["y_pred"]) for param_idx in range(n)]
        y_preds = np.stack(y_preds, axis=1)  # shape: (num_test_samples, n)
        y_test = np.array(all_topn_runs[0]["results"][run_idx]["y_test"])

        # Manual normalization (z-score)
        mean = np.mean(y_train_preds, axis=0)
        std = np.std(y_train_preds, axis=0) + 1e-8 

        y_train_preds_norm = (y_train_preds - mean) / std
        y_preds_norm = (y_preds - mean) / std

        # Build 1-layer NN model (no normalization layer)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n,)),
            tf.keras.layers.Dense(1, activation=activation)
        ])
        model.compile(optimizer='adam', loss=loss)

        # Early stopping callback
        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=patience, restore_best_weights=True
            ))

        # Train NN
        model.fit(y_train_preds_norm, y_train, epochs=epochs, verbose=0, callbacks=callbacks, shuffle=False)

        # Predict
        y_pred_nn = model.predict(y_preds_norm, verbose=0).flatten()

        mae = np.mean(np.abs(y_test - y_pred_nn))
        mre = np.mean(np.abs(y_test - y_pred_nn) / (np.abs(y_test))) * 100

        results.append({
            "y_test": y_test.tolist(),
            "y_pred": y_pred_nn.tolist(),
            "mae": mae,
            "mre": mre
        })

    mean_mae = np.mean([r["mae"] for r in results])
    mean_mre = np.mean([r["mre"] for r in results])
    median_mae = np.median([r["mae"] for r in results])
    median_mre = np.median([r["mre"] for r in results])

    print(f"NN Aggregated Top-{n} Mean MAE: {mean_mae:.4f}, Mean MRE: {mean_mre:.4f}")

    return results, mean_mae, mean_mre,     median_mae, median_mre

# IRWM Combination Rule for top-n models
def aggregate_top_n_irwm_results(all_topn_runs, n):
    """
    Combines predictions using Inverse Ranked Weighted Mean (IRWM).
    Models are ranked using Borda Rank.
    Lower rank/score = higher weight.
    """
    n = min(n, len(all_topn_runs))
    k = len(all_topn_runs[0]["results"])

    # Prefer Borda Rank if available, else mean_mae
    if all("borda_rank" in run and run["borda_rank"] is not None for run in all_topn_runs[:n]):
        # Use Borda Rank for ranking (lower is better)
        model_ranks = [run["borda_rank"] for run in all_topn_runs[:n]]
        print(f"Using Borda Rank for IRWM: {model_ranks}")
    else:
        # Fallback to mean_mae (lower is better)
        model_ranks = np.argsort([run["mean_mae"] for run in all_topn_runs[:n]]) + 1
        print(f"Using mean MAE rank for IRWM: {model_ranks}")

    model_ranks = np.array(model_ranks, dtype=float)

    # Inverse rank weights
    weights = 1.0 / model_ranks
    weights /= weights.sum()  # Normalize

    results = []
    for run_idx in range(k):
        np.random.seed(25 + run_idx)
        y_preds = [np.array(all_topn_runs[i]["results"][run_idx]["y_pred"]) for i in range(n)]
        y_test = all_topn_runs[0]["results"][run_idx]["y_test"]

        y_pred_irwm = np.average(y_preds, axis=0, weights=weights).tolist()

        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred_irwm)
        mae = np.mean(np.abs(y_test_arr - y_pred_arr))
        mre = np.mean(np.abs(y_test_arr - y_pred_arr) / (np.abs(y_test_arr))) * 100

        results.append({
            "y_test": y_test,
            "y_pred": y_pred_irwm,
            "mae": mae,
            "mre": mre
        })

    mean_mae = np.mean([r["mae"] for r in results])
    mean_mre = np.mean([r["mre"] for r in results])
    median_mae = np.median([r["mae"] for r in results])
    median_mre = np.median([r["mre"] for r in results])

    print(f"IRWM Aggregated Top-{n} Mean MAE: {mean_mae:.4f}, Mean MRE: {mean_mre:.4f}")

    return results, mean_mae, mean_mre, median_mae, median_mre


# Metric computations for ensemble results
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mre = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    mbre = np.mean(np.abs(y_true - y_pred) / (np.minimum(y_true, y_pred) + 1e-8))
    mibre = np.mean(np.abs(y_true - y_pred) / (np.maximum(y_true, y_pred) + 1e-8))
    return {
        "MAE": mae,
        "MRE": mre,
        "MBRE": mbre,
        "MIBRE": mibre
    }

def aggregate_ensemble_metrics(ensemble_results):
    metrics = {"MAE": [], "MRE": [], "MBRE": [], "MIBRE": []}
    runs = []
    for run in ensemble_results:
        m = compute_metrics(run["y_test"], run["y_pred"])
        for k in metrics:
            metrics[k].append(m[k])
        runs.append({
            "y_test": run["y_test"],
            "y_pred": run["y_pred"]
        })
    # Compute means and medians for each metric
    mean_metrics = {f"Mean_{k}": float(np.mean(metrics[k])) for k in metrics}
    median_metrics = {f"Median_{k}": float(np.median(metrics[k])) for k in metrics}
    # Return in the same format as single results
    return {
        "Metrics": metrics,
        **mean_metrics,
        **median_metrics,
        "Runs": runs
    }

def log_time(message):
    with open("times.txt", "a") as f:
        f.write(message + "\n")

# Run experiment task for multiprocessing
def run_experiment_task(model_name, dataset, sample_index, s):
    print(f"Running model '{model_name}' on dataset '{dataset} with sample size {sample_index}'")
    evaluator = MODEL_REGISTRY[model_name](dataset, split_mode=False, split=s, num_runs=30)
    print(f"PID {os.getpid()} is running {model_name} on {dataset} sample {s}")


    results_dir = f"results/{dataset}/{s}"
    os.makedirs(results_dir, exist_ok=True)
    metrics_results_path = os.path.join(results_dir, f"{model_name}_metrics_results.json")


    if os.path.exists(metrics_results_path):
        print(f"Loading metrics_results from {metrics_results_path}")
        with open(metrics_results_path, "r") as f:
            metrics_results = json.load(f)
    else:
        t0 = time.time()

        metrics_results = evaluator.run_experiment()
        elapsed = time.time() - t0
        log_time(f"Completed {model_name} on {dataset} sample {s} in {elapsed:.2f} seconds")


        for entry in metrics_results:
            if "Metrics" in entry:
                for key, value in entry["Metrics"].items():
                    if isinstance(value, np.ndarray):
                        entry["Metrics"][key] = value.tolist()
        # Evaluate and enrich metrics with borda ranks
        if not model_name == "baseline":
            metrics_results = eval_top10(metrics_results)
        with open(metrics_results_path, "w") as f:
            json.dump(metrics_results, f, indent=2)

    if not model_name == "baseline":
        # Prepare all_top10_runs for ensemble aggregation
        all_top10_runs = []
        for entry in metrics_results:
            all_top10_runs.append({
                "results": entry["Runs"],
                "borda_rank": entry.get("Borda_Rank", None),
                "borda_score": entry.get("Borda_Score", None),
                "mean_mae": entry.get("Mean_MAE", None)
            })

        if len(all_top10_runs) >= 2:
            top_n_list = [2,3,4,5,6,7,8,9,10]


            for top_n_g in top_n_list:
                # Mean aggregation
                mean_path = os.path.join(results_dir, f"{model_name}_top{top_n_g}_predictions.json")
                if not os.path.exists(mean_path):
                    t0 = time.time()
                    agg_results, *_ = aggregate_top_n_results(all_top10_runs, n=top_n_g)
                    elapsed = time.time() - t0
                    log_time(f"Completed {model_name} Mean (Top-{top_n_g}) on {dataset} sample {s} in {elapsed:.2f} seconds")

                    agg_metrics = aggregate_ensemble_metrics(agg_results)
                    with open(mean_path, "w") as f:
                        json.dump(agg_metrics, f, indent=2)
                else:
                    print(f"Skipping Mean Top-{top_n_g}, file exists: {mean_path}")

                # NN aggregation
                nn_path = os.path.join(results_dir, f"{model_name}_top{top_n_g}_nn_predictions.json")
                if not os.path.exists(nn_path):
                    t0 = time.time()
                    nn_results, *_ = aggregate_top_n_nn_results(
                        all_top10_runs, n=top_n_g, epochs=600, activation='linear', loss='mae', early_stopping=True, patience=20
                    )
                    elapsed = time.time() - t0
                    log_time(f"Completed {model_name} NN (Top-{top_n_g}) on {dataset} sample {s} in {elapsed:.2f} seconds")

                    nn_metrics = aggregate_ensemble_metrics(nn_results)
                    with open(nn_path, "w") as f:
                        json.dump(nn_metrics, f, indent=2)
                else:
                    print(f"Skipping NN Top-{top_n_g}, file exists: {nn_path}")

                # IRWM aggregation
                irwm_path = os.path.join(results_dir, f"{model_name}_top{top_n_g}_irwm_predictions.json")
                if not os.path.exists(irwm_path):
                    t0 = time.time()
                    irwm_results, *_ = aggregate_top_n_irwm_results(all_top10_runs, n=top_n_g)
                    elapsed = time.time() - t0
                    log_time(f"Completed {model_name} IRWM (Top-{top_n_g}) on {dataset} sample {s} in {elapsed:.2f} seconds")

                    irwm_metrics = aggregate_ensemble_metrics(irwm_results)
                    with open(irwm_path, "w") as f:
                        json.dump(irwm_metrics, f, indent=2)
                else:
                    print(f"Skipping IRWM Top-{top_n_g}, file exists: {irwm_path}")

    return f"Completed {model_name} on {dataset} sample {s}"


def main():
    parser = argparse.ArgumentParser(description='Starting ...')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name')
    parser.add_argument('--model', type=str, default="all", help='Model name')
    parser.add_argument('--mode', type=str, default="normal", help='Execution mode')
    parser.add_argument('--num_runs', type=int, default=30 , help='Number of experiments')

    args = parser.parse_args()
    

    if args.dataset == "all":
        datasets = DATASET_REGISTRY
    else:
        datasets = [args.dataset]

    if args.model == "all":
        models = [m for m in MODEL_REGISTRY.keys() if m != "baseline"]
    else:
        models = [args.model]

    if args.mode == "normal":
        task_args = []
        for model_name in models:
            for dataset, sample_sizes in datasets.items():
                for i, s in enumerate(sample_sizes):
                    task_args.append((model_name, dataset, i, s))
        with multiprocessing.Pool(processes=cpu_count - 3) as pool:

            results = pool.starmap(run_experiment_task, task_args)
        end_time = time.time()
        for res in results:
            print(res)
        print(f"All experiments completed at {end_time}")


if __name__ == "__main__":
    main()
