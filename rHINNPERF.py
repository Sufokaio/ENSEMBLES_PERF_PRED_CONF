import os
import pprint
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from base import Base
from HINNPERF.runHINNPerf import get_HINNPerf_MRE_and_predictions, get_HINNPerf_MRE


class HINNPerf(Base):
    def __init__(self, dataset_name, data_path="data", split=0.8, split_mode=True, num_runs=30):
        super().__init__({}, "HINNPerf", dataset_name, data_path, split, split_mode, num_runs)
        self.init_seed = 25
        np.random.seed(self.init_seed)
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.file_path)
        self.whole_data = np.genfromtxt(self.file_path, delimiter=',', skip_header=1)
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def get_top_10(self):
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        config_errors = {}

        for run in tqdm(range(self.num_runs)):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            training_index = indices[:num_train]
            testing_index = indices[num_train:num_train + num_test]

            # Get all configs and their errors for this run
            _, all_configs = get_HINNPerf_MRE_and_predictions([self.whole_data, training_index, testing_index, False, []])
            for config, error in all_configs:
                config_key = str(config)
                if config_key not in config_errors:
                    config_errors[config_key] = {"config": config, "errors": []}
                config_errors[config_key]["errors"].append(error)

        # Average errors for each config
        averaged_configs = []
        for entry in config_errors.values():
            avg_error = sum(entry["errors"]) / len(entry["errors"])
            averaged_configs.append((entry["config"], avg_error))

        top_10_configs = sorted(averaged_configs, key=lambda x: x[1])[:10]

        return top_10_configs

    def run_experiment(self):
        top_10 = self.get_top_10()
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train

        metrics_results = []

        for rank, (config, _) in enumerate(top_10, 1):
            param_metrics = {
                "Rank": rank,
                "Params": config,
                "Metrics": {
                    "MRE": [],
                    #"MRE2": [],
                    "MAE": [],
                    "SA": [],
                    "SA_5": [],
                    "D": [],
                    "MBRE": [],
                    "MIBRE": [],
                    "LSD": []
                },
                "Runs": []  #  all predictions
            }

            for run in range(self.num_runs):
                np.random.seed(self.init_seed + run)
                indices = np.random.permutation(len(self.y))
                training_index = indices[:num_train]
                testing_index = indices[num_train:num_train + num_test]

                config_with_lists = {k: [v] for k, v in config.items()}
                rel_error, y_pred, y_test, X_test, y_train_pred = get_HINNPerf_MRE([self.whole_data, training_index, testing_index, False, config_with_lists ])

                mre = rel_error
                mre2 = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                mae = mean_absolute_error(y_test, y_pred)
                mae_p0, Sp0, Sa_5 = self.load_baseline()
                sa = self.standardized_accuracy(mae, mae_p0)
                d = self.effect_size_test(mae, mae_p0, Sp0)
                mibre, mbre, lsd = self.compute_metrics(y_test, y_pred)

                param_metrics["Metrics"]["MRE"].append(mre)
                param_metrics["Metrics"]["MAE"].append(mae)
                param_metrics["Metrics"]["SA"].append(sa)
                param_metrics["Metrics"]["SA_5"].append(Sa_5)
                param_metrics["Metrics"]["D"].append(d)
                param_metrics["Metrics"]["MBRE"].append(mbre)
                param_metrics["Metrics"]["MIBRE"].append(mibre)
                param_metrics["Metrics"]["LSD"].append(lsd)

                # Store run data for aggregation
                param_metrics["Runs"].append({
                    # "X_test": X_test.tolist(),
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    # "mae": mae,
                    # "X_train": self.X[training_index].tolist(),
                    "y_train": self.y[training_index].tolist(),
                    "y_train_pred": y_train_pred.tolist()
                })

            metrics_results.append(param_metrics)

        return metrics_results



    def _run_single_config(self, config, num_train, num_test):
        """
        Runs a single configuration for all runs.
        """
        results = []
        for run in range(self.num_runs):
            np.random.seed(self.init_seed + run)
            indices = np.random.permutation(len(self.y))
            training_index = indices[:num_train]
            testing_index = indices[num_train:num_train + num_test]

            X_train, y_train = self.X[training_index], self.y[training_index]


            config_with_lists = {k: [v] for k, v in config.items()}
            _, y_pred, y_test, X_test,y_train_pred = get_HINNPerf_MRE([self.whole_data, training_index, testing_index, False, config_with_lists])

            mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            mae = mean_absolute_error(y_test, y_pred)
            results.append((X_test, y_test, y_pred, mae, mre, X_train, y_train, y_train_pred))
        return results

    def run_single_config_experiment(self, config):
        """
        Runs experiment for a single config set.
        """
        num_train = round(len(self.y) * self.split) if self.split_mode else self.split
        num_test = len(self.y) - num_train
        return self._run_single_config(config, num_train, num_test)
