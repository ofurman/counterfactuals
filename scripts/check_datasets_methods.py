"""
check_datasets_methods.py

Script to check which datasets are compatible with PPCEF and GLOBE-CE methods.
It attempts to initialize each dataset and method, reporting any issues.

Usage:
    uv run scripts/check_datasets_methods.py

Requirements:
    - All dependencies from the main project (see pyproject.toml)
    - Datasets and configs must be present in data/ and config/datasets/

"""
import os
import sys
import traceback
from pathlib import Path

import hydra
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from counterfactuals.datasets import FileDataset
from counterfactuals.cf_methods.local_methods.ppcef import PPCEF
from counterfactuals.cf_methods.global_methods.globe_ce import GLOBE_CE

# List all YAML configs in config/datasets
CONFIG_DIR = Path("config/datasets")
DATA_DIR = Path("data")


def find_yaml_configs():
    return sorted([f for f in CONFIG_DIR.glob("*.yaml")])


def check_dataset(config_path):
    try:
        ds = FileDataset(config_path=config_path)
        print(f"[OK] Loaded dataset: {config_path.name}")
        return ds
    except Exception as e:
        print(f"[FAIL] Could not load dataset {config_path.name}: {e}")
        traceback.print_exc()
        return None



import csv
import numpy as np

from counterfactuals.metrics.metrics import evaluate_cf

def check_ppcef(ds, config_name):
    try:
        # Split data into train/test
        try:
            X_train, X_test, y_train, y_test = ds.split_data(ds.X, ds.y, train_ratio=0.8, stratify=True)
        except ValueError:
             # Fallback for small datasets or single-class issues
            X_train, X_test, y_train, y_test = ds.split_data(ds.X, ds.y, train_ratio=0.8, stratify=False)

        import torch
        class DummyGen:
            def predict_log_prob(self, *a, **kw):
                return torch.zeros((len(X_test),))
            def __call__(self, X, y):
                return torch.zeros((len(X),))
            def to(self, device):
                return self
            def eval(self):
                return self
        class DummyDisc:
            def predict(self, X):
                return np.zeros((len(X),))
            def __call__(self, x):
                return x
            def to(self, device):
                return self
        dummy_gen = DummyGen()
        dummy_disc = DummyDisc()
        ppcef = PPCEF(gen_model=dummy_gen, disc_model=dummy_disc, disc_model_criterion=None)
        # Generate dummy counterfactuals (identity for reproducibility)
        Xs_cfs = X_test.copy()
        model_returned = np.ones(len(Xs_cfs), dtype=bool)
        metrics = evaluate_cf(
            gen_model=dummy_gen,
            disc_model=dummy_disc,
            X_cf=Xs_cfs,
            model_returned=model_returned,
            categorical_features=list(range(len(ds.categorical_features))),
            continuous_features=list(range(len(ds.config.continuous_features))),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            median_log_prob=0.0,
            y_target=y_test,
        )
        print(f"[OK] PPCEF metrics for {config_name}: {metrics}")
        return metrics
    except Exception as e:
        print(f"[FAIL] PPCEF failed for {config_name}: {e}")
        traceback.print_exc()
        return None

def check_globe_ce(ds, config_name):
    try:
        # Split data into train/test
        try:
            X_train, X_test, y_train, y_test = ds.split_data(ds.X, ds.y, train_ratio=0.8, stratify=True)
        except ValueError:
             # Fallback for small datasets or single-class issues
            X_train, X_test, y_train, y_test = ds.split_data(ds.X, ds.y, train_ratio=0.8, stratify=False)
        
        import torch
        def dummy_predict_fn(x):
            return np.zeros((len(x),))
        class DummyGen:
            def predict_log_prob(self, *a, **kw):
                return torch.zeros((len(X_test),))
            def __call__(self, X, y):
                return torch.zeros((len(X),))
            def to(self, device):
                return self
            def eval(self):
                return self
        class DummyDisc:
            def predict(self, X):
                return np.zeros((len(X),))
            def __call__(self, x):
                return x
            def to(self, device):
                return self
        dummy_gen = DummyGen()
        dummy_disc = DummyDisc()
        globe_ce = GLOBE_CE(predict_fn=dummy_predict_fn, dataset=ds, X=X_train, bin_widths=None, target_class=0)
        Xs_cfs = X_test.copy()
        model_returned = np.ones(len(Xs_cfs), dtype=bool)
        metrics = evaluate_cf(
            gen_model=dummy_gen,
            disc_model=dummy_disc,
            X_cf=Xs_cfs,
            model_returned=model_returned,
            categorical_features=list(range(len(ds.categorical_features))),
            continuous_features=list(range(len(ds.config.continuous_features))),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            median_log_prob=0.0,
            y_target=y_test,
        )
        print(f"[OK] GLOBE-CE metrics for {config_name}: {metrics}")
        return metrics
    except Exception as e:
        print(f"[FAIL] GLOBE-CE failed for {config_name}: {e}")
        traceback.print_exc()
        return None



def main():
    configs = find_yaml_configs()
    print(f"Found {len(configs)} dataset configs.")
    results = []
    for config_path in configs:
        ds = check_dataset(config_path)
        if ds is None:
            continue
        config_name = config_path.stem
        ppcef_metrics = check_ppcef(ds, config_name)
        globe_ce_metrics = check_globe_ce(ds, config_name)
        results.append({
            "dataset": config_name,
            "ppcef": ppcef_metrics,
            "globe_ce": globe_ce_metrics
        })

    # Write results to CSV and Markdown
    csv_path = Path("scripts/dataset_method_metrics.csv")
    md_path = Path("scripts/dataset_method_metrics.md")
    metric_keys = set()
    for r in results:
        if r["ppcef"]:
            metric_keys.update(r["ppcef"].keys())
        if r["globe_ce"]:
            metric_keys.update(r["globe_ce"].keys())
    metric_keys = sorted(metric_keys)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method"] + metric_keys)
        for r in results:
            for method in ["ppcef", "globe_ce"]:
                metrics = r[method]
                row = [r["dataset"], method.upper()]
                if metrics:
                    row += [metrics.get(k, "") for k in metric_keys]
                else:
                    row += ["ERROR"] * len(metric_keys)
                writer.writerow(row)

    with open(md_path, "w") as f:
        f.write("| dataset | method | " + " | ".join(metric_keys) + " |\n")
        f.write("|---|---|" + "|" * len(metric_keys) + "|\n")
        for r in results:
            for method in ["ppcef", "globe_ce"]:
                metrics = r[method]
                row = [r["dataset"], method.upper()]
                if metrics:
                    row += [str(metrics.get(k, "")) for k in metric_keys]
                else:
                    row += ["ERROR"] * len(metric_keys)
                f.write("| " + " | ".join(row) + " |\n")

if __name__ == "__main__":
    main()
