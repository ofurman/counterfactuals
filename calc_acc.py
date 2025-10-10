from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = [
    "LendingClubDataset",
    "GiveMeSomeCreditDataset",
    "BankMarketingDataset",
    "CreditDefaultDataset",
    "AdultCensusDataset",
]
METHOD = "PPCEF"
MODELS = ["MultilayerPerceptron", "MultinomialLogisticRegression", "NODE"]

root_path = Path("models")
for model in MODELS:
    for dataset in DATASETS:
        dataset_path = root_path / dataset
        method_path = dataset_path / METHOD
        accs = []
        for fold in range(5):
            fold_path = method_path / f"fold_{fold}"
            metrics_path = fold_path / f"eval_disc_model_{model}.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                acc = metrics.iloc[2].to_list()[-1]
                accs.append(acc)
            else:
                raise ValueError(f"Metrics file {metrics_path} does not exist")

        print(dataset, model, round(np.mean(accs), 2))
