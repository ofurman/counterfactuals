import logging
import os
from time import time
from uuid import uuid4

import hydra
import mlflow
import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.utils import instantiate
from joblib import dump, load
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from counterfactuals.sace.casebased_sace import CaseBasedSACE
from counterfactuals.sace.blackbox import BlackBox
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.utils import process_classification_report

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf/other_methods", config_name="config_cbce", version_base="1.2")
def main(cfg: DictConfig):
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Log parameters using Hydra config
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model"] = "Artelt"

    available_disc_models = ["LogisticRegression", "DecisionTreeClassifier"]
    if cfg.disc_model._target_.split(".")[-1] not in available_disc_models:
        raise ValueError(f"Disc model not supported. Please choose one of {available_disc_models}")
    use_decision_tree = cfg.disc_model._target_.split(".")[-1] == "DecisionTreeClassifier"

    dataset = instantiate(cfg.dataset)
    X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test
    

    disc_model = instantiate(cfg.disc_model)
    disc_model.fit(dataset.X_train, dataset.y_train.reshape(-1))
    report = classification_report(dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True)

    mlflow.log_metrics(process_classification_report(report, prefix="disc_test"))
    run["metrics"] = process_classification_report(report, prefix="disc_test")

    disc_model_path = os.path.join(output_folder, f"disc_model_{uuid4()}.joblib")
    dump(disc_model, disc_model_path)
    run["disc_model"].upload(disc_model_path)

    X_test_pred_path = os.path.join(output_folder, "X_test_pred.csv")
    pd.DataFrame(disc_model.predict(dataset.X_test)).to_csv(X_test_pred_path, index=False)
    run["X_test_pred"].upload(X_test_pred_path)

    y_train = disc_model.predict(X_train)
    y_test = disc_model.predict(X_test)

    # Start CBCE Method
    variable_features = dataset.numerical_features + dataset.categorical_features
    metric = ('euclidean', 'jaccard')
        #         ('cosine', 'jaccard'),
        #         # ('euclidean', 'hamming')
    cf = CaseBasedSACE(
        variable_features=variable_features,
        weights=None,
        metric=metric,
        feature_names=None,
        continuous_features=dataset.numerical_features,
        categorical_features_lists=dataset.categorical_features_lists,
        normalize=False,
        random_samples=None,
        diff_features=5,
        tolerance=0.001,
    )
    bb = BlackBox(disc_model)
    cf.fit(bb, X_train)
    cf_time = time()

    Xs_cfs = []
    model_returned = []
    for x in tqdm(X_test):
        x_cf = cf.get_counterfactuals(x, k=1)
        Xs_cfs.append(x_cf)
        model_returned.append(True)

    run["metrics/avg_time_one_cf"] = time() - cf_time / len(X_test)

    Xs_cfs = np.array(Xs_cfs).squeeze()
    print(Xs_cfs.shape)
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    metrics = evaluate_cf(
        disc_model=disc_model,
        X=X_test,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print(metrics)
    run["metrics/cf"] = metrics

    run.stop()


if __name__ == "__main__":
    main()
