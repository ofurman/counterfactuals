import hydra
import os
import mlflow
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import neptune
from uuid import uuid4
from joblib import dump, load
from hydra.utils import instantiate
from tqdm import tqdm

from omegaconf import DictConfig
from sklearn.metrics import classification_report
from alibi.explainers import CEM

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.utils import process_classification_report


@hydra.main(config_path="../conf/other_methods", config_name="config_cem", version_base="1.2")
def main(cfg: DictConfig):
    tf.compat.v1.disable_eager_execution()

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
    run["parameters/gen_model"] = "WACH"

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

    X_test = X_test[:20]
    y_test = y_test[:20]

    mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
    shape = (1,) + X_train.shape[1:]  # instance shape
    kappa = .2  # minimum difference needed between the prediction probability for the perturbed instance on the
    # class predicted by the original instance and the max probability on the other classes
    # in order for the first loss term to be minimized
    beta = .1  # weight of the L1 loss term
    c_init = 10.  # initial weight c of the loss term encouraging to predict a different class (PN) or
    # the same class (PP) for the perturbed instance compared to the original instance to be explained
    # DEFAULT c_steps = 10  # nb of updates for c
    c_steps = 5
    # DEFAULT max_iterations = 1000  # nb of iterations per value of c
    max_iterations = 200
    clip = (-1000., 1000.)  # gradient clipping
    lr_init = 1e-2  # initial learning rate
    feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                     X_train.max(axis=0).reshape(shape))

    cf = CEM(disc_model.predict_proba, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
                  max_iterations=max_iterations, c_init=c_init, c_steps=c_steps,
                  learning_rate_init=lr_init, clip=clip)
    cf.fit(X_train, no_info_type='median')


    # X_test = X_test[:20]
    # y_test = y_test[:20]
    Xs_cfs = []
    start_time = time()
    for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        explanation = cf.explain(X.reshape(1, -1), verbose=False)
        Xs_cfs.append(explanation.PN if explanation.PN is not None else X)
    run["metrics/avg_time_one_cf"] = (time() - start_time) / X_test.shape[0]

    Xs_cfs = np.array(Xs_cfs).squeeze()
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    metrics = evaluate_cf(
        disc_model=disc_model,
        X=X_test,
        X_cf=Xs_cfs,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=X_train,
        y_train=y_train.reshape(-1),
        X_test=X_test,
        y_test=y_test.reshape(-1),
    )
    run["metrics/cf"] = metrics

    run.stop()

if __name__ == "__main__":
    main()

