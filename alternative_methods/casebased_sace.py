import logging
import os
from time import time
from uuid import uuid4

import hydra
import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from hydra.utils import instantiate
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from tqdm import tqdm

from counterfactuals.discriminative_models import LogisticRegression, MultilayerPerceptron
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.cf_methods.ppcef import PPCEF
from counterfactuals.cf_methods.sace.blackbox import BlackBox
from counterfactuals.cf_methods.sace.casebased_sace import CaseBasedSACE
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
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model"] = cfg.disc_model.model
    run["parameters/gen_model"] = cfg.gen_model.model
    run["parameters/reference_method"] = "CBCE"
    # run["parameters/pca_dim"] = cfg.pca_dim
    run.wait()

    models_folder = cfg.experiment.models_folder

    available_disc_models = ["LR"]
    if cfg.disc_model.model not in available_disc_models:
        raise ValueError(f"Disc model not supported. Please choose one of {available_disc_models}")

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    logger.info("Loading discriminator model")
    disc_model_path = os.path.join(models_folder, f"disc_model_{cfg.disc_model.model}_{run['parameters/dataset'].fetch()}.pt")
    disc_model = torch.load(disc_model_path)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    logger.info("Loading generator model")
    gen_model_path = os.path.join(models_folder, f"gen_model_{cfg.gen_model.model}_orig_{run['parameters/dataset'].fetch()}.pt")
    gen_model = torch.load(gen_model_path)
    cf_class = PPCEF(
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=torch.nn.BCELoss(),
        neptune_run=neptune
    )

    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    time_start = time()
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
    run["metrics/eval_time"] = np.mean(time() - time_start)

    Xs_cfs = np.array(Xs_cfs, dtype=np.float32).squeeze()
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)
    delta = cf_class.calculate_median_log_prob(dataset.train_dataloader(batch_size=64, shuffle=False))
    metrics = evaluate_cf(
        cf_class=cf_class,
        delta=delta,
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
