import hydra
import os
import logging
import torch
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

from counterfactuals.optimizers.approach_gen_disc_loss import ApproachGenDiscLoss
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from counterfactuals.discriminative_models import LogisticRegression, MultilayerPerceptron
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.utils import process_classification_report

logger = logging.getLogger(__name__)


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
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model"] = cfg.disc_model.model
    run["parameters/gen_model"] = cfg.gen_model.model
    run["parameters/reference_method"] = "CEM"
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
    cf_class = ApproachGenDiscLoss(
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=torch.nn.BCELoss(),
        neptune_run=neptune
    )

    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    time_start = time()

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

    Xs_cfs = []
    model_returned = []
    start_time = time()
    for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        explanation = cf.explain(X.reshape(1, -1), verbose=False)
        if explanation.PN is None:
            model_returned.append(False)
        else:
            Xs_cfs.append(explanation.PN)
            model_returned.append(True)
    run["metrics/avg_time_one_cf"] = (time() - start_time) / X_test.shape[0]
    run["metrics/eval_time"] = np.mean(time() - time_start)

    Xs_cfs = np.array(Xs_cfs).squeeze()
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
        y_train=y_train.reshape(-1),
        X_test=X_test,
        y_test=y_test.reshape(-1),
    )
    run["metrics/cf"] = metrics

    run.stop()

if __name__ == "__main__":
    main()

