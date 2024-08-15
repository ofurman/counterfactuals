import logging
import os
import json
import hydra
import neptune
from neptune.utils import stringify_unsupported
import numpy as np
import pandas as pd
from time import time
import torch
import pickle
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.ares import AReS
from counterfactuals.cf_methods.global_ce import GLOBE_CE

from counterfactuals.global_cfs_utils.datasets_split import dataset_loader_split
import counterfactuals.global_cfs_utils.models as models
from counterfactuals.metrics.metrics import (
    continuous_distance,
    categorical_distance,
    distance_l2_jaccard,
    distance_mad_hamming,
    sparsity,
    perc_valid_cf,
)


NORMALISERS = {"dnn": models.dnn_normalisers, "lr": models.lr_normalisers, "xgboost": 3}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_global_ce", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    dataset_name = cfg.dataset_name
    disc_model_name = cfg.model
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    disc_model_path = os.path.join(
        f"{cfg.model_path}/{dataset_name}_{disc_model_name}.pkl"
    )
    logger.info(disc_model_path)

    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset_name
    run["parameters/disc_model"] = cfg.model
    run["parameters/counterfactuals"] = cfg.counterfactuals
    run.wait()

    logger.info("Loading dataset")
    dropped_features = []
    dataset = dataset_loader_split(
        dataset_name,
        dropped_features=dropped_features,
        n_bins=None,
        data_path="./data/",
    )
    X_train, y_train, X_test, y_test, x_means, x_std = dataset.get_split(
        normalise=False, shuffle=False, return_mean_std=True
    )

    X = pd.DataFrame(X_train)
    X.columns = dataset.features[:-1]
    X_test = pd.DataFrame(X_test)
    X_test.columns = dataset.features[:-1]

    logger.info("Loading discriminator model")
    with open(disc_model_path, "rb") as f:
        disc_model = pickle.load(f)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    normalisers = NORMALISERS.get(cfg.model, {dataset_name: False})

    ares = AReS(
        model=disc_model,
        dataset=dataset,
        X=X,
        n_bins=10,
        normalise=normalisers[dataset_name],
    )  # 1MB
    bin_widths = ares.bin_widths

    ordinal_features = ["Present-Employment"] if dataset_name == "german_credit" else []
    globe_ce = GLOBE_CE(
        model=disc_model,
        dataset=dataset,
        X=X,
        affected_subgroup=None,
        dropped_features=[],
        ordinal_features=ordinal_features,
        delta_init="zeros",
        normalise=None,
        bin_widths=bin_widths,
        monotonicity=None,
        p=1,
    )

    logger.info("Handling counterfactual generation")
    time_start = time()

    best_delta = get_best_delta(globe_ce)
    best_k_s = get_best_k_s(globe_ce, best_delta)
    Xs_cfs = get_counterfactuals(globe_ce, best_delta, best_k_s)

    run["metrics/eval_time"] = np.mean(time() - time_start)
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

    logger.info("Calculating metrics")

    X_aff = ares.X_aff_original.values
    metrics = evaluate_globe_ce(
        Xs_cfs, X_aff, X_test.values, disc_model, model_returned
    )
    print(metrics)
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info("Finalizing and stopping run")
    run.stop()


def get_best_k_s(globe_ce, best_delta):
    _, cos_s, k_s = globe_ce.scale(
        best_delta, disable_tqdm=False, vector=True
    )  # Algorithm 1, Line 3
    _, min_costs_idxs = globe_ce.min_scalar_costs(
        cos_s, return_idxs=True, inf=True
    )  # Implicitly computes Algorithm 1, Lines 4-6, returning minimum costs per input and their indices in the k vector
    best_k_s = k_s[min_costs_idxs.astype(np.int16)]
    return best_k_s


def get_counterfactuals(globe_ce, best_delta, best_k_s):
    muls_ = best_k_s.reshape(-1, 1) @ best_delta.reshape(1, -1)
    Xs_cfs = globe_ce.x_aff + muls_
    return Xs_cfs


def get_best_delta(globe_ce):
    globe_ce.sample(
        n_sample=1000,
        magnitude=2,
        sparsity_power=1,
        idxs=None,
        n_features=2,
        disable_tqdm=False,  # 2 random features chosen at each sample, no sparsity smoothing (p=1)
        plot=False,
        seed=0,
        scheme="random",
        dropped_features=[],
    )  # plot=False
    delta = globe_ce.best_delta  # pick best delta
    return delta


def evaluate_globe_ce(X_cf, X_aff, X_test, model, model_returned):
    categorical_features = range(X_cf.shape[1])
    continuous_features = range(X_cf.shape[1])

    model_returned_smth = np.sum(model_returned) / len(model_returned)

    ys_cfs_disc_pred = torch.tensor(model.predict(X_cf))

    valid_cf_disc_metric = perc_valid_cf(
        torch.zeros_like(ys_cfs_disc_pred), y_cf=ys_cfs_disc_pred
    )

    hamming_distance_metric = categorical_distance(
        X=X_aff,
        X_cf=X_cf,
        categorical_features=categorical_features,
        metric="hamming",
        agg="mean",
    )
    jaccard_distance_metric = categorical_distance(
        X=X_aff,
        X_cf=X_cf,
        categorical_features=categorical_features,
        metric="jaccard",
        agg="mean",
    )
    manhattan_distance_metric = continuous_distance(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="cityblock",
        X_all=X_test,
    )
    euclidean_distance_metric = continuous_distance(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="euclidean",
        X_all=X_test,
    )
    mad_distance_metric = continuous_distance(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="mad",
        X_all=X_test,
    )
    l2_jaccard_distance_metric = distance_l2_jaccard(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    mad_hamming_distance_metric = distance_mad_hamming(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        X_all=X_test,
        agg="mean",
    )

    X_aff, X_cf = torch.tensor(X_aff), torch.tensor(X_cf)
    sparsity_metric = sparsity(X_aff, X_cf)

    metrics = {
        "model_returned_smth": model_returned_smth,
        "valid_cf_disc": valid_cf_disc_metric,
        "dissimilarity_proximity_categorical_hamming": hamming_distance_metric,
        "dissimilarity_proximity_categorical_jaccard": jaccard_distance_metric,
        "dissimilarity_proximity_continuous_manhatan": manhattan_distance_metric,
        "dissimilarity_proximity_continuous_euclidean": euclidean_distance_metric,
        "dissimilarity_proximity_continuous_mad": mad_distance_metric,
        "distance_l2_jaccard": l2_jaccard_distance_metric,
        "distance_mad_hamming": mad_hamming_distance_metric,
        "sparsity": sparsity_metric,
    }
    return metrics


if __name__ == "__main__":
    main()
