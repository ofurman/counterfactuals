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

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.global_cfs_utils.ares import AReS
from counterfactuals.global_cfs_utils.global_ce import GLOBE_CE

from counterfactuals.global_cfs_utils.datasets_split import dataset_loader_split
import counterfactuals.global_cfs_utils.models as models
from counterfactuals.metrics.metrics import continuous_distance, categorical_distance, distance_l2_jaccard, distance_mad_hamming, sparsity, perc_valid_cf


NORMALISERS = {
    "dnn": models.dnn_normalisers,
    "lr": models.lr_normalisers,
    "xgboost": 3
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf/other_methods", config_name="config_ares", version_base="1.2")
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
    disc_model_path = os.path.join(f"{cfg.model_path}/{dataset_name}_{disc_model_name}.pkl")
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
    dataset = dataset_loader_split(dataset_name, dropped_features=dropped_features, n_bins=None)
    X_train, y_train, X_test, y_test, x_means, x_std = dataset.get_split(normalise=False, shuffle=False,
                                                                     return_mean_std=True)
    
    X = pd.DataFrame(X_train)
    X.columns = dataset.features[:-1]
    X_test = pd.DataFrame(X_test)
    X_test.columns = dataset.features[:-1]

    logger.info("Loading discriminator model")
    with open(disc_model_path, 'rb') as f:
        disc_model = pickle.load(f)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    normalisers = NORMALISERS.get(cfg.model, {dataset_name: False})

    ares = AReS(model=disc_model, dataset=dataset, X=X, n_bins=10, normalise=normalisers[dataset_name])  # 1MB
    bin_widths = ares.bin_widths

    ordinal_features = ['Present-Employment'] if dataset_name == 'german_credit' else []
    globe_ce = GLOBE_CE(model=disc_model, dataset=dataset, X=X, affected_subgroup=None,
                        dropped_features=[], ordinal_features=ordinal_features, delta_init='zeros',
                        normalise=None, bin_widths=bin_widths, monotonicity=None, p=1)

    logger.info("Handling counterfactual generation")
    time_start = time()

    best_delta = get_best_delta(globe_ce)

    min_costs = np.zeros(globe_ce.x_aff.shape[0])
    _, cos_s, k_s = globe_ce.scale(best_delta, disable_tqdm=False, vector=True)  # Algorithm 1, Line 3
    min_costs, min_costs_idxs = globe_ce.min_scalar_costs(cos_s, return_idxs=True, inf=True)  # Implicitly computes Algorithm 1, Lines 4-6, returning minimum costs per input and their indices in the k vector
    min_costs = min_costs.min(axis=0)

    Xs_cfs = np.zeros((k_s.shape[0], globe_ce.x_aff.shape[0], globe_ce.x_aff.shape[1]))
    for i, k_val in enumerate(k_s):
        Xs_cfs[i] = globe_ce.x_aff + k_val*best_delta


    run["metrics/eval_time"] = np.mean(time() - time_start)
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    # pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    # run["counterfactuals"].upload(counterfactuals_path)

    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

    logger.info("Calculating metrics")

    X_aff = ares.X_aff_original.values
    evaluate_globe_ce(Xs_cfs, X_aff, X_test.values, disc_model, model_returned)
    run["metrics/cf"] = stringify_unsupported({})
    logger.info("Finalizing and stopping run")
    run.stop()


def get_best_delta(globe_ce):
    globe_ce.sample(n_sample=1000, magnitude=2, sparsity_power=1,
                    idxs=None, n_features=2, disable_tqdm=False,  # 2 random features chosen at each sample, no sparsity smoothing (p=1)
                    plot=False, seed=0, scheme='random', dropped_features=[])  # plot=False
    delta = globe_ce.best_delta  # pick best delta
    return delta

def evaluate_globe_ce(X_cf, X_aff, X_test, model, model_returned):
    return
    categorical_features = range(X_cf.shape[1])
    continuous_features = range(X_cf.shape[1])

    model_returned_smth = np.sum(model_returned) / len(model_returned)

    ys_cfs_disc_pred = np.array(model.predict(X_cf))

    valid_cf_disc_metric = perc_valid_cf(np.zeros_like(ys_cfs_disc_pred), y_cf=ys_cfs_disc_pred)


    hamming_distance_metric = categorical_distance(
        X=X_aff, X_cf=X_cf, categorical_features=categorical_features, metric="hamming", agg="mean"
    )
    jaccard_distance_metric = categorical_distance(
        X=X_aff, X_cf=X_cf, categorical_features=categorical_features, metric="jaccard", agg="mean"
    )
    manhattan_distance_metric = continuous_distance(
        X=X_aff, X_cf=X_cf, continuous_features=continuous_features, metric="cityblock", X_all=X_test
    )
    euclidean_distance_metric = continuous_distance(
        X=X_aff, X_cf=X_cf, continuous_features=continuous_features, metric="euclidean", X_all=X_test
    )
    mad_distance_metric = continuous_distance(
        X=X_aff, X_cf=X_cf, continuous_features=continuous_features, metric="mad", X_all=X_test
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