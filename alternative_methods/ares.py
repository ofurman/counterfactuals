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
from counterfactuals.global_cfs_utils.datasets_split import dataset_loader_split
import counterfactuals.global_cfs_utils.models as models


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
    # disc_model_path = "/home/lukasz/genwro/counterfactuals/counterfactuals/global_cfs_utils/models/german_credit_lr.pkl"

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

    ares = AReS(model=disc_model, dataset=dataset, X=X, dropped_features=[],
            n_bins=10, ordinal_features=[], normalise=normalisers[dataset_name],
            constraints=[20,7,10])

    logger.info("Handling counterfactual generation")
    time_start = time()

    Xs_cfs = generate_ares_counterfactuals(ares)

    run["metrics/eval_time"] = np.mean(time() - time_start)  # probably pointless because many versions of counterfactuals are generated
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

    logger.info("Calculating metrics")

    # TODO: add evaluation

    run["metrics/cf"] = stringify_unsupported({})
    logger.info("Finalizing and stopping run")
    run.stop()


def generate_ares_counterfactuals(ares):
    ares.generate_itemsets(apriori_threshold=0.2, max_width=None,
                        affected_subgroup=None, save_copy=True)
    ares.generate_groundset(max_width=None, RL_reduction=True,
                            then_generation=None, save_copy=False)
    ares.evaluate_groundset(lams=[1, 0], r=3000, save_mode=1,
                            disable_tqdm=False, plot_accuracy=False)
    Xs_cfs = ares.V.cfx_matrix[-1]
    return Xs_cfs


if __name__ == "__main__":
    main()
