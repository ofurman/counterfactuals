import logging
import os
from time import time

import hydra
import numpy as np
import neptune
from neptune.utils import stringify_unsupported
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.helper_nodes import log_parameters, set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model

from counterfactuals.cf_methods.ares.ares import AReS, dnn_normalisers, lr_normalisers
from counterfactuals.cf_methods.ares.utils import ares_one_hot, add_method_variables

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


NORMALISERS = {
    "dnn": dnn_normalisers,
    "lr": lr_normalisers,
}

r_values = {"HelocDataset": 3000, "AuditDataset": 100}


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    run: neptune.Run,
    save_folder: str,
) -> torch.nn.Module:
    """
    Create a counterfactual model
    """

    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    dataset_name = cfg.dataset._target_.split(".")[-1]

    logger.info("Creating counterfactual model")
    normalisers = NORMALISERS.get(cfg.model, {dataset_name: False})

    Xs = pd.DataFrame(dataset.X_test, columns=dataset.feature_columns).astype(
        np.float32
    )

    dataset.X_train = pd.DataFrame(dataset.X_train, columns=dataset.feature_columns)
    dataset.X_test = pd.DataFrame(dataset.X_test, columns=dataset.feature_columns)
    target_class = cfg.counterfactuals_params.target_class

    ares = AReS(
        model=disc_model,
        dataset=dataset,
        X=Xs,
        dropped_features=[],
        n_bins=10,
        ordinal_features=[],
        normalise=normalisers[dataset_name],
        constraints=[20, 7, 10],
        dataset_name=dataset_name,
        target_class=target_class,
        logger=logger,
    )
    dataset.X_train = dataset.X_train.to_numpy()
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals.batch_size, shuffle=False
    )

    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    run["parameters/log_prob_threshold"] = log_prob_threshold
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    logger.info("Handling counterfactual generation")
    time_start = time()

    Xs_cfs = ares.generate_counterfactuals(r_values.get(dataset_name))
    Xs = ares.X_aff_original.values

    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)
    ys_orig = np.ones(Xs_cfs.shape[0]) * target_class
    ys_target = 1 - ys_orig
    return Xs_cfs, Xs, log_prob_threshold, model_returned, ys_orig, ys_target


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: list,
    continuous_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: np.ndarray = None,
):
    metrics = evaluate_cf(
        X_cf=Xs_cfs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        disc_model=disc_model,
        gen_model=gen_model,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        median_log_prob=median_log_prob,
        y_target=y_target,
    )
    return metrics


@hydra.main(config_path="./conf", config_name="ares_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    log_parameters(cfg, run)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    add_method_variables(dataset)
    ares_one_hot(dataset)

    for fold_n, (_, _, _, _) in enumerate(dataset.get_cv_splits(n_splits=5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder, run)

        run[f"{fold_n}/disc_model"].upload(disc_model_path)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

        gen_model = create_gen_model(cfg, dataset, gen_model_path, run)

        Xs_cfs, Xs, log_prob_threshold, model_returned, ys_orig, ys_target = (
            search_counterfactuals(
                cfg, dataset, gen_model, disc_model, run, save_folder
            )
        )

        logger.info("Calculating metrics")
        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )
        run[f"metrics/cf/fold_{fold_n}"] = stringify_unsupported(metrics)
        logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(os.path.join(save_folder, "cf_metrics.csv"), index=False)
    run.stop()


if __name__ == "__main__":
    main()
