import logging
import os
import dice_ml
import hydra
import numpy as np
import pandas as pd
from time import time
from typing import List
import torch
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.utils
import torch.nn as nn

from counterfactuals.metrics.metrics import evaluate_cf

from counterfactuals.pipelines.nodes.helper_nodes import log_parameters, set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DiscWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = torch.sigmoid(self.model(x))
        return x


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

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating dataset interface")
    X_train, y_train = dataset.X_train, dataset.y_train

    features = list(range(dataset.X_train.shape[1])) + ["label"]
    features = list(map(str, features))
    input_dataframe = pd.DataFrame(
        np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
        columns=features,
    )

    dice = dice_ml.Data(
        dataframe=input_dataframe,
        continuous_features=list(map(str, dataset.numerical_columns)),
        outcome_name=features[-1],
    )

    logger.info("Creating counterfactual model")

    disc_model_w = DiscWrapper(disc_model)

    model = dice_ml.Model(disc_model_w, backend="PYT")
    exp = dice_ml.Dice(dice, model, method="gradient")

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
    query_instance = pd.DataFrame(X_test_origin, columns=features[:-1])
    query_instance = query_instance
    time_start = time()
    cfs = exp.generate_counterfactuals(
        query_instance,
        total_CFs=1,
        desired_class="opposite",
        posthoc_sparsity_param=None,
        learning_rate=0.05,
    )

    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    Xs_cfs = []
    for orig, cf in zip(X_test_origin, cfs.cf_examples_list):
        out = cf.final_cfs_df.to_numpy()
        if out.shape[0] > 0:
            print(out[0].shape)
            Xs_cfs.append(out[0][:-1])
        else:
            print(orig.shape)
            Xs_cfs.append(orig)

    Xs_cfs = np.array(Xs_cfs)
    ys_target = np.abs(1 - y_test_origin)

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)
    return (
        Xs_cfs,
        X_test_origin,
        log_prob_threshold,
        y_test_origin,
        ys_target,
        cf_search_time,
    )


def get_categorical_intervals(
    use_categorical: bool, categorical_features_lists: List[List[int]]
):
    return categorical_features_lists if use_categorical else None


def apply_categorical_discretization(
    categorical_features_lists: List[List[int]], Xs_cfs: np.ndarray
) -> np.ndarray:
    for interval in categorical_features_lists:
        max_indices = np.argmax(Xs_cfs[:, interval], axis=1)
        Xs_cfs[:, interval] = np.eye(Xs_cfs[:, interval].shape[1])[max_indices]

    return Xs_cfs


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
    run: neptune.Run,
    y_target: np.ndarray = None,
):
    """
    Calculate metrics for counterfactuals
    """
    logger.info("Calculating metrics")
    metrics = evaluate_cf(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        median_log_prob=median_log_prob,
        y_target=y_target,
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
    return metrics


@hydra.main(config_path="./conf", config_name="dice_config", version_base="1.2")
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
    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder, run)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
        dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()
    gen_model = create_gen_model(cfg, dataset, gen_model_path, run)

    # Custom code
    Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, cf_search_time = (
        search_counterfactuals(cfg, dataset, gen_model, disc_model, run, save_folder)
    )

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
        run=run,
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics["cf_search_time"] = cf_search_time
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    df_metrics.to_csv(
        os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
    )

    run.stop()


if __name__ == "__main__":
    main()
