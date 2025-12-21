import logging
import os
from time import time
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.glance.glance import GLANCE
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.dequantization.utils import DequantizationWrapper
from counterfactuals.metrics.metrics import evaluate_cf_for_glance
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]
]:
    """Generate counterfactual explanations using the GLANCE method.

    Args:
        cfg: Hydra configuration with experiment parameters.
        dataset: Dataset object containing train/test splits and metadata.
        gen_model: Trained generative model (kept for interface compatibility).
        disc_model: Trained discriminative model used by GLANCE.
        save_folder: Directory where generated counterfactuals are stored.

    Returns:
        Tuple with generated counterfactuals, original instances, original labels,
        target labels, success mask, and average search time.
    """
    _ = gen_model  # GLANCE does not rely on the generative model directly.
    cf_method_name = GLANCE.__name__
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    target_class = cfg.counterfactuals_params.target_class
    if target_class != 1:
        logger.warning(
            "GLANCE assumes target class 1; overriding configured target_class=%s",
            target_class,
        )
        target_class = 1

    logger.info("Filtering out target class data for counterfactual generation")
    Xs = dataset.X_test[dataset.y_test != target_class]
    ys_orig = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating counterfactual model")
    cf_method_cfg = cfg.counterfactuals_params.cf_method
    cf_method = GLANCE(
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        model=disc_model,
        features=list(dataset.features),
        k=int(cf_method_cfg.get("k", -1)),
        s=int(cf_method_cfg.get("s", 4)),
        m=int(cf_method_cfg.get("m", 1)),
        target_class=target_class,
    )

    logger.info("Handling counterfactual generation")
    time_start = time()
    cf_method.prep(dataset.X_train, dataset.y_train)
    Xs_cfs = cf_method.explain()
    ys_target = np.abs(ys_orig - 1)
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)
    cf_search_time = np.mean(time() - time_start)
    logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    extras = {"cf_group_ids": np.asarray(cf_method.clusters)}
    return Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time, extras


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
    y_target: np.ndarray,
    cf_group_ids: np.ndarray | None = None,
    **_: dict,
) -> dict:
    """Calculate evaluation metrics for GLANCE counterfactuals."""
    logger.info("Calculating metrics")
    metrics = evaluate_cf_for_glance(
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
        y_target=y_target,
        median_log_prob=median_log_prob,
        cf_group_ids=cf_group_ids,
        metrics_conf_path="counterfactuals/pipelines/conf/metrics/group_metrics.yaml",
    )
    logger.info("Metrics calculated: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="glance_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run GLANCE pipeline with preprocessing and standardized evaluation."""
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )

    logger.info("Loading dataset")
    file_dataset = instantiate(cfg.dataset)
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    dequantizer = GroupDequantizer(dataset.categorical_features_lists)

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        dequantizer.fit(dataset.X_train)
        gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer)

        dataset.X_train = dequantizer.transform(dataset.X_train)
        log_prob_threshold = torch.quantile(
            gen_model.predict_log_prob(
                dataset.train_dataloader(
                    batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
                )
            ),
            cfg.counterfactuals_params.log_prob_quantile,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)

        (
            Xs_cfs,
            Xs,
            ys_orig,
            ys_target,
            model_returned,
            cf_search_time,
            extras,
        ) = search_counterfactuals(cfg, dataset, gen_model, disc_model, save_folder)

        gen_model = DequantizationWrapper(gen_model, dequantizer)

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
            **extras,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
