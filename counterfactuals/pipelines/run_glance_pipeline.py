import logging
import os
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.glance.glance import GLANCE
from counterfactuals.metrics.metrics import evaluate_cf_for_rppcef
from counterfactuals.pipelines.full_pipeline.full_pipeline import full_pipeline
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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

    return Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time


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
) -> dict:
    """Calculate evaluation metrics for GLANCE counterfactuals."""
    logger.info("Calculating metrics")
    metrics = evaluate_cf_for_rppcef(
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
        X_test_target=X_test,
    )
    logger.info("Metrics calculated: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="glance_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run GLANCE pipeline with preprocessing and standardized evaluation."""
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    full_pipeline(
        cfg,
        preprocessing_pipeline,
        logger,
        search_counterfactuals,
        calculate_metrics,
    )


if __name__ == "__main__":
    main()
