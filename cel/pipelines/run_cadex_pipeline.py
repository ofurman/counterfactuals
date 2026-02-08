import logging
import os
from time import time
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cel.cf_methods.local_methods.cadex import CADEX
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.full_pipeline.full_pipeline import full_pipeline
from cel.pipelines.utils import apply_categorical_discretization
from cel.preprocessing import (
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
    """
    Generate counterfactuals using the CADEX method.

    Args:
        cfg: Hydra configuration containing counterfactual parameters
        dataset: Dataset containing training and test data
        gen_model: Pre-trained generative model (for density estimation/metrics)
        disc_model: Pre-trained discriminative model
        save_folder: Directory path where counterfactuals will be saved

    Returns:
        tuple: Generated counterfactuals, originals, original labels, target labels,
            model_returned mask, and search time.
    """
    _ = gen_model
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]
    y_target = np.full_like(y_test_origin, target_class)

    logger.info("Creating counterfactual model")
    cadex_params = cfg.counterfactuals_params.get("cadex", {})
    ordinal_attributes = cadex_params.get("ordinal_attributes")
    if ordinal_attributes:
        raise ValueError("cadex.ordinal_attributes requires scale/unscale hooks in CADEX.")

    cf_method = CADEX(
        disc_model=disc_model,
        categorical_attributes=dataset.categorical_features_lists,
        ordinal_attributes=ordinal_attributes,
        device=cfg.counterfactuals_params.get("device"),
    )

    logger.info("Handling counterfactual generation")
    time_start = time()
    explanation_result = cf_method.explain(
        X_test_origin,
        y_test_origin,
        y_target,
        num_changed_attributes=cadex_params.get("num_changed_attributes"),
        max_epochs=cadex_params.get("max_epochs", 1000),
        skip_attributes=cadex_params.get("skip_attributes", 0),
        categorical_threshold=cadex_params.get("categorical_threshold", 0.0),
        direction_constraints=cadex_params.get("direction_constraints"),
    )

    Xs = explanation_result.x_origs
    Xs_cfs = explanation_result.x_cfs
    ys_orig = explanation_result.y_origs
    ys_target = explanation_result.y_cf_targets

    cf_search_time = time() - time_start
    logger.info("Counterfactual search time: %.4f seconds", cf_search_time)
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    if cfg.counterfactuals_params.use_categorical:
        Xs_cfs = apply_categorical_discretization(dataset.categorical_features_lists, Xs_cfs)
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to: %s", counterfactuals_path)
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
    y_target: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for generated counterfactuals.

    Args:
        gen_model: Generative model used for plausibility assessment
        disc_model: Discriminative model used for validity assessment
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean array indicating successful counterfactual generation
        categorical_features: List of categorical feature indices
        continuous_features: List of continuous feature indices
        X_train: Training data features
        y_train: Training data labels
        X_test: Original test examples
        y_test: Original test labels
        median_log_prob: Log probability threshold for plausibility
        y_target: Target labels for counterfactuals (optional)

    Returns:
        dict: Dictionary containing computed metrics
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
    logger.info("Metrics:\n%s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="cadex_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    full_pipeline(cfg, preprocessing_pipeline, logger, search_counterfactuals, calculate_metrics)


if __name__ == "__main__":
    main()
