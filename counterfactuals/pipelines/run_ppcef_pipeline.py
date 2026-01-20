import logging
import os
from time import time
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.ppcef import PPCEF
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.full_pipeline.full_pipeline import full_pipeline
from counterfactuals.pipelines.utils import apply_categorical_discretization
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactuals using the PPCEF method.

    This function filters the test data to exclude the target class, creates a PPCEF
    counterfactual method, calculates a log probability threshold, and generates
    counterfactuals for the filtered data.

    Args:
        cfg: Hydra configuration containing counterfactual parameters
        dataset: Dataset containing training and test data
        gen_model: Pre-trained generative model
        disc_model: Pre-trained discriminative model
        save_folder: Directory path where counterfactuals will be saved

    Returns:
        tuple: A tuple containing:
            - Xs_cfs (np.ndarray): Generated counterfactual examples
            - Xs (np.ndarray): Original examples used for counterfactual generation
            - ys_orig (np.ndarray): Original labels
            - ys_target (np.ndarray): Target labels for counterfactuals
            - cf_search_time (float): Time taken for counterfactual search in seconds
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]
    # X_test_target = dataset.X_test[dataset.y_test == target_class]

    logger.info("Creating counterfactual model")
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_model_criterion)

    cf_method = PPCEF(
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=disc_model_criterion,
    )

    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    logger.info("Handling counterfactual generation")
    cf_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )
    time_start = time()
    explanation_result = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        patience=cfg.counterfactuals_params.patience,
        alpha=cfg.counterfactuals_params.alpha,
        alpha_s=cfg.counterfactuals_params.alpha_s,
        alpha_k=cfg.counterfactuals_params.alpha_k,
        log_prob_threshold=log_prob_threshold,
        categorical_intervals=get_categorical_intervals(
            cfg.counterfactuals_params.use_categorical,
            dataset.categorical_features_lists,
        ),
        plausibility_weight=cfg.counterfactuals_params.plausibility_weight,
        plausibility_bias=cfg.counterfactuals_params.plausibility_bias,
    )
    Xs = explanation_result.x_origs
    Xs_cfs = explanation_result.x_cfs
    ys_orig = explanation_result.y_origs
    ys_target = explanation_result.y_cf_targets

    cf_search_time = time() - time_start
    logger.info(f"Counterfactual search time: {cf_search_time:.4f} seconds")
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    if cfg.counterfactuals_params.use_categorical:
        Xs_cfs = apply_categorical_discretization(
            dataset.categorical_features_lists, Xs_cfs
        )
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to: {counterfactuals_path}")
    return Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time


def get_categorical_intervals(
    use_categorical: bool, categorical_features_lists: List[List[int]]
) -> Optional[List[List[int]]]:
    """
    Get categorical feature intervals based on configuration.

    Returns the categorical features lists if categorical processing is enabled,
    otherwise returns None.

    Args:
        use_categorical: Whether to use categorical feature processing
        categorical_features_lists: List of lists containing categorical feature indices

    Returns:
        List of categorical feature intervals if use_categorical is True, None otherwise
    """
    return categorical_features_lists if use_categorical else None


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

    Evaluates the quality of counterfactuals using various metrics including validity,
    plausibility, proximity, and diversity measures.

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
    logger.info(f"Metrics:\n{metrics}")
    return metrics


@hydra.main(config_path="./conf", config_name="ppcef_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    full_pipeline(
        cfg, preprocessing_pipeline, logger, search_counterfactuals, calculate_metrics
    )


if __name__ == "__main__":
    main()
