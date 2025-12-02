import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from counterfactuals.cf_methods.local_methods.c_chvae.c_chvae import CCHVAE
from counterfactuals.cf_methods.local_methods.c_chvae.data import CustomData
from counterfactuals.cf_methods.local_methods.c_chvae.mlmodel import CustomMLModel
from counterfactuals.metrics.metrics import evaluate_cf
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
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactuals using the CCHVAE method.

    Prepares the wrapped dataset/model, computes a log-probability threshold using the
    generative model for plausibility, and generates counterfactuals for the filtered
    test set (excluding the target class).

    Args:
        cfg: Hydra configuration containing experiment parameters
        dataset: Dataset object containing train/test data and feature metadata
        gen_model: Trained generative model for plausibility threshold
        disc_model: Trained discriminative model wrapped by CCHVAE
        save_folder: Directory where generated counterfactuals CSV will be saved

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactuals (np.ndarray)
            - Xs: Original instances used for CF generation (np.ndarray)
            - ys_orig: Original labels (np.ndarray)
            - ys_target: Target labels (np.ndarray)
            - cf_search_time: Average counterfactual search time (float)
    """

    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating dataset interface")
    custom_dataset = CustomData(dataset)

    logger.info("Creating counterfactual model")

    wrapped_model = CustomMLModel(disc_model, custom_dataset)

    hyperparams = OmegaConf.to_container(
        cfg.counterfactuals_params.hyperparams, resolve=True
    )

    input_size = dataset.X_train.shape[1]
    hyperparams["vae_params"]["layers"] = [input_size] + hyperparams["vae_params"][
        "layers"
    ]

    exp = CCHVAE(wrapped_model, hyperparams)

    logger.info("Handling counterfactual generation")
    time_start = time()
    factuals = pd.DataFrame(X_test_origin, columns=wrapped_model.feature_input_order)
    cfs = exp.get_counterfactuals_without_check(factuals)

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search time: {cf_search_time:.4f} seconds")
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    Xs_cfs = cfs.to_numpy()
    ys_target = np.abs(1 - y_test_origin)

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)
    return (
        Xs_cfs,
        X_test_origin,
        y_test_origin,
        ys_target,
        cf_search_time,
    )


def get_log_prob_threshold(
    gen_model: torch.nn.Module,
    dataset: DictConfig,
    batch_size: int,
    log_prob_quantile: float,
) -> float:
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
    return log_prob_threshold


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for generated counterfactual explanations.

    Args:
        gen_model: Generative model for plausibility computations
        disc_model: Discriminative model for validity computations
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean mask for successful CF generations
        categorical_features: Indices of categorical features
        continuous_features: Indices of continuous features
        X_train: Training features
        y_train: Training labels
        X_test: Original instances used for CFs
        y_test: Original labels
        median_log_prob: Plausibility threshold (median log-probability)
        y_target: Optional target labels for CFs

    Returns:
        Dictionary containing computed evaluation metrics.
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


@hydra.main(config_path="./conf", config_name="cchvae_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
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
