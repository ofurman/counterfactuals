import logging
import os
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.pumal import PUMAL
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactuals using the PUMAL method.

    This function filters the test data to select instances from the origin class,
    instantiates the PUMAL counterfactual model, computes a log-probability threshold,
    and generates counterfactuals for the selected data.

    Args:
        cfg: Hydra configuration containing counterfactual parameters.
        dataset: Dataset containing training and test data.
        gen_model: Pre-trained generative model.
        disc_model: Pre-trained discriminative model.
        save_folder: Directory path where counterfactuals will be saved.

    Returns:
        tuple: A tuple containing:
            - Xs_cfs (np.ndarray): Generated counterfactual examples.
            - Xs (np.ndarray): Original examples used for counterfactual generation.
            - ys_orig (np.ndarray): Original labels.
            - ys_target (np.ndarray): Target labels for counterfactuals.
            - model_returned (np.ndarray): Boolean mask indicating valid counterfactuals.
            - cf_search_time (float): Time taken for counterfactual search in seconds.
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    origin_class = cfg.counterfactuals_params.origin_class
    target_class = cfg.counterfactuals_params.target_class
    y_test = dataset.y_test
    y_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test.reshape(-1)
    y_indices = y_labels.astype(int)
    mask_origin = y_indices == origin_class
    X_test_origin = dataset.X_test[mask_origin]
    if y_test.ndim > 1:
        y_test_origin = y_test[mask_origin]
    else:
        n_classes = int(np.max(y_indices)) + 1
        y_test_origin = np.eye(n_classes)[y_indices][mask_origin]
    actionable_features = getattr(dataset, "actionable_features", None)
    not_actionable_features = None
    if actionable_features:
        not_actionable_features = [
            idx
            for idx, feature in enumerate(dataset.features)
            if feature not in actionable_features
        ]

    logger.info("Creating counterfactual model")
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_model_criterion)
    cf_method = PUMAL(
        cf_method_type=cfg.counterfactuals_params.cf_method.cf_method_type,
        K=cfg.counterfactuals_params.cf_method.K,
        X=X_test_origin,
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=disc_model_criterion,
        not_actionable_features=not_actionable_features,
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
    delta, Xs, _, _ = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        target_class=target_class,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        patience=cfg.counterfactuals_params.patience,
        alpha_dist=cfg.counterfactuals_params.alpha_dist,
        alpha_plaus=cfg.counterfactuals_params.alpha_plaus,
        alpha_class=cfg.counterfactuals_params.alpha_class,
        alpha_s=cfg.counterfactuals_params.alpha_s,
        alpha_k=cfg.counterfactuals_params.alpha_k,
        alpha_d=cfg.counterfactuals_params.alpha_d,
        log_prob_threshold=log_prob_threshold,
        decrease_loss_patience=cfg.counterfactuals_params.decrease_loss_patience,
    )

    cf_search_time = np.mean(time() - time_start)
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    Xs_cfs = Xs + delta().detach().numpy()
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to: %s", counterfactuals_path)

    ys_orig = y_indices[mask_origin]
    ys_target = np.full_like(ys_orig, target_class)
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)
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


@hydra.main(config_path="./conf", config_name="pumal_config", version_base="1.2")
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
