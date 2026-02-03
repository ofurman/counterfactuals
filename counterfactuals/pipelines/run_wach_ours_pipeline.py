import logging
import os
from time import time
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.wach.wach_ours import WACH_OURS
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Generate counterfactual deltas using the WACH_OURS method.

    This builds the WACH_OURS counterfactual searcher, computes a plausibility
    threshold from the generative model, and runs batched CF search over
    non-target-class test samples.

    Args:
        cfg: Hydra configuration for experiment and method settings.
        dataset: Dataset object exposing train/test arrays and metadata.
        gen_model: Trained generative model used to compute log-probability threshold.
        disc_model: Trained discriminative model used in the search loss.
        save_folder: Output directory to persist results (CSVs, timings).

    Returns:
        A tuple:
        - Xs_cfs: Counterfactual deltas from WACH_OURS (same shape as inputs)
        - Xs: Original inputs corresponding to the deltas
        - ys_orig: Original labels
        - ys_target: Target labels for counterfactuals
        - log_prob_threshold: Quantile-based log-probability threshold
        - cf_search_time: Average CF search time in seconds
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating counterfactual model")
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_model_criterion)
    cf_method: WACH_OURS = WACH_OURS(
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
    results = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        alpha=cfg.counterfactuals_params.alpha,
    )
    Xs_cfs = results.x_cfs
    Xs = results.x_origs
    ys_orig = results.y_origs
    ys_target = results.y_cf_targets

    model_returned = (np.ones(Xs_cfs.shape[0]),)

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactual deltas saved to %s", counterfactuals_path)
    return Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: List[int] | List[str],
    continuous_features: List[int] | List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute evaluation metrics for WACH_OURS-generated counterfactuals.

    Args:
        gen_model: Generative model used for plausibility metrics.
        disc_model: Discriminative model used to evaluate outcomes.
        Xs_cfs: Generated counterfactual deltas.
        model_returned: Boolean mask indicating successful generations.
        categorical_features: Indices or names of categorical features.
        continuous_features: Indices or names of continuous features.
        X_train: Training features.
        y_train: Training labels.
        X_test: Original instances.
        y_test: Original labels.
        median_log_prob: Log-probability threshold for plausibility.
        y_target: Target labels for the original instances.

    Returns:
        Mapping from metric names to values.
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
    logger.info("Metrics computed: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="wach_ours_config", version_base="1.2")
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
