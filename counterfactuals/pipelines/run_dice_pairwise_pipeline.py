import logging
import os
import warnings
from time import time
from typing import Any, Dict, List, Tuple

import dice_ml
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import pdist

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.full_pipeline.full_pipeline import full_pipeline
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="dice_ml")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

CF_PER_INSTANCE = 100


class DiscWrapper(nn.Module):
    """Wrap discriminative model with sigmoid forward pass for DiCE."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def compute_pairwise_mean_distance(cfs: np.ndarray) -> float:
    """Average minimum pairwise distance across counterfactual sets.

    Args:
        cfs: Array of shape (n_instances, cfs_per_instance, n_features)

    Returns:
        Mean of minimum pairwise distances across all instances
    """
    if cfs.size == 0 or cfs.shape[1] < 2:
        return float("nan")
    mean_dists: list[float] = []
    for group in cfs:
        distances = pdist(group, metric="euclidean")
        if distances.size > 0:
            mean_dists.append(float(distances.mean()))
    return float(np.mean(mean_dists)) if mean_dists else float("nan")


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Generate counterfactuals with DiCE and expand to fixed CF_PER_INSTANCE per factual.

    Returns:
        Tuple containing:
            - Xs_cfs_bundle: Tuple of (Xs_cfs_first, Xs_cfs_all) where:
                - Xs_cfs_first: First CF per factual instance (for original metrics)
                - Xs_cfs_all: All CFs expanded (for pairwise distance)
            - X_test_origin: Original test instances
            - y_test_origin: Original test labels
            - ys_target: Target labels
            - model_returned_first: Mask indicating successful CF generation (first CF only)
            - cf_search_time: Average time for CF search
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class].astype(np.float64)
    y_test_origin = dataset.y_test[dataset.y_test != target_class].astype(np.float64)

    logger.info("Creating dataset interface")
    X_train, y_train = dataset.X_train, dataset.y_train
    features = list(range(dataset.X_train.shape[1])) + ["label"]
    features = list(map(str, features))

    logger.info("Combining train and test data for DiCE range establishment")
    X_combined = np.concatenate([X_train, X_test_origin], axis=0)
    y_combined = np.concatenate([y_train, y_test_origin], axis=0)
    combined_dataframe = pd.DataFrame(
        np.concatenate((X_combined, y_combined.reshape(-1, 1)), axis=1),
        columns=features,
    )

    dice = dice_ml.Data(
        dataframe=combined_dataframe,
        continuous_features=list(map(str, dataset.numerical_features_indices)),
        outcome_name=features[-1],
    )

    logger.info("Creating counterfactual model")
    disc_model_w = DiscWrapper(disc_model)
    model = dice_ml.Model(disc_model_w, backend=cfg.counterfactuals_params.backend)
    exp = dice_ml.Dice(dice, model, method=cfg.counterfactuals_params.method)

    logger.info("Handling counterfactual generation")
    query_instance = pd.DataFrame(X_test_origin, columns=features[:-1])
    time_start = time()
    generation_params: Dict[str, Any] = OmegaConf.to_container(
        cfg.counterfactuals_params.generation_params, resolve=True
    )
    generation_params["total_CFs"] = CF_PER_INSTANCE
    cfs = exp.generate_counterfactuals(query_instance, **generation_params)
    cf_search_time = np.mean(time() - time_start)
    logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    # Store first CF per instance (for original metrics)
    Xs_cfs_first_list: list[np.ndarray] = []
    model_returned_first_list: list[bool] = []

    # Store all CFs as 3D array (n_instances, CF_PER_INSTANCE, n_features)
    Xs_cfs_all_list: list[np.ndarray] = []

    for orig, cf in zip(X_test_origin, cfs.cf_examples_list):
        cf_df = cf.final_cfs_df
        if cf_df is None or cf_df.empty:
            Xs_cfs_first_list.append(orig)
            model_returned_first_list.append(False)
            cf_block = np.repeat(orig[None, :], CF_PER_INSTANCE, axis=0)
        else:
            cf_array = cf_df.to_numpy()[:, :-1]
            Xs_cfs_first_list.append(cf_array[0])
            model_returned_first_list.append(True)

            cf_block = cf_array[:CF_PER_INSTANCE]
            if cf_block.shape[0] < CF_PER_INSTANCE:
                deficit = CF_PER_INSTANCE - cf_block.shape[0]
                padding = np.repeat(orig[None, :], deficit, axis=0)
                cf_block = np.vstack([cf_block, padding])

        Xs_cfs_all_list.append(cf_block)

    Xs_cfs_first = np.array(Xs_cfs_first_list)
    model_returned_first = np.array(model_returned_first_list)
    Xs_cfs_all = np.stack(
        Xs_cfs_all_list
    )  # Shape: (n_instances, CF_PER_INSTANCE, n_features)
    ys_target = np.abs(1 - y_test_origin)

    # Save all CFs to file (flatten for CSV)
    pd.DataFrame(Xs_cfs_all.reshape(-1, Xs_cfs_all.shape[-1])).to_csv(
        counterfactuals_path, index=False
    )
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    return (
        (Xs_cfs_first, Xs_cfs_all),
        X_test_origin,
        y_test_origin,
        ys_target,
        model_returned_first,
        cf_search_time,
    )


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray | tuple,
    model_returned: np.ndarray,
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: np.ndarray | None = None,
    metrics_conf_path: str | None = None,
    Xs_cfs_all: np.ndarray | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """Calculate metrics using first CF only, then append pairwise distance from all CFs.

    Args:
        Xs_cfs: Tuple of (Xs_cfs_first, Xs_cfs_all) where:
            - Xs_cfs_first: First counterfactual per instance (n_instances, n_features)
            - Xs_cfs_all: All counterfactuals (n_instances * CF_PER_INSTANCE, n_features)
        Other args: Standard metric calculation arguments

    Returns:
        Dictionary of metrics with pairwise_min_distance included
    """
    Xs_cfs_first, Xs_cfs_all = Xs_cfs
    Xs_cfs = Xs_cfs_first

    logger.info("Calculating standard metrics using first CF per instance...")
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
        metrics_conf_path=metrics_conf_path
        or "counterfactuals/pipelines/conf/metrics/default.yaml",
    )

    # Calculate pairwise distance on all CFs
    logger.info("Calculating pairwise minimum distance across all CFs...")
    metrics["pairwise_min_distance"] = compute_pairwise_mean_distance(Xs_cfs_all)

    logger.info("Metrics calculated: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="dice_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run DiCE pipeline with fixed-number CFs and additional diversity metric."""
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
