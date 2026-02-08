import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import pdist

from cel.cf_methods.local_methods.c_chvae.c_chvae import CCHVAE
from cel.cf_methods.local_methods.c_chvae.data import CustomData
from cel.cf_methods.local_methods.c_chvae.mlmodel import CustomMLModel
from cel.datasets.method_dataset import MethodDataset
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.full_pipeline.full_pipeline import full_pipeline
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def compute_pairwise_mean_distance(cfs: np.ndarray) -> float:
    """Average pairwise distance across counterfactual sets.

    Args:
        cfs: Array of shape (n_instances, cfs_per_instance, n_features).

    Returns:
        Mean of pairwise distances across all instances.
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
    dataset: MethodDataset,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Generate multiple counterfactuals using CCHVAE for diversity metrics.

    Returns:
        Tuple containing:
            - Xs_cfs_bundle: Tuple of (Xs_cfs_first, Xs_cfs_all) where:
                - Xs_cfs_first: First CF per factual instance
                - Xs_cfs_all: All CFs per instance
            - X_test_origin: Original test instances
            - y_test_origin: Original test labels
            - y_target: Target labels
            - model_returned_first: Mask indicating successful CF generation
            - cf_search_time: Average time for CF search
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
    if not hyperparams.get("data_name"):
        hyperparams["data_name"] = cfg.dataset.config_path.split("/")[-1].split(".")[0]

    input_size = dataset.X_train.shape[1]
    hyperparams["vae_params"]["layers"] = [input_size] + hyperparams["vae_params"][
        "layers"
    ]

    exp = CCHVAE(wrapped_model, hyperparams)

    logger.info("Handling counterfactual generation")
    cf_per_instance = int(cfg.counterfactuals_params.get("num_counterfactuals", 1))
    cf_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )

    time_start = time()
    cfs_list: list[np.ndarray] = []
    Xs = X_test_origin
    ys_orig = y_test_origin
    y_target = np.abs(1 - y_test_origin)
    for _ in range(cf_per_instance):
        explanation_result = exp.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=cfg.counterfactuals_params.epochs,
            lr=cfg.counterfactuals_params.lr,
            y_target=y_target,
        )
        cfs_list.append(explanation_result.x_cfs)

    cf_search_time = time() - time_start
    logger.info("Counterfactual search time: %.4f seconds", cf_search_time)

    Xs_cfs_all = np.stack(cfs_list, axis=1)
    Xs_cfs_first = Xs_cfs_all[:, 0, :]

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs_all.reshape(-1, Xs_cfs_all.shape[-1])).to_csv(
        counterfactuals_path, index=False
    )
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    model_returned_first = np.ones(Xs_cfs_first.shape[0], dtype=bool)
    return (
        (Xs_cfs_first, Xs_cfs_all),
        Xs,
        ys_orig,
        y_target,
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
    y_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate metrics for CCHVAE CFs and append diversity score."""
    Xs_cfs_first, Xs_cfs_all = Xs_cfs

    logger.info("Calculating standard metrics using first CF per instance...")
    metrics = evaluate_cf(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs_first,
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

    logger.info("Calculating pairwise distance across all CFs...")
    metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(Xs_cfs_all)

    logger.info("Metrics calculated: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="cchvae_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run CCHVAE pipeline with multiple CFs for diversity metrics."""
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
