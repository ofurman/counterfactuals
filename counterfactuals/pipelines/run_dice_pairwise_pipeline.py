import logging
import os
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


def compute_pairwise_min_distance(samples: np.ndarray, group_size: int) -> float:
    """Average minimum pairwise distance across counterfactual sets."""
    if samples.size == 0 or group_size < 2:
        return float("nan")
    min_dists: list[float] = []
    num_groups = samples.shape[0] // group_size
    for gid in range(num_groups):
        start = gid * group_size
        end = start + group_size
        group_points = samples[start:end]
        if group_points.shape[0] < 2:
            continue
        distances = pdist(group_points, metric="euclidean")
        if distances.size > 0:
            min_dists.append(float(distances.min()))
    return float(np.mean(min_dists)) if min_dists else float("nan")


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate counterfactuals with DiCE and expand to fixed CF_PER_INSTANCE per factual."""
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

    Xs_cfs_blocks: list[np.ndarray] = []
    Xs_blocks: list[np.ndarray] = []
    ys_orig_blocks: list[np.ndarray] = []
    ys_target_blocks: list[np.ndarray] = []
    model_returned_blocks: list[np.ndarray] = []

    for orig_idx, (orig, cf) in enumerate(zip(X_test_origin, cfs.cf_examples_list)):
        cf_df = cf.final_cfs_df
        if cf_df is None or cf_df.empty:
            cf_block = np.repeat(orig[None, :], CF_PER_INSTANCE, axis=0)
            returned_mask = np.zeros(CF_PER_INSTANCE, dtype=bool)
        else:
            cf_array = cf_df.to_numpy()[:, :-1]
            cf_block = cf_array[:CF_PER_INSTANCE]
            returned_mask = np.ones(cf_block.shape[0], dtype=bool)
            if cf_block.shape[0] < CF_PER_INSTANCE:
                deficit = CF_PER_INSTANCE - cf_block.shape[0]
                padding = np.repeat(orig[None, :], deficit, axis=0)
                cf_block = np.vstack([cf_block, padding])
                returned_mask = np.concatenate(
                    [returned_mask, np.zeros(deficit, dtype=bool)]
                )

        Xs_cfs_blocks.append(cf_block)
        Xs_blocks.append(np.repeat(orig[None, :], CF_PER_INSTANCE, axis=0))
        ys_orig_blocks.append(np.repeat(y_test_origin[orig_idx], CF_PER_INSTANCE))
        ys_target_blocks.append(np.repeat(target_class, CF_PER_INSTANCE))
        model_returned_blocks.append(returned_mask)

    Xs_cfs = np.vstack(Xs_cfs_blocks)
    Xs_expanded = np.vstack(Xs_blocks)
    ys_orig_expanded = np.concatenate(ys_orig_blocks)
    ys_target = np.concatenate(ys_target_blocks)
    model_returned = np.concatenate(model_returned_blocks)

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    return (
        Xs_cfs,
        Xs_expanded,
        ys_orig_expanded,
        ys_target,
        model_returned,
        cf_search_time,
    )


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
    y_target: np.ndarray | None = None,
    metrics_conf_path: str | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """Calculate metrics and append pairwise minimum distance per factual."""
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
    metrics["pairwise_min_distance"] = compute_pairwise_min_distance(
        Xs_cfs, CF_PER_INSTANCE
    )
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
