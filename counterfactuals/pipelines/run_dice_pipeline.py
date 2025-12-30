import logging
import os
import warnings
from time import time
from typing import Any, Dict, List, Optional, Tuple

import dice_ml
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.full_pipeline.full_pipeline import full_pipeline
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DiscWrapper(nn.Module):
    """
    Wrap a discriminative model with a PyTorch-style `forward` method for DiCE.

    This thin wrapper simply applies a sigmoid to the raw model outputs to obtain
    probabilities, as expected by `dice-ml` when using the `PYT` backend.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactual explanations using the DiCE method.

    This function constructs a `dice-ml` Data and Model interface around the provided
    dataset and discriminator, computes a plausibility threshold (log-probability)
    using the generative model, and generates counterfactuals per eligible test
    instance (those not in the target class). DiCE-specific parameters (backend,
    method, total_cfs, desired_class, posthoc_sparsity_param, learning_rate) are
    read from the configuration.

    Args:
        cfg: Hydra configuration containing experiment and DiCE method parameters
        dataset: Dataset object with training/test splits and metadata
        gen_model: Trained generative model used to compute plausibility threshold
        disc_model: Trained discriminative model used for classification
        save_folder: Directory path where generated counterfactuals will be saved

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactual explanations
            - Xs: Original test instances used for CF generation
            - ys_orig: Original predicted labels for the test instances
            - ys_target: Target labels corresponding to counterfactuals
            - cf_search_time: Average time taken for counterfactual search
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

    # Combine train and test data for DiCE to establish proper feature ranges
    # This prevents DiCE from rejecting test instances that are outside training range
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

    generation_params = OmegaConf.to_container(
        cfg.counterfactuals_params.generation_params
    )

    cfs = exp.generate_counterfactuals(query_instance, **generation_params)

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    Xs_cfs = []
    for orig, cf in zip(X_test_origin, cfs.cf_examples_list):
        if cf.final_cfs_df is None:
            Xs_cfs.append(orig)
            continue
        out = cf.final_cfs_df.to_numpy()
        if out.shape[0] > 0:
            Xs_cfs.append(out[0][:-1])
        else:
            Xs_cfs.append(orig)

    Xs_cfs = np.array(Xs_cfs)
    ys_target = np.abs(1 - y_test_origin)
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)
    return (
        Xs_cfs,
        X_test_origin,
        y_test_origin,
        ys_target,
        model_returned,
        cf_search_time,
    )


def get_categorical_intervals(
    use_categorical: bool, categorical_features_lists: List[List[int]]
) -> Optional[List[List[int]]]:
    """Return categorical intervals if categorical processing is enabled.

    Args:
        use_categorical: Whether to apply categorical processing
        categorical_features_lists: Indices grouped by one-hot categorical blocks

    Returns:
        The provided intervals if enabled, otherwise ``None``.
    """
    return categorical_features_lists if use_categorical else None


def apply_categorical_discretization(
    categorical_features_lists: List[List[int]], Xs_cfs: np.ndarray
) -> np.ndarray:
    """Project counterfactuals onto one-hot vertices for categorical features.

    For each categorical group (one-hot block), selects the argmax index and
    sets the block to the corresponding one-hot vector.

    Args:
        categorical_features_lists: Indices grouped by one-hot categorical blocks
        Xs_cfs: Counterfactuals array to discretize

    Returns:
        The discretized counterfactuals array.
    """
    for interval in categorical_features_lists:
        max_indices = np.argmax(Xs_cfs[:, interval], axis=1)
        Xs_cfs[:, interval] = np.eye(Xs_cfs[:, interval].shape[1])[max_indices]

    return Xs_cfs


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
    Calculate comprehensive metrics for generated counterfactual explanations.

    This evaluates counterfactual quality across validity, plausibility (density),
    proximity, and diversity measures using the provided discriminative and
    generative models.

    Args:
        gen_model: Trained generative model used for plausibility assessment
        disc_model: Trained discriminative model used for validity assessment
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean mask indicating successful CF generation
        categorical_features: Indices of categorical features
        continuous_features: Indices of continuous features
        X_train: Training features
        y_train: Training labels
        X_test: Original instances used for CFs
        y_test: Original labels
        median_log_prob: Plausibility threshold (median log-probability)
        y_target: Optional target labels for counterfactuals

    Returns:
        Dictionary containing computed metrics for counterfactual quality evaluation.
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


@hydra.main(config_path="./conf", config_name="dice_config", version_base="1.2")
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
