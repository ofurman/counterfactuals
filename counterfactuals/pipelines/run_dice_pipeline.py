import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import dice_ml
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.dequantization.utils import DequantizationWrapper
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

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
    using the generative model, and generates one counterfactual per eligible test
    instance (those not in the target class).

    Args:
        cfg: Hydra configuration containing experiment parameters
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
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating dataset interface")
    X_train, y_train = dataset.X_train, dataset.y_train

    features = list(range(dataset.X_train.shape[1])) + ["label"]
    features = list(map(str, features))
    input_dataframe = pd.DataFrame(
        np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
        columns=features,
    )

    dice = dice_ml.Data(
        dataframe=input_dataframe,
        continuous_features=list(map(str, dataset.numerical_features_indices)),
        outcome_name=features[-1],
    )

    logger.info("Creating counterfactual model")

    disc_model_w = DiscWrapper(disc_model)

    model = dice_ml.Model(disc_model_w, backend="PYT")
    exp = dice_ml.Dice(dice, model, method="gradient")

    logger.info("Handling counterfactual generation")
    query_instance = pd.DataFrame(X_test_origin, columns=features[:-1])
    query_instance = query_instance
    time_start = time()
    cfs = exp.generate_counterfactuals(
        query_instance,
        total_CFs=1,
        desired_class="opposite",
        posthoc_sparsity_param=None,
        learning_rate=0.05,
    )

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    Xs_cfs = []
    for orig, cf in zip(X_test_origin, cfs.cf_examples_list):
        out = cf.final_cfs_df.to_numpy()
        if out.shape[0] > 0:
            Xs_cfs.append(out[0][:-1])
        else:
            Xs_cfs.append(orig)

    Xs_cfs = np.array(Xs_cfs)
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
    train_dataloader_for_log_prob = dataset.train_dataloader(batch_size=batch_size, shuffle=False)
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
    """
    Main pipeline function for DiCE counterfactual generation and evaluation.

    This function orchestrates the end-to-end pipeline for generating and evaluating
    counterfactual explanations using the DiCE method. It performs 5-fold
    cross-validation, prepares dequantization where applicable, generates
    counterfactuals, and computes comprehensive evaluation metrics.

    The pipeline includes:
    1. Dataset loading and preprocessing
    2. Cross-validation setup (5 folds)
    3. Discriminative and generative model training/loading
    4. Counterfactual generation using DiCE
    5. Metrics calculation and results saving

    Args:
        cfg: Hydra configuration containing dataset, model, and experiment parameters
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)

    logger.info("Loading dataset")
    file_dataset = instantiate(cfg.dataset)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    dequantizer = GroupDequantizer(dataset.categorical_features_lists)
    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        logger.info(f"Processing fold {fold_n}")
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            logger.info("Relabeling dataset with discriminative model predictions")
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        dequantizer.fit(dataset.X_train)
        dataset.X_train = dequantizer.transform(dataset.X_train)
        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        # Custom code
        dataset.X_train = dequantizer.transform(dataset.X_train)
        log_prob_threshold = get_log_prob_threshold(
            gen_model,
            dataset,
            cfg.counterfactuals_params.batch_size,
            cfg.counterfactuals_params.log_prob_quantile,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)
        Xs_cfs, Xs, ys_orig, ys_target, cf_search_time = search_counterfactuals(
            cfg, dataset, gen_model, disc_model, save_folder
        )

        Xs = dequantizer.inverse_transform(Xs)
        gen_model = DequantizationWrapper(gen_model, dequantizer)

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
