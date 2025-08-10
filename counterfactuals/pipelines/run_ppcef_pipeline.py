import logging
import os
from time import time
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.ppcef import PPCEF
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths

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
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
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
        tuple (A tuple containing):
            - Xs_cfs (np.ndarray): Generated counterfactual examples
            - Xs (np.ndarray): Original examples used for counterfactual generation
            - log_prob_threshold (float): Calculated log probability threshold
            - ys_orig (np.ndarray): Original labels
            - ys_target (np.ndarray): Target labels for counterfactuals
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
    delta, Xs, ys_orig, ys_target, logs = cf_method.explain_dataloader(
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
    )

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search time: {cf_search_time:.4f} seconds")
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    Xs_cfs = Xs + delta

    if cfg.counterfactuals_params.use_categorical:
        Xs_cfs = apply_categorical_discretization(
            dataset.categorical_features_lists, Xs_cfs
        )

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to: {counterfactuals_path}")
    return Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target


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


def get_categorical_intervals(
    use_categorical: bool, categorical_features_lists: list
):
    """Get categorical intervals for feature discretization."""
    return categorical_features_lists if use_categorical else None


def apply_categorical_discretization(
    categorical_features_lists: list, Xs_cfs: np.ndarray
) -> np.ndarray:
    """Apply categorical discretization to counterfactuals."""
    for interval in categorical_features_lists:
        max_indices = np.argmax(Xs_cfs[:, interval], axis=1)
        Xs_cfs[:, interval] = np.eye(Xs_cfs[:, interval].shape[1])[max_indices]
    return Xs_cfs


@hydra.main(config_path="./conf", config_name="ppcef_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing pipeline")

    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    
    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)
    gen_model = create_gen_model(cfg, dataset, gen_model_path)

    # Custom code
    Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target = search_counterfactuals(
        cfg, dataset, gen_model, disc_model, save_folder
    )

    metrics = calculate_metrics(
        gen_model=gen_model,
        disc_model=disc_model,
        Xs_cfs=Xs_cfs,
        model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=Xs,
        y_test=ys_orig,
        y_target=ys_target,
        median_log_prob=log_prob_threshold,
    )
    print(metrics)


if __name__ == "__main__":
    main()
