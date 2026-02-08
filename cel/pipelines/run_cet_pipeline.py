import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cel.cf_methods.local_methods.cet.cet import (
    CounterfactualExplanationTree,
)
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.full_pipeline.full_pipeline import full_pipeline
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

MAX_ITERATION = 50
# LAMBDA, GAMMA = 0.01, 0.75
LAMBDA, GAMMA = 0.02, 1.0


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactuals using the CET method.

    Prepares the dataset (filters target class, wraps discrim model), computes a
    log-probability threshold using the generative model, fits CET, and predicts
    counterfactuals for the selected instances.

    Args:
        cfg: Hydra configuration with experiment parameters
        dataset: Dataset object with train/test data and metadata
        gen_model: Trained generative model to compute log-prob threshold
        disc_model: Trained discriminative model used via a wrapper
        save_folder: Directory for saving generated counterfactuals CSV

    Returns:
        Tuple containing:
            - Xs_cfs: Counterfactual candidates.
            - Xs: Corresponding originals.
            - ys_orig: Original model predictions.
            - ys_target: Target labels for CFs.
            - model_returned: Boolean mask indicating valid CFs.
            - cf_search_time: Duration of the counterfactual search.
    """
    _ = gen_model  # CET does not need the generative model directly.
    cf_method_name = "CET"
    disc_model.eval()
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    X_train = dataset.inverse_transform(dataset.X_train)
    y_train = dataset.y_train
    X_test = dataset.inverse_transform(dataset.X_test)

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    ys_pred = disc_model.predict(dataset.X_test)
    Xs = dataset.X_test[ys_pred != target_class]
    ys_orig = ys_pred[ys_pred != target_class]

    logger.info("Creating counterfactual model")
    X_train_df = pd.DataFrame(X_train, columns=dataset.features)
    columns = X_train_df.columns
    X_train = X_train_df.to_numpy()
    feature_types = ["I" for _ in range(X_train.shape[1])]
    feature_constraints = ["" for _ in range(X_train.shape[1])]
    feature_categories = []

    disc_model_wrapper = DiscModelWrapper(disc_model)

    cet = CounterfactualExplanationTree(
        disc_model_wrapper,
        X_train,
        y_train,
        max_iteration=MAX_ITERATION,
        lime_approximation=False,
        feature_names=columns,
        feature_types=feature_types,
        feature_categories=feature_categories,
        feature_constraints=feature_constraints,
        target_name=dataset.features[-1],
        target_labels=[0, 1],
    )

    logger.info("Handling counterfactual generation")
    time_start = time()
    cet = cet.fit(
        X_test,
        max_change_num=3,
        cost_type="MPS",
        C=LAMBDA,
        gamma=GAMMA,
        time_limit=60,
        verbose=True,
    )
    Xs_cfs = cet.predict(X_test)
    ys_target = np.abs(ys_orig - 1)
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)
    cf_search_time = time() - time_start
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    return Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time


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
        Xs_cfs: Generated counterfactuals
        model_returned: Boolean mask for successful CF generations
        categorical_features: Indices of categorical features
        continuous_features: Indices of continuous features
        X_train: Training features
        y_train: Training labels
        X_test: Original instances used for CFs
        y_test: Original labels
        median_log_prob: Plausibility threshold (median log-probability)
        y_target: Optional target labels

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


class DiscModelWrapper:
    """Simple wrapper adapting torch-based discriminator to CET's numpy API."""

    def __init__(self, disc_model: torch.nn.Module) -> None:
        self.disc_model = disc_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.disc_model.predict(X)
        return out.detach().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = self.disc_model.predict_proba(X)
        return out.detach().numpy()


@hydra.main(config_path="./conf", config_name="cet_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset, shuffle=False)

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, model_returned = search_counterfactuals(
            cfg, dataset, gen_model, disc_model, save_folder
        )

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
