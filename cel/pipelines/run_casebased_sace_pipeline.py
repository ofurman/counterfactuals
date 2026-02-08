import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from cel.cf_methods.local_methods.casebased_sace.casebased_sace import (
    CaseBasedSACE,
)
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


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactual explanations using the Case-Based SACE method.

    This function prepares the data by filtering out the target class, configures the
    CaseBasedSACE counterfactual generator with dataset feature metadata, computes a
    log-probability threshold from the generative model for plausibility, and then
    generates counterfactuals for the selected instances.

    Args:
        cfg: Hydra configuration containing experiment parameters
        dataset: Dataset object with train/test data and feature metadata
        gen_model: Trained generative model used to compute log-probability threshold
        disc_model: Trained discriminative model used by the CF method
        save_folder: Directory where generated counterfactuals CSV will be saved

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactual examples.
            - Xs: Original instances used for CF generation.
            - ys_orig: Original predicted labels.
            - ys_target: Target labels for counterfactuals.
            - model_returned: Boolean mask of successful generations.
            - cf_search_time: Duration of the counterfactual search.
    """
    _ = gen_model  # Required by pipeline interface; CaseBasedSACE does not use it directly.
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating counterfactual model")
    cf_method = CaseBasedSACE(
        disc_model=disc_model,
        variable_features=dataset.numerical_features_indices
        + dataset.categorical_features_indices,
        continuous_features=dataset.numerical_features_indices,
        categorical_features_lists=dataset.categorical_features_lists,
        **cfg.counterfactuals_params.cf_method,
    )

    logger.info("Handling counterfactual generation")
    cf_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )
    time_start = time()
    Xs_cfs, Xs, ys_orig, ys_target, model_returned = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        X_train=np.asarray(dataset.X_train),
        y_train=np.asarray(dataset.y_train),
    )

    cf_search_time = time() - time_start
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    Xs_cfs = np.asarray(Xs_cfs)
    Xs = np.asarray(Xs)
    ys_orig = np.asarray(ys_orig)
    ys_target = np.asarray(ys_target)
    model_returned = np.asarray(model_returned).astype(bool)

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

    Uses the provided generative and discriminative models to evaluate validity,
    plausibility, proximity, and diversity metrics on the generated counterfactuals.

    Args:
        gen_model: Trained generative model for plausibility computations
        disc_model: Trained discriminative model for validity computations
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean mask for successful generations
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


@hydra.main(config_path="./conf", config_name="casebased_sace_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline for Case-Based SACE counterfactual generation and evaluation.

    Steps:
    1. Load dataset and iterate over CV folds
    2. Create/load discriminative and generative models per fold
    3. Generate counterfactuals using CaseBasedSACE
    4. Compute evaluation metrics and save results as CSV per fold

    Args:
        cfg: Hydra configuration with dataset/model/experiment parameters

    Returns:
        None. Results are logged and written to disk.
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

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
        df_metrics.to_csv(os.path.join(save_folder, "cf_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
