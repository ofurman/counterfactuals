import logging
import os
from time import time
from typing import Any, Dict, Tuple

import hydra
import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend to prevent Qt issues
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.artelt.artelt import Artelt
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


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate counterfactual explanations using the Artelt method.

    This function implements the Artelt counterfactual generation algorithm, which uses
    density estimators to generate valid counterfactuals. It filters the test data to
    exclude the target class, fits density estimators on the training data, and then
    generates counterfactuals for the filtered test instances.

    Args:
        cfg: Hydra configuration containing experiment parameters
        dataset: Dataset object containing training and test data
        gen_model: Trained generative model for density estimation
        disc_model: Trained discriminative model for classification
        save_folder: Directory path where results will be saved

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactual explanations
            - Xs: Original test instances
            - ys_orig: Original labels for test instances
            - ys_target: Target labels for counterfactuals
            - model_returned: Boolean array indicating successful generation
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating counterfactual model")
    cf_method: Artelt = Artelt(disc_model=disc_model)

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
    cf_method.fit_density_estimators(X_train=dataset.X_train, y_train=dataset.y_train)
    explanation_result = cf_method.explain_dataloader(
        dataloader=cf_dataloader, X_train=dataset.X_train, y_train=dataset.y_train
    )
    
    Xs_cfs = explanation_result.x_cfs
    Xs = explanation_result.x_origs
    ys_orig = explanation_result.y_origs
    ys_target = explanation_result.y_cf_targets
    model_returned = explanation_result.logs["model_returned"]

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to {counterfactuals_path}")

    return Xs_cfs, Xs, ys_orig, ys_target, model_returned


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
    Calculate comprehensive metrics for generated counterfactual explanations.

    This function evaluates the quality of counterfactual explanations using various
    metrics including validity, coverage, proximity, diversity, and density-based
    plausibility measures.

    Args:
        gen_model: Trained generative model used for density estimation
        disc_model: Trained discriminative model used for classification
        Xs_cfs: Generated counterfactual explanations
        model_returned: Boolean array indicating successful generation
        categorical_features: List of categorical feature indices
        continuous_features: List of continuous feature indices
        X_train: Training data features
        y_train: Training data labels
        X_test: Test data features (original instances)
        y_test: Test data labels (original labels)
        median_log_prob: Median log probability threshold for plausibility
        y_target: Target labels for counterfactuals (optional)

    Returns:
        Dictionary containing computed metrics for counterfactual quality evaluation
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
    logger.info(f"Metrics calculated: {list(metrics.keys())}")
    return metrics


@hydra.main(config_path="./conf", config_name="artelt_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline function for Artelt counterfactual generation.

    This function orchestrates the complete pipeline for generating and evaluating
    counterfactual explanations using the Artelt method. It performs cross-validation
    across multiple folds, trains models if required, generates counterfactuals,
    and calculates comprehensive evaluation metrics.

    The pipeline includes:
    1. Dataset loading and preprocessing
    2. Cross-validation setup (5 folds)
    3. Discriminative and generative model training/loading
    4. Counterfactual generation using Artelt method
    5. Metrics calculation and results saving

    Args:
        cfg: Hydra configuration containing all experiment parameters including
             dataset settings, model configurations, and counterfactual parameters
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

        dequantizer.fit(dataset.X_train)
        gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer)

        # Calculate log_prob_threshold
        dataset.X_train = dequantizer.transform(dataset.X_train)
        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
        )
        log_prob_threshold = torch.quantile(
            gen_model.predict_log_prob(train_dataloader_for_log_prob),
            cfg.counterfactuals_params.log_prob_quantile,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)

        Xs_cfs, Xs, ys_orig, ys_target, model_returned = (
            search_counterfactuals(cfg, dataset, gen_model, disc_model, save_folder)
        )

        if not any(model_returned):
            logger.info("No counterfactuals found, skipping metrics calculation")
            continue

        gen_model = DequantizationWrapper(gen_model, dequantizer)

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )

        logger.info(f"Fold {fold_n} completed. Metrics: {metrics}")
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(os.path.join(save_folder, "cf_metrics.csv"), index=False)

    logger.info("Artelt pipeline completed successfully")


if __name__ == "__main__":
    main()
