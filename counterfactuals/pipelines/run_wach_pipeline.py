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

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.cf_methods.wach.wach import WACH
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model

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
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Generate counterfactuals using the WACH method.

    Filters test instances to those not in the target class, builds the WACH
    counterfactual explainer, computes a plausibility threshold from the
    generative model, and runs CF search in batches.

    Args:
        cfg: Hydra configuration with experiment parameters.
        dataset: Dataset object exposing train/test arrays and metadata.
        gen_model: Trained generative model for plausibility thresholding.
        disc_model: Trained discriminative model used by WACH.
        save_folder: Directory for saving generated CFs as CSV.

    Returns:
        Tuple containing:
        - Xs_cfs: Generated counterfactuals
        - Xs: Original instances
        - log_prob_threshold: Computed log-prob threshold
        - ys_orig: Original labels
        - ys_target: Target labels for CFs
        - model_returned: Boolean array indicating successful generation
        - cf_search_time: Average CF search time in seconds
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]
    # X_test_target = dataset.X_test[dataset.y_test == target_class]

    logger.info("Creating counterfactual model")
    cf_method: WACH = WACH(disc_model=disc_model)

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
    Xs_cfs, Xs, ys_orig, ys_target, model_returned = cf_method.explain_dataloader(
        dataloader=cf_dataloader, target_class=target_class
    )

    cf_search_time = np.mean(time() - time_start)
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)
    return (
        Xs_cfs,
        Xs,
        log_prob_threshold,
        ys_orig,
        ys_target,
        model_returned,
        cf_search_time,
    )


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
    """Compute evaluation metrics for WACH-generated counterfactuals.

    Args:
        gen_model: Generative model used for plausibility metrics.
        disc_model: Discriminative model used to evaluate outcomes.
        Xs_cfs: Generated counterfactuals.
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


@hydra.main(config_path="./conf", config_name="wach_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """WACH pipeline: generate and evaluate counterfactuals.

    Runs 5-fold CV: loads dataset, prepares models, generates WACH CFs for
    non-target-class samples, evaluates via ``evaluate_cf``, and writes results
    locally.

    Args:
        cfg: Hydra configuration including dataset, model, and CF parameters.
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        logger.info("Processing fold %d", fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            logger.info("Relabeling dataset with discriminative model predictions")
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        (
            Xs_cfs,
            Xs,
            log_prob_threshold,
            ys_orig,
            ys_target,
            model_returned,
            cf_search_time,
        ) = search_counterfactuals(cfg, dataset, gen_model, disc_model, save_folder)

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

        logger.info("Metrics: %s", metrics)
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
