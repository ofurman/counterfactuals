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

from cel.cf_methods.local_methods.cem.cem import CEM_CF
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.nodes.disc_model_nodes import create_disc_model
from cel.pipelines.nodes.gen_model_nodes import create_gen_model
from cel.pipelines.nodes.helper_nodes import set_model_paths
from hydra.utils import instantiate

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
    Generate counterfactuals using the CEM method.

    Filters the test set to exclude the target class, configures the CEM counterfactual
    generator, and produces counterfactuals for the selected instances.

    Args:
        cfg: Hydra configuration with experiment parameters
        dataset: Dataset object with train/test data and metadata
        gen_model: Trained generative model for plausibility threshold
        disc_model: Trained discriminative model used by CEM
        save_folder: Directory path for saving generated counterfactuals

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactuals.
            - Xs: Original instances used for counterfactual search.
            - ys_orig: Original labels for the selected instances.
            - ys_target: Target labels for the counterfactuals.
            - model_returned: Boolean mask indicating successful generations.
            - cf_search_time: Duration of the counterfactual search.
    """
    _ = gen_model  # Required by pipeline interface, not used directly in CEM search.
    cf_method_name = "CEM"
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating counterfactual model")
    cf_method = CEM_CF(
        disc_model=disc_model,
        mode=cfg.counterfactuals_params.mode,
        kappa=cfg.counterfactuals_params.kappa,
        beta=cfg.counterfactuals_params.beta,
        c_init=cfg.counterfactuals_params.c_init,
        c_steps=cfg.counterfactuals_params.c_steps,
        max_iterations=cfg.counterfactuals_params.max_iterations,
        learning_rate_init=cfg.counterfactuals_params.learning_rate_init,
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
    explanation_result = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        target_class=target_class,
    )

    cf_search_time = time() - time_start
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    Xs_cfs = np.asarray(explanation_result.x_cfs)
    Xs = np.asarray(explanation_result.x_origs)
    ys_orig = np.asarray(explanation_result.y_origs)
    ys_target = np.asarray(explanation_result.y_cf_targets)
    logs = explanation_result.logs or {}
    model_returned = np.asarray(logs.get("model_returned", np.ones(len(Xs_cfs), dtype=bool)))

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to {counterfactuals_path}")

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
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean mask for successful CF generations
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


@hydra.main(config_path="./conf", config_name="cem_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
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
