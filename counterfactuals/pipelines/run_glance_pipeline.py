import logging
import os
from time import time
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.metrics.metrics import evaluate_cf_for_rppcef
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
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Generate counterfactual explanations using the GLANCE-style method.

    This function instantiates the configured counterfactual method, computes a
    plausibility threshold using the generative model, prepares any method-specific
    state, and generates counterfactuals for eligible test instances.

    Args:
        cfg: Hydra configuration with experiment parameters
        dataset: Dataset object with train/test data and metadata
        gen_model: Trained generative model for plausibility thresholding
        disc_model: Trained discriminative model for predictions
        save_folder: Directory for saving generated counterfactuals CSV

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactuals
            - Xs: Original instances used for counterfactuals
            - log_prob_threshold: Computed log-probability threshold
            - ys_orig: Original labels
            - ys_target: Target labels for counterfactuals
            - model_returned: Boolean array indicating successful generation
            - cf_search_time: Average time taken for CF search
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = 1
    Xs = dataset.X_test[dataset.y_test != target_class]
    ys_orig = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating counterfactual model")
    cf_method = instantiate(
        cfg.counterfactuals_params.cf_method,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        features=dataset.features,
        model=disc_model,
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
    time_start = time()
    cf_method.prep(dataset.X_train, dataset.y_train)
    Xs_cfs = cf_method.explain()
    ys_target = np.abs(ys_orig - 1)
    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)
    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
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


@hydra.main(config_path="./conf", config_name="glance_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline function for GLANCE counterfactual generation and evaluation.

    Orchestrates a 5-fold cross-validation pipeline: loads dataset, creates
    discriminative and generative models, generates counterfactuals using the
    configured GLANCE-like method, and evaluates them via `evaluate_cf_for_rppcef`.

    Args:
        cfg: Hydra configuration including dataset, model, and CF parameters
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        logger.info(f"Processing fold {fold_n}")
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

        logger.info("Calculating metrics")
        metrics = evaluate_cf_for_rppcef(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
            X_test_target=Xs,
        )
        logger.info(f"Metrics:\n{metrics}")
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
