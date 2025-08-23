import logging
import os
from time import time
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.metrics.metrics import evaluate_cf_for_rppcef
from counterfactuals.cf_methods.group_ppcef import RPPCEF
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
) -> Tuple[np.ndarray, np.ndarray, float, torch.Tensor, np.ndarray, np.ndarray, float]:
    """Generate counterfactuals using the RPPCEF method.

    Builds the RPPCEF counterfactual searcher, computes a plausibility threshold
    via the generative model, runs batched search, and saves generated CFs.

    Args:
        cfg: Hydra configuration with method and experiment parameters.
        dataset: Dataset object with train/test arrays and metadata.
        gen_model: Trained generative model to compute log-prob threshold.
        disc_model: Trained discriminative model used in the search loss.
        save_folder: Output directory to persist artifacts.

    Returns:
        Tuple of:
        - Xs_cfs: Counterfactuals (original + learned delta)
        - Xs: Original inputs used for CFs
        - log_prob_threshold: Quantile-based plausibility threshold
        - S: Assignment/selection matrix from the RPPCEF delta module
        - ys_orig: Original labels
        - ys_target: Target labels for CFs
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
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_model_criterion)

    cf_method: RPPCEF = instantiate(
        cfg.counterfactuals_params.cf_method,
        X=X_test_origin,
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
    delta, Xs, ys_orig, ys_target = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        patience=cfg.counterfactuals_params.patience,
        alpha=cfg.counterfactuals_params.alpha,
        alpha_s=cfg.counterfactuals_params.alpha_s,
        alpha_k=cfg.counterfactuals_params.alpha_k,
        # beta=cfg.counterfactuals_params.beta,
        log_prob_threshold=log_prob_threshold,
    )

    cf_search_time = np.mean(time() - time_start)
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )
    M, S, D = delta.get_matrices()

    Xs_cfs = Xs + delta().detach().numpy()
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)
    return Xs_cfs, Xs, log_prob_threshold, S, ys_orig, ys_target, cf_search_time


@hydra.main(config_path="./conf", config_name="rppcef_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """RPPCEF pipeline: generate and evaluate grouped counterfactuals.

    Executes a 5-fold CV workflow: loads dataset, prepares models, runs the
    RPPCEF CF search on non-target-class samples, evaluates results using
    ``evaluate_cf_for_rppcef``, and saves both CFs and metrics locally.
    Neptune integration has been removed.

    Args:
        cfg: Hydra configuration with dataset, model, and CF method parameters.
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
            S,
            ys_orig,
            ys_target,
            cf_search_time,
        ) = search_counterfactuals(cfg, dataset, gen_model, disc_model, save_folder)

        logger.info("Calculating metrics")
        metrics = evaluate_cf_for_rppcef(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
            S_matrix=S.detach().numpy(),
            X_test_target=Xs,
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
