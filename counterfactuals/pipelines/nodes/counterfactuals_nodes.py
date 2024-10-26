import logging
import os
import numpy as np
import pandas as pd
from time import time
import torch
import neptune
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.utils

from counterfactuals.datasets.base import AbstractDataset

logger = logging.getLogger(__name__)


def create_counterfactual_method(
    cfg: DictConfig,
    disc_model: torch.nn.Module,
    gen_model: torch.nn.Module,
    run: neptune.Run,
    device: str = None,
) -> torch.nn.Module:
    """
    Create a counterfactual method
    """
    cf_method = instantiate(
        cfg.counterfactuals_params.cf_method,
        gen_model=gen_model,
        disc_model=disc_model,
        device=device,
        neptune_run=run,
    )
    return cf_method


def calculate_log_prob_threshold(gen_model, dataset, cfg, run):
    """
    Calculate log_prob_threshold
    """
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    run["parameters/log_prob_threshold"] = log_prob_threshold
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
    return log_prob_threshold


def search_counterfactuals(
    cfg: DictConfig,
    cf_method: torch.nn.Module,
    X_test_origin: np.ndarray,
    y_test_origin: np.ndarray,
    log_prob_threshold: float,
    run: neptune.Run,
) -> torch.nn.Module:
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
    Xs_cfs, Xs, ys_orig, ys_target, _ = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        patience=cfg.counterfactuals_params.patience,
        alpha=cfg.counterfactuals_params.alpha,
        alpha_s=cfg.counterfactuals_params.alpha_s,
        alpha_k=cfg.counterfactuals_params.alpha_k,
        beta=cfg.counterfactuals_params.beta,
        log_prob_threshold=log_prob_threshold,
    )
    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time
    return Xs_cfs, Xs, ys_orig, ys_target


def create_counterfactuals(
    cfg: DictConfig,
    dataset: AbstractDataset,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    run: neptune.Run,
    save_folder: str,
) -> torch.nn.Module:
    """
    Create a counterfactual model
    """

    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    _, _, X_test_target, y_test_target = dataset.get_target_class_splits(target_class)
    _, _, X_test_origin, y_test_origin = dataset.get_non_target_class_splits(
        target_class
    )

    logger.info("Creating counterfactual model")
    cf_method = create_counterfactual_method(
        cfg=cfg,
        disc_model=disc_model,
        gen_model=gen_model,
        run=run,
        device=cfg.device or "cpu",
    )
    log_prob_threshold = calculate_log_prob_threshold(gen_model, dataset, cfg, run)

    Xs_cfs, Xs, ys_orig, ys_target = search_counterfactuals(
        cfg=cfg,
        cf_method=cf_method,
        X_test_origin=X_test_origin,
        y_test_origin=y_test_origin,
        log_prob_threshold=log_prob_threshold,
        run=run,
    )
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )
    # M, S, D = deltas[0].get_matrices()

    # Xs_cfs = Xs + deltas[0]().detach().numpy()
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)
    return Xs_cfs, log_prob_threshold, X_test_target
