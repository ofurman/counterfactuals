import logging
import os
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, TensorDataset

from cel.cf_methods.local_methods.tabdce.tabdce import TabDCE
from cel.datasets.method_dataset import MethodDataset
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.full_pipeline.full_pipeline import full_pipeline
from cel.pipelines.run_tabdce_pipeline import (
    build_preprocessing_pipeline,
    create_diffusion_model,
    prepare_tabular_dataset,
    train_tabdce_diffusion,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def compute_pairwise_mean_distance(cfs: np.ndarray) -> float:
    """Average pairwise distance across counterfactual sets."""
    if cfs.size == 0 or cfs.shape[1] < 2:
        return float("nan")
    mean_dists: list[float] = []
    for group in cfs:
        distances = pdist(group, metric="euclidean")
        if distances.size > 0:
            mean_dists.append(float(distances.mean()))
    return float(np.mean(mean_dists)) if mean_dists else float("nan")


def search_counterfactuals(
    cfg: DictConfig,
    dataset: MethodDataset,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Generate multiple counterfactuals per instance with TabDCE."""
    _ = gen_model, disc_model
    cf_method_name = "TabDCEPairwise"
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    use_gpu = torch.cuda.is_available() and cfg.tabdce.get("use_gpu", False)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info("Using device: %s", device)

    target_class = cfg.counterfactuals_params.target_class
    mask = dataset.y_test != target_class
    if not np.any(mask):
        logger.info("All samples already belong to the target class %s", target_class)
        empty = np.empty((0, dataset.X_test.shape[1]))
        return (
            (empty, empty),
            empty,
            np.array([]),
            np.array([]),
            np.array([], dtype=bool),
            0.0,
        )

    tab_dataset = prepare_tabular_dataset(dataset, cfg, device)
    train_loader = DataLoader(
        tab_dataset, batch_size=cfg.tabdce.batch_size, shuffle=True
    )
    diffusion_model = create_diffusion_model(tab_dataset, cfg, device)
    diffusion_path = Path(save_folder) / "tabdce_diffusion.pt"
    train_tabdce_diffusion(
        model=diffusion_model,
        dataloader=train_loader,
        epochs=cfg.tabdce.epochs,
        lr=cfg.tabdce.lr,
        model_path=diffusion_path,
    )

    cf_method = TabDCE(
        diffusion_model=diffusion_model,
        spec=tab_dataset.spec,
        qt=tab_dataset.qt,
        ohe=tab_dataset.ohe,
        device=device,
    )

    X_test_origin = dataset.X_test[mask]
    y_test_origin = dataset.y_test[mask]
    cf_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )

    cf_per_instance = int(cfg.counterfactuals_params.get("cf_samples_per_factual", 5))
    time_start = time()
    cf_samples: list[np.ndarray] = []
    x_origs = None
    y_origs = None
    y_targets = None
    for _ in range(cf_per_instance):
        explanation_result = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            target_class=target_class,
        )
        cf_samples.append(np.asarray(explanation_result.x_cfs))
        if x_origs is None:
            x_origs = np.asarray(explanation_result.x_origs)
            y_origs = np.asarray(explanation_result.y_origs)
            y_targets = np.asarray(explanation_result.y_cf_targets)
    cf_search_time = time() - time_start
    logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

    x_origs = x_origs if x_origs is not None else np.empty((0, dataset.X_test.shape[1]))
    y_origs = y_origs if y_origs is not None else np.array([])
    y_targets = y_targets if y_targets is not None else np.array([])

    Xs_cfs_all = np.stack(cf_samples, axis=1)
    Xs_cfs_first = Xs_cfs_all[:, 0, :]
    model_returned_first = np.ones(Xs_cfs_first.shape[0], dtype=bool)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs_all.reshape(-1, Xs_cfs_all.shape[-1])).to_csv(
        counterfactuals_path, index=False
    )
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    return (
        (Xs_cfs_first, Xs_cfs_all),
        x_origs,
        y_origs,
        y_targets,
        model_returned_first,
        cf_search_time,
    )


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray | tuple,
    model_returned: np.ndarray,
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: Optional[np.ndarray] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Calculate metrics using first CF only, then append pairwise distance."""
    Xs_cfs_first, Xs_cfs_all = Xs_cfs

    logger.info("Calculating standard metrics using first CF per instance...")
    metrics = evaluate_cf(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs_first,
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

    logger.info("Calculating pairwise distance across all CFs...")
    metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(Xs_cfs_all)
    logger.info("Metrics calculated: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run TabDCE pipeline with pairwise diversity metric."""
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)
    preprocessing_pipeline = build_preprocessing_pipeline()
    full_pipeline(
        cfg, preprocessing_pipeline, logger, search_counterfactuals, calculate_metrics
    )


if __name__ == "__main__":
    main()
