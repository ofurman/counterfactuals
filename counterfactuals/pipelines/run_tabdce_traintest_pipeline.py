"""TabDCE pipeline for pre-split train/test datasets, aligned with the
DiCE/CCHVAE/DiCoFlex train-test pipelines so all four methods can be scored
side by side under the actionability-constraint setups.

Differences from ``run_tabdce_pipeline.py``:
    * single train/test split (fold 0), not 5-fold CV;
    * shared-classifier contract (``set_model_paths`` + ``create_disc_model``),
      so ``disc_model.train_model=false`` reloads a classifier trained once;
    * MinMax + one-hot model space (``build_model_space_pipeline``), matching
      the baselines rather than TabDCE's own MinMax-only preprocessing;
    * M candidates per factual (paper protocol M=100), saved factual-major;
    * actionability projection: immutable features frozen to the factual value,
      numeric monotonic features clamped to their allowed direction. Categorical
      monotonic features fold to immutable (no direction enforceable on one-hot
      groups), matching how DiCE/CCHVAE treat them.

Usage:
    uv run python -m counterfactuals.pipelines.run_tabdce_traintest_pipeline \
        dataset.train_data_path=data/my_train.csv \
        dataset.test_data_path=data/my_test.csv
"""

import logging
import os
from pathlib import Path
from time import time
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.cf_methods.local_methods.tabdce.tabdce import TabDCE
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.dequantization.utils import DequantizationWrapper
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.factual_selection import (
    resolve_target_labels,
    select_factual_indices,
)
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.pipelines.nodes.seeding import set_global_seed
from counterfactuals.pipelines.run_tabdce_pipeline import (
    create_diffusion_model,
    prepare_tabular_dataset,
    train_tabdce_diffusion,
)
from counterfactuals.preprocessing import build_model_space_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def compute_pairwise_mean_distance(cfs: np.ndarray) -> float:
    """Average of within-instance mean pairwise (euclidean) CF distances."""
    if cfs.size == 0 or cfs.shape[1] < 2:
        return float("nan")
    mean_dists: list[float] = []
    for group in cfs:
        distances = pdist(group, metric="euclidean")
        if distances.size > 0:
            mean_dists.append(float(distances.mean()))
    return float(np.mean(mean_dists)) if mean_dists else float("nan")


def get_log_prob_threshold(
    gen_model: torch.nn.Module,
    dataset: MethodDataset,
    batch_size: int,
    log_prob_quantile: float,
) -> float:
    """Calculate log probability threshold from training data."""
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(batch_size=batch_size, shuffle=False)
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        log_prob_quantile,
    )
    logger.info("log_prob_threshold: %.4f", log_prob_threshold)
    return log_prob_threshold


def resolve_constraint_columns(
    cfg: DictConfig, dataset: MethodDataset
) -> Tuple[List[int], List[int], List[int]]:
    """Map the setup's actionability config onto model-space column indices.

    Returns (immutable_cols, increase_cols, decrease_cols). A feature marked
    non-actionable in the dataset yaml is frozen (immutable), EXCEPT numeric
    features named in ``monotonic_overrides``, which are instead clamped to the
    given direction. Categorical monotonic features stay frozen because TabDCE
    cannot enforce a direction on a one-hot group.
    """
    features = list(dataset.features)  # expanded names, one per model-space column
    actionable = set(getattr(dataset, "actionable_features", features))
    frozen_cols = [i for i, name in enumerate(features) if name not in actionable]

    num_names = list(dataset.numerical_features)
    num_idx = list(dataset.numerical_features_indices)
    name_to_col = dict(zip(num_names, num_idx))

    overrides = dict(cfg.counterfactuals_params.get("monotonic_overrides") or {})
    increase_cols: list[int] = []
    decrease_cols: list[int] = []
    for name, direction in overrides.items():
        col = name_to_col.get(name)
        if col is None:
            continue  # categorical override -> folds to immutable (stays frozen)
        if str(direction).upper() == "INCREASE":
            increase_cols.append(col)
        elif str(direction).upper() == "DECREASE":
            decrease_cols.append(col)

    directional = set(increase_cols) | set(decrease_cols)
    immutable_cols = [c for c in frozen_cols if c not in directional]
    logger.info(
        "Constraints: %d immutable cols, %d increase, %d decrease",
        len(immutable_cols),
        len(increase_cols),
        len(decrease_cols),
    )
    return immutable_cols, increase_cols, decrease_cols


def apply_constraints(
    cf: np.ndarray,
    factual: np.ndarray,
    immutable_cols: List[int],
    increase_cols: List[int],
    decrease_cols: List[int],
) -> np.ndarray:
    """Project counterfactuals onto the actionable set (row-aligned to factual)."""
    cf = cf.copy()
    if immutable_cols:
        cf[:, immutable_cols] = factual[:, immutable_cols]
    if increase_cols:
        cf[:, increase_cols] = np.maximum(cf[:, increase_cols], factual[:, increase_cols])
    if decrease_cols:
        cf[:, decrease_cols] = np.minimum(cf[:, decrease_cols], factual[:, decrease_cols])
    return cf


def search_counterfactuals(
    cfg: DictConfig,
    dataset: MethodDataset,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Train the TabDCE diffusion model and generate M constrained candidates."""
    _ = gen_model, disc_model
    cf_method_name = "TabDCE"
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    device = torch.device("cpu")  # host has no CUDA; MPS unsupported by the sampler

    target_class = cfg.counterfactuals_params.target_class
    factual_indices = select_factual_indices(
        dataset.y_test,
        target_class=target_class,
        n_test_samples=cfg.counterfactuals_params.get("n_test_samples", None),
        seed=cfg.experiment.get("seed", 42),
    )
    X_test_origin = dataset.X_test[factual_indices].astype(np.float64)
    y_test_origin = dataset.y_test[factual_indices].astype(np.float64)
    n_factual = X_test_origin.shape[0]
    n_features = X_test_origin.shape[1]
    m_candidates = int(cfg.counterfactuals_params.get("cf_samples_per_factual", 100))

    logger.info("Training TabDCE diffusion model")
    tab_dataset = prepare_tabular_dataset(dataset, cfg, device)
    train_loader = DataLoader(tab_dataset, batch_size=cfg.tabdce.batch_size, shuffle=True)
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

    immutable_cols, increase_cols, decrease_cols = resolve_constraint_columns(cfg, dataset)

    # Factual-major replication: rows [f0]*M, [f1]*M, ...
    factual_rep = np.repeat(X_test_origin, m_candidates, axis=0)
    y_target_rep = np.full(factual_rep.shape[0], int(target_class), dtype=np.float32)
    gen_loader = DataLoader(
        TensorDataset(
            torch.tensor(factual_rep).float(),
            torch.tensor(y_target_rep).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )

    logger.info(
        "Generating %d candidates for each of %d factuals (%d samples)",
        m_candidates,
        n_factual,
        factual_rep.shape[0],
    )
    time_start = time()
    result = cf_method.explain_dataloader(dataloader=gen_loader, target_class=target_class)
    cf_search_time = time() - time_start
    logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

    cf_all = np.asarray(result.x_cfs, dtype=np.float64)
    cf_all = apply_constraints(cf_all, factual_rep, immutable_cols, increase_cols, decrease_cols)

    # Sanity: immutable columns must match the factual exactly after projection.
    if immutable_cols:
        assert np.allclose(cf_all[:, immutable_cols], factual_rep[:, immutable_cols]), (
            "Immutable columns changed after projection"
        )
    if increase_cols:
        assert np.all(cf_all[:, increase_cols] >= factual_rep[:, increase_cols] - 1e-6), (
            "Increase columns dropped below factual after projection"
        )

    Xs_cfs_all = cf_all.reshape(n_factual, m_candidates, n_features)
    Xs_cfs_first = Xs_cfs_all[:, 0, :]
    model_returned_first = np.ones(n_factual, dtype=bool)
    ys_target = resolve_target_labels(y_test_origin, target_class)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(cf_all).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    factuals_path = os.path.join(save_folder, f"factuals_{cf_method_name}_{disc_model_name}.csv")
    pd.DataFrame(X_test_origin).to_csv(factuals_path, index=False)
    logger.info("Factuals saved to %s", factuals_path)

    return (
        (Xs_cfs_first, Xs_cfs_all),
        X_test_origin,
        y_test_origin,
        ys_target,
        model_returned_first,
        cf_search_time,
    )


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: tuple,
    model_returned: np.ndarray,
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: np.ndarray | None = None,
    metrics_conf_path: str | None = None,
    **_: Any,
) -> Dict[str, Any]:
    """First-CF metrics via evaluate_cf, plus pairwise diversity over all CFs."""
    Xs_cfs_first, Xs_cfs_all = Xs_cfs
    logger.info("Calculating standard metrics using first CF per instance")
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
        metrics_conf_path=metrics_conf_path
        or "counterfactuals/pipelines/conf/metrics/default.yaml",
    )
    metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(Xs_cfs_all)
    logger.info("Metrics calculated: %s", metrics)
    return metrics


def run_pipeline(cfg: DictConfig, dataset: MethodDataset) -> None:
    """Run the TabDCE pipeline on a single train-test split."""
    logger.info("Running TabDCE pipeline on train-test split")

    dequantizer = GroupDequantizer(dataset.categorical_features_lists)
    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=0)

    disc_train_start = time()
    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)
    disc_train_time = time() - disc_train_start

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    dequantizer.fit(dataset.X_train)
    gen_train_start = time()
    gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer)
    gen_train_time = time() - gen_train_start

    dataset.X_train = dequantizer.transform(dataset.X_train)
    log_prob_threshold = get_log_prob_threshold(
        gen_model,
        dataset,
        cfg.counterfactuals_params.batch_size,
        cfg.counterfactuals_params.log_prob_quantile,
    )
    dataset.X_train = dequantizer.inverse_transform(dataset.X_train)

    Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time = search_counterfactuals(
        cfg, dataset, gen_model, disc_model, save_folder
    )

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

    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics["cf_search_time"] = cf_search_time
    df_metrics["disc_train_time"] = disc_train_time
    df_metrics["gen_train_time"] = gen_train_time
    df_metrics["seed"] = cfg.experiment.get("seed", 42)
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    metrics_path = os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv")
    df_metrics.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)


@hydra.main(config_path="./conf", config_name="tabdce_traintest_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run the TabDCE pipeline on pre-split train/test data."""
    set_global_seed(cfg.experiment.get("seed", 42))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    preprocessing_pipeline = build_model_space_pipeline(
        cfg.experiment.get("model_space_scaler", "minmax")
    )
    file_dataset = instantiate(cfg.dataset)
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    run_pipeline(cfg, dataset)


if __name__ == "__main__":
    main()
