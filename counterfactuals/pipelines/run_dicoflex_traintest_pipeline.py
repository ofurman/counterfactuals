"""DiCoFlex pipeline for pre-split train/test datasets.

This script runs the DiCoFlex counterfactual generation pipeline on datasets
where train and test splits are provided as separate files, rather than using
cross-validation.

Usage:
    uv run python -m counterfactuals.pipelines.run_dicoflex_traintest_pipeline \
        dataset.train_data_path=data/my_train.csv \
        dataset.test_data_path=data/my_test.csv
"""

import logging
import os
from pathlib import Path
from time import time
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.spatial.distance import pdist

from counterfactuals.cf_methods.local_methods.dicoflex import DiCoFlex, DiCoFlexParams
from counterfactuals.cf_methods.local_methods.dicoflex.context_utils import (
    DiCoFlexGeneratorMetricsAdapter,
    build_context_matrix,
    get_numpy_pointer,
)
from counterfactuals.cf_methods.local_methods.dicoflex.data import (
    build_actionability_mask,
    create_dicoflex_dataloaders,
)
from counterfactuals.cf_methods.local_methods.dicoflex.visualization import (
    visualize_counterfactual_samples,
    visualize_flow_contours,
    visualize_training_batch,
)
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


def build_masks(dataset: MethodDataset, cfg: DictConfig) -> List[np.ndarray]:
    """Assemble the mask catalogue used during DiCoFlex training."""
    masks: List[np.ndarray] = []
    if cfg.use_actionability_mask:
        masks.append(build_actionability_mask(dataset))
    for custom_mask in cfg.get("custom_masks", []):
        mask_vec = np.asarray(custom_mask, dtype=np.float32).reshape(-1)
        if mask_vec.shape[0] != dataset.X_train.shape[1]:
            raise ValueError(
                "Custom mask length does not match the feature dimension after preprocessing."
            )
        masks.append(mask_vec)
    if not masks:
        masks.append(np.ones(dataset.X_train.shape[1], dtype=np.float32))
    return masks


def instantiate_gen_model(
    cfg: DictConfig, dataset: MethodDataset, context_dim: int, device: str
):
    """Instantiate the conditional flow used by DiCoFlex."""
    model = instantiate(
        cfg.gen_model.model,
        features=dataset.X_train.shape[1],
        context_features=context_dim,
    )
    return model.to(device)


def train_dicoflex_generator(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    model_path: str,
    device: str,
) -> None:
    """Train the flow model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gen_model.lr)
    best_val = float("inf")
    patience_counter = 0
    eps = cfg.gen_model.get("eps", 1e-5)

    for epoch in range(cfg.gen_model.epochs):
        model.train()
        train_loss = 0.0
        for batch_cf, batch_context in train_loader:
            batch_cf = batch_cf.reshape(-1, batch_cf.shape[-1]).to(device)
            batch_context = batch_context.reshape(-1, batch_context.shape[-1]).to(
                device
            )
            optimizer.zero_grad()
            log_prob = model(
                batch_cf,
                context=batch_context,
            )
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_cf, batch_context in val_loader:
                batch_cf = batch_cf.reshape(-1, batch_cf.shape[-1]).to(device)
                batch_context = batch_context.reshape(-1, batch_context.shape[-1]).to(
                    device
                )
                log_prob = model(
                    batch_cf,
                    context=batch_context,
                )
                val_loss += (-log_prob.mean()).item()
        val_loss /= max(1, len(val_loader))
        logger.info(
            "Epoch %s | train loss %.4f | val loss %.4f",
            epoch,
            train_loss,
            val_loss,
        )

        if val_loss < best_val - eps:
            best_val = val_loss
            patience_counter = 0
            model.save(model_path)
        else:
            patience_counter += 1
            if patience_counter > cfg.gen_model.patience:
                logger.info("Early stopping after %s epochs", epoch + 1)
                break

    model.load(model_path)


def compute_log_prob_threshold(
    model,
    dataloader: torch.utils.data.DataLoader,
    quantile: float,
    device: str,
) -> float:
    """Estimate a plausibility threshold based on training log probabilities."""
    log_probs = []
    model.eval()
    with torch.no_grad():
        for batch_cf, batch_context in dataloader:
            batch_cf = batch_cf.reshape(-1, batch_cf.shape[-1]).to(device)
            batch_context = batch_context.reshape(-1, batch_context.shape[-1]).to(
                device
            )
            batch_scores = model(
                batch_cf,
                context=batch_context,
            )
            log_probs.append(batch_scores.cpu())
    concat = torch.cat(log_probs)
    return torch.quantile(concat, quantile).item()


def get_full_training_loader(
    subset_loader: torch.utils.data.DataLoader, batch_size: int
) -> torch.utils.data.DataLoader:
    """Create a loader that iterates over the complete DiCoFlex dataset."""
    base_dataset = (
        subset_loader.dataset.dataset
        if hasattr(subset_loader.dataset, "dataset")
        else subset_loader.dataset
    )
    return torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def compute_feature_bounds(
    data: np.ndarray, padding: float = 0.05
) -> List[tuple[float, float]]:
    """Return padded min/max bounds for the first two features."""
    if data.shape[1] < 2:
        raise ValueError("At least two features are required for contour plots.")
    subset = data[:, :2]
    mins = subset.min(axis=0)
    maxs = subset.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-6)
    bounds: List[tuple[float, float]] = []
    for mn, mx, rng in zip(mins, maxs, ranges):
        pad = rng * padding
        bounds.append((mn - pad, mx + pad))
    return bounds


def compute_pairwise_mean_distance(samples: np.ndarray, group_ids: np.ndarray) -> float:
    """Average minimum pairwise distance across counterfactual groups."""
    if samples.size == 0 or group_ids.size == 0:
        return float("nan")

    min_dists: list[float] = []
    for gid in np.unique(group_ids):
        group_points = samples[group_ids == gid]
        group_points = group_points[~np.isnan(group_points).any(axis=1)]
        if group_points.shape[0] < 2:
            continue
        distances = pdist(group_points, metric="euclidean")
        if distances.size > 0:
            min_dists.append(float(distances.min()))

    return float(np.mean(min_dists)) if min_dists else float("nan")


def run_pipeline(cfg: DictConfig, dataset: MethodDataset, device: str):
    """Run the DiCoFlex pipeline on a single train-test split."""
    logger.info("Running DiCoFlex pipeline on train-test split")
    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=0)
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    if cfg.experiment.relabel_with_disc_model:
        cf_gen_model_filename = (
            f"gen_model_{gen_model_name}_dicoflex_relabeled_by_{disc_model_name}.pt"
        )
    else:
        cf_gen_model_filename = f"gen_model_{gen_model_name}_dicoflex.pt"
    cf_gen_model_path = os.path.join(save_folder, cf_gen_model_filename)
    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)
    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    masks = build_masks(dataset, cfg.counterfactuals_params)
    (
        train_loader,
        val_loader,
        class_to_index,
        mask_vectors,
        context_dim,
    ) = create_dicoflex_dataloaders(
        dataset.X_train,
        dataset.y_train,
        masks=masks,
        p_values=list(cfg.counterfactuals_params.p_values),
        n_neighbors=cfg.counterfactuals_params.n_neighbors,
        noise_level=cfg.counterfactuals_params.noise_level,
        factual_batch_size=cfg.counterfactuals_params.train_batch_factuals,
        val_ratio=cfg.counterfactuals_params.val_ratio,
        seed=cfg.experiment.seed,
        numerical_indices=dataset.numerical_features_indices,
        categorical_indices=dataset.categorical_features_indices,
        factual_chunk_size=cfg.counterfactuals_params.get(
            "neighbor_factual_chunk_size"
        ),
        target_chunk_size=cfg.counterfactuals_params.get("neighbor_target_chunk_size"),
    )
    vis_cfg = cfg.get("visualization")
    if vis_cfg and vis_cfg.get("enable_training_batch", False):
        try:
            batch_cf, batch_context = next(iter(train_loader))
        except StopIteration:
            logger.warning("No batches available for visualization.")
        else:
            flat_cf = batch_cf.reshape(-1, batch_cf.shape[-1])
            flat_context = batch_context.reshape(-1, batch_context.shape[-1])
            visualize_training_batch(
                batch_cf=flat_cf.cpu(),
                batch_context=flat_context.cpu(),
                feature_names=dataset.features,
                save_path=Path(save_folder) / "training_batch_neighbors.png",
                max_points=vis_cfg.get("training_batch_max_points", 200),
                dataset_points=dataset.X_train[:, :2],
            )
    gen_model = instantiate_gen_model(cfg, dataset, context_dim, device)
    if cfg.gen_model.train_model:
        train_dicoflex_generator(
            gen_model,
            train_loader,
            val_loader,
            cfg,
            cf_gen_model_path,
            device,
        )
    else:
        gen_model.load(cf_gen_model_path)

    full_loader = get_full_training_loader(
        train_loader, cfg.counterfactuals_params.train_batch_factuals
    )
    log_prob_threshold = compute_log_prob_threshold(
        gen_model,
        full_loader,
        cfg.counterfactuals_params.log_prob_quantile,
        device,
    )

    params = DiCoFlexParams(
        mask_index=cfg.counterfactuals_params.inference_mask_index,
        p_value=cfg.counterfactuals_params.inference_p_value,
        num_counterfactuals=cfg.counterfactuals_params.num_counterfactuals,
        target_class=cfg.counterfactuals_params.target_class,
        sampling_batch_size=cfg.counterfactuals_params.sampling_batch_size,
        cf_samples_per_factual=cfg.counterfactuals_params.cf_samples_per_factual,
    )
    mask_vector = mask_vectors[params.mask_index]
    cf_method = DiCoFlex(
        gen_model=gen_model,
        disc_model=disc_model,
        class_to_index=class_to_index,
        mask_vectors=mask_vectors,
        params=params,
        device=device,
    )

    target_class = cfg.counterfactuals_params.target_class
    test_mask = dataset.y_test != target_class
    if not np.any(test_mask):
        logger.warning("All test samples already belong to the target class.")
        return
    filtered_X_test = dataset.X_test[test_mask]
    filtered_y_test = dataset.y_test[test_mask]
    cf_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(filtered_X_test).float(),
            torch.from_numpy(filtered_y_test).float(),
        ),
        batch_size=cfg.counterfactuals_params.sampling_batch_size,
        shuffle=False,
    )
    start_time = time()
    explanation_result = cf_method.explain_dataloader(
        cf_loader,
        epochs=0,
        lr=0.0,
    )
    cf_time = time() - start_time
    explanation_result.x_cfs = np.ascontiguousarray(
        explanation_result.x_cfs.astype(np.float32, copy=False)
    )
    explanation_result.x_origs = np.ascontiguousarray(
        explanation_result.x_origs.astype(np.float32, copy=False)
    )
    cf_group_ids = (
        None
        if explanation_result.cf_group_ids is None
        else np.asarray(explanation_result.cf_group_ids, dtype=int)
    )
    model_returned_mask = np.array(
        explanation_result.logs.get("model_returned_mask", []), dtype=bool
    )
    if model_returned_mask.size == 0:
        model_returned_mask = np.ones(explanation_result.x_cfs.shape[0], dtype=bool)

    # Replace NaN rows (where counterfactuals couldn't be found) with original factuals
    x_cfs_cleaned = explanation_result.x_cfs.copy()
    nan_rows = np.any(np.isnan(x_cfs_cleaned), axis=1)
    if np.any(nan_rows):
        logger.info(
            "Replacing %d rows with NaN (failed counterfactuals) with original factuals",
            np.sum(nan_rows),
        )
        x_cfs_cleaned[nan_rows] = explanation_result.x_origs[nan_rows]

    # Handle multiple CFs per instance: extract first CF for metrics
    cf_per_instance = params.cf_samples_per_factual
    if cf_per_instance > 1:
        n_instances = x_cfs_cleaned.shape[0] // cf_per_instance
        x_cfs_3d = x_cfs_cleaned.reshape(n_instances, cf_per_instance, -1)
        x_cfs_for_metrics = x_cfs_3d[:, 0, :].copy()
        x_origs_for_metrics = explanation_result.x_origs[::cf_per_instance].copy()
        y_origs_for_metrics = explanation_result.y_origs[::cf_per_instance].copy()
        y_targets_for_metrics = explanation_result.y_cf_targets[
            ::cf_per_instance
        ].copy()
        model_returned_for_metrics = model_returned_mask[::cf_per_instance].copy()
        cf_group_ids_for_metrics = (
            cf_group_ids[::cf_per_instance] if cf_group_ids is not None else None
        )
    else:
        x_cfs_for_metrics = x_cfs_cleaned
        x_origs_for_metrics = explanation_result.x_origs
        y_origs_for_metrics = explanation_result.y_origs
        y_targets_for_metrics = explanation_result.y_cf_targets
        model_returned_for_metrics = model_returned_mask
        cf_group_ids_for_metrics = cf_group_ids

    # Ensure arrays are contiguous for pointer-based context lookup
    x_cfs_for_metrics = np.ascontiguousarray(x_cfs_for_metrics, dtype=np.float32)
    x_origs_for_metrics = np.ascontiguousarray(x_origs_for_metrics, dtype=np.float32)

    mask_vector = mask_vectors[params.mask_index]
    cf_contexts = build_context_matrix(
        factual_points=x_origs_for_metrics,
        labels=y_targets_for_metrics,
        mask_vector=mask_vector,
        p_value=params.p_value,
        class_to_index=class_to_index,
    )
    test_contexts = build_context_matrix(
        factual_points=x_origs_for_metrics,
        labels=y_origs_for_metrics,
        mask_vector=mask_vector,
        p_value=params.p_value,
        class_to_index=class_to_index,
    )
    context_lookup = {
        get_numpy_pointer(x_cfs_for_metrics): cf_contexts,
        get_numpy_pointer(x_origs_for_metrics): test_contexts,
    }
    metrics_gen_model = DiCoFlexGeneratorMetricsAdapter(
        base_model=gen_model,
        context_lookup=context_lookup,
    )
    if vis_cfg and vis_cfg.get("enable_cf_scatter", False):
        try:
            visualize_counterfactual_samples(
                factual_points=explanation_result.x_origs[:, :2],
                counterfactual_points=x_cfs_cleaned[:, :2],
                feature_names=dataset.features,
                save_path=Path(save_folder) / "counterfactuals_scatter.png",
                dataset_points=dataset.X_train[:, :2],
                max_points=vis_cfg.get("cf_scatter_max_points", 200),
            )
        except ValueError as exc:
            logger.info("Skipping counterfactual scatter plot: %s", exc)
    if vis_cfg and vis_cfg.get("enable_flow_contour", False):
        try:
            bounds = compute_feature_bounds(
                dataset.X_train, vis_cfg.get("contour_padding", 0.05)
            )
            factual_idx = min(
                vis_cfg.get("contour_factual_index", 0),
                filtered_X_test.shape[0] - 1,
            )
            factual_point = filtered_X_test[factual_idx]
            visualize_flow_contours(
                gen_model=gen_model,
                factual_point=factual_point,
                target_label=target_class,
                mask_vector=mask_vector,
                p_value=params.p_value,
                class_to_index=class_to_index,
                feature_bounds=bounds,
                save_path=Path(save_folder) / "flow_logprob_contour.png",
                feature_names=dataset.features,
                grid_size=vis_cfg.get("contour_grid_size", 200),
                device=device,
                dataset_points=dataset.X_train[:, :2],
            )
        except ValueError as exc:
            logger.info("Skipping flow contour plot: %s", exc)

    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    cf_path = os.path.join(
        save_folder,
        f"counterfactuals_DiCoFlex_{disc_model_name}.csv",
    )

    cf_original_space = dataset.inverse_transform(x_cfs_cleaned)
    pd.DataFrame(cf_original_space).to_csv(cf_path, index=False)
    logger.info("Saved counterfactuals to %s", cf_path)

    metrics = evaluate_cf(
        gen_model=metrics_gen_model,
        disc_model=disc_model,
        X_cf=x_cfs_for_metrics,
        model_returned=model_returned_for_metrics,
        categorical_features=dataset.categorical_features_indices,
        continuous_features=dataset.numerical_features_indices,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=x_origs_for_metrics,
        y_test=y_origs_for_metrics,
        median_log_prob=log_prob_threshold,
        y_target=y_targets_for_metrics,
        cf_group_ids=cf_group_ids_for_metrics,
        metrics_conf_path=cfg.counterfactuals_params.metrics_conf_path,
    )
    if cf_group_ids is not None:
        if cf_group_ids.shape[0] != x_cfs_cleaned.shape[0]:
            logger.warning(
                "Skipping pairwise_mean_distance: %s cf_group_ids for %s counterfactuals",
                cf_group_ids.shape[0],
                x_cfs_cleaned.shape[0],
            )
        else:
            metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(
                x_cfs_cleaned, cf_group_ids
            )
    logger.info(f"Metrics:\n{metrics}")

    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics["cf_search_time"] = cf_time
    metrics_path = os.path.join(
        save_folder, f"cf_metrics_DiCoFlex_{disc_model_name}.csv"
    )
    df_metrics.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)


@hydra.main(
    config_path="./conf", config_name="dicoflex_traintest_config", version_base="1.2"
)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.experiment.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = (
        "cuda"
        if torch.cuda.is_available() and cfg.experiment.get("use_gpu", False)
        else "cpu"
    )

    file_dataset = instantiate(cfg.dataset)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    run_pipeline(cfg, dataset, device)


if __name__ == "__main__":
    main()
