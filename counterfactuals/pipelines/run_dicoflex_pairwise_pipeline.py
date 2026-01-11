"""DiCoFlex pipeline with pairwise diversity metric.

Generates multiple counterfactuals per instance and computes min pairwise distance.
"""

import logging
import os
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
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    OneHotEncodingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)
from counterfactuals.preprocessing.base import PreprocessingContext

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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


def compute_pairwise_mean_distance(cfs: np.ndarray) -> float:
    """Average minimum pairwise distance across counterfactual sets.

    Args:
        cfs: Array of shape (n_instances, cfs_per_instance, n_features)

    Returns:
        Mean of minimum pairwise distances across all instances.
    """
    if cfs.size == 0 or cfs.shape[1] < 2:
        return float("nan")
    min_dists: list[float] = []
    for group in cfs:
        if group.shape[0] < 2:
            continue
        distances = pdist(group, metric="euclidean")
        if distances.size > 0:
            min_dists.append(float(distances.mean()))
    return float(np.mean(min_dists)) if min_dists else float("nan")


def run_fold(cfg: DictConfig, dataset: MethodDataset, device: str, fold_idx: int):
    """Run DiCoFlex pipeline for a single fold with pairwise diversity metric."""
    cf_per_instance = cfg.counterfactuals_params.cf_samples_per_factual
    logger.info(
        "Running DiCoFlex pairwise pipeline for fold %s with %d CFs per instance",
        fold_idx,
        cf_per_instance,
    )
    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_idx)
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
    )

    gen_model = instantiate_gen_model(cfg, dataset, context_dim, device)
    if cfg.gen_model.train_model:
        train_dicoflex_generator(
            gen_model,
            train_loader,
            val_loader,
            cfg,
            gen_model_path,
            device,
        )
    else:
        gen_model.load(gen_model_path)

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
        cf_samples_per_factual=cf_per_instance,
    )
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
    logger.info("CF search completed in %.4f seconds", cf_time)

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

    # Reshape to 3D for pairwise distance: (n_instances, cf_per_instance, n_features)
    n_instances = explanation_result.x_cfs.shape[0] // cf_per_instance
    x_cfs_raw = explanation_result.x_cfs.reshape(n_instances, cf_per_instance, -1)
    x_origs_raw = explanation_result.x_origs.reshape(n_instances, cf_per_instance, -1)

    # DiCE-style padding: keep valid CFs; if block empty, fill entire block with factual
    cleaned_blocks = []
    model_returned_blocks: list[bool] = []
    for idx in range(n_instances):
        cf_block = x_cfs_raw[idx]
        factual_block = np.repeat(
            x_origs_raw[idx : idx + 1, 0, :], cf_per_instance, axis=0
        )
        valid_mask = ~np.isnan(cf_block).any(axis=1)
        valid_rows = cf_block[valid_mask]
        if valid_rows.size == 0:
            cleaned_block = factual_block
            model_returned_blocks.append(False)
        else:
            cleaned_block = factual_block.copy()
            cleaned_block[: min(cf_per_instance, valid_rows.shape[0])] = valid_rows[
                :cf_per_instance
            ]
            model_returned_blocks.append(True)
        cleaned_blocks.append(cleaned_block)

    x_cfs_3d = np.stack(cleaned_blocks)
    model_returned_mask = np.repeat(
        np.array(model_returned_blocks, dtype=bool), cf_per_instance
    )
    # Decode one-hot categories for diversity calculation while keeping scaled numeric features
    decoded_for_diversity = x_cfs_3d
    onehot_step = dataset.preprocessing_pipeline.get_step("onehot")
    if onehot_step is not None:
        flat = x_cfs_3d.reshape(-1, x_cfs_3d.shape[-1])
        decode_context = PreprocessingContext(
            X_train=flat,
            categorical_indices=dataset.categorical_features_indices,
            continuous_indices=dataset.numerical_features_indices,
        )
        decoded_context = onehot_step.inverse_transform(decode_context)
        decoded_for_diversity = decoded_context.X_train.reshape(
            x_cfs_3d.shape[0], x_cfs_3d.shape[1], -1
        )

    # Extract first CF per instance for standard metrics
    x_cfs_first = x_cfs_3d[:, 0, :].copy()
    x_origs_first = x_origs_raw[:, 0, :].copy()
    y_origs_first = explanation_result.y_origs.reshape(n_instances, cf_per_instance)[
        :, 0
    ].copy()
    y_targets_first = explanation_result.y_cf_targets.reshape(
        n_instances, cf_per_instance
    )[:, 0].copy()
    model_returned_first = np.array(model_returned_blocks, dtype=bool)

    # Ensure arrays are contiguous for pointer-based context lookup
    x_cfs_first = np.ascontiguousarray(x_cfs_first, dtype=np.float32)
    x_origs_first = np.ascontiguousarray(x_origs_first, dtype=np.float32)

    mask_vector = mask_vectors[params.mask_index]
    cf_contexts = build_context_matrix(
        factual_points=x_origs_first,
        labels=y_targets_first,
        mask_vector=mask_vector,
        p_value=params.p_value,
        class_to_index=class_to_index,
    )
    test_contexts = build_context_matrix(
        factual_points=x_origs_first,
        labels=y_origs_first,
        mask_vector=mask_vector,
        p_value=params.p_value,
        class_to_index=class_to_index,
    )
    context_lookup = {
        get_numpy_pointer(x_cfs_first): cf_contexts,
        get_numpy_pointer(x_origs_first): test_contexts,
    }
    metrics_gen_model = DiCoFlexGeneratorMetricsAdapter(
        base_model=gen_model,
        context_lookup=context_lookup,
    )

    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    cf_path = os.path.join(
        save_folder,
        f"counterfactuals_DiCoFlexPairwise_{disc_model_name}.csv",
    )

    x_cfs_flat = x_cfs_3d.reshape(-1, x_cfs_3d.shape[-1])
    cf_original_space = dataset.inverse_transform(x_cfs_flat)
    pd.DataFrame(cf_original_space).to_csv(cf_path, index=False)
    logger.info("Saved all counterfactuals to %s", cf_path)

    # Calculate standard metrics on first CF only
    logger.info("Calculating standard metrics using first CF per instance...")
    metrics = evaluate_cf(
        gen_model=metrics_gen_model,
        disc_model=disc_model,
        X_cf=x_cfs_first,
        model_returned=model_returned_first,
        categorical_features=dataset.categorical_features_indices,
        continuous_features=dataset.numerical_features_indices,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=x_origs_first,
        y_test=y_origs_first.flatten(),
        median_log_prob=log_prob_threshold,
        y_target=y_targets_first,
        metrics_conf_path=cfg.counterfactuals_params.metrics_conf_path,
    )

    # Calculate pairwise min distance on all CFs (x_cfs_3d already computed above)
    logger.info(
        "Calculating pairwise minimum distance across %d CFs per instance...",
        cf_per_instance,
    )
    metrics["pairwise_mean_distance"] = compute_pairwise_mean_distance(
        decoded_for_diversity
    )
    logger.info("pairwise_mean_distance: %.6f", metrics["pairwise_mean_distance"])

    logger.info("Metrics:\n%s", metrics)

    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics["cf_search_time"] = cf_time
    df_metrics["cf_per_instance"] = cf_per_instance
    metrics_path = os.path.join(
        save_folder, f"cf_metrics_DiCoFlexPairwise_{disc_model_name}.csv"
    )
    df_metrics.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)


@hydra.main(config_path="./conf", config_name="dicoflex_config", version_base="1.2")
def main(cfg: DictConfig):
    """Run DiCoFlex pipeline with pairwise diversity metric."""
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
            ("onehot", OneHotEncodingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    for fold_idx, _ in enumerate(dataset.get_cv_splits(cfg.experiment.cv_folds)):
        run_fold(cfg, dataset, device, fold_idx)


if __name__ == "__main__":
    main()
