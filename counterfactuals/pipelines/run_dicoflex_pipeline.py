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

from counterfactuals.cf_methods.local.dicoflex import DiCoFlex, DiCoFlexParams
from counterfactuals.cf_methods.local.dicoflex.context_utils import (
    DiCoFlexGeneratorMetricsAdapter,
    build_context_matrix,
    get_numpy_pointer,
)
from counterfactuals.cf_methods.local.dicoflex.data import (
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
            optimizer.zero_grad()
            log_prob = model(
                batch_cf.to(device),
                context=batch_context.to(device),
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
                log_prob = model(
                    batch_cf.to(device),
                    context=batch_context.to(device),
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
            batch_scores = model(
                batch_cf.to(device),
                context=batch_context.to(device),
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


@hydra.main(config_path="./conf", config_name="dicoflex_config", version_base="1.2")
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
            ("onehot", OneHotEncodingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)

    use_cv = cfg.experiment.get("cv_folds", 0) and cfg.experiment.cv_folds > 1
    fold_iterator = dataset.get_cv_splits(cfg.experiment.cv_folds) if use_cv else [None]

    for fold_idx, _ in enumerate(fold_iterator):
        logger.info("Running DiCoFlex pipeline for fold %s", fold_idx)
        disc_model_path, gen_model_path, save_folder = set_model_paths(
            cfg, fold=fold_idx if use_cv else None
        )
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
            batch_size=cfg.counterfactuals_params.train_batch_size,
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
            train_loader, cfg.counterfactuals_params.train_batch_size
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
            continue

        cf_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(dataset.X_test[test_mask]).float(),
                torch.from_numpy(dataset.y_test[test_mask]).float(),
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
        model_returned_mask = np.array(
            explanation_result.logs.get("model_returned_mask", []), dtype=bool
        )
        if model_returned_mask.size == 0:
            model_returned_mask = np.ones(explanation_result.x_cfs.shape[0], dtype=bool)

        mask_vector = mask_vectors[params.mask_index]
        cf_contexts = build_context_matrix(
            factual_points=explanation_result.x_origs,
            labels=explanation_result.y_cf_targets,
            mask_vector=mask_vector,
            p_value=params.p_value,
            class_to_index=class_to_index,
        )
        test_contexts = build_context_matrix(
            factual_points=explanation_result.x_origs,
            labels=explanation_result.y_origs,
            mask_vector=mask_vector,
            p_value=params.p_value,
            class_to_index=class_to_index,
        )
        context_lookup = {
            get_numpy_pointer(explanation_result.x_cfs): cf_contexts,
            get_numpy_pointer(explanation_result.x_origs): test_contexts,
        }
        metrics_gen_model = DiCoFlexGeneratorMetricsAdapter(
            base_model=gen_model,
            context_lookup=context_lookup,
        )

        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        cf_path = os.path.join(
            save_folder,
            f"counterfactuals_DiCoFlex_{disc_model_name}.csv",
        )
        cf_original_space = dataset.inverse_transform(explanation_result.x_cfs.copy())
        pd.DataFrame(cf_original_space).to_csv(cf_path, index=False)
        logger.info("Saved counterfactuals to %s", cf_path)

        metrics = evaluate_cf(
            gen_model=metrics_gen_model,
            disc_model=disc_model,
            X_cf=explanation_result.x_cfs,
            model_returned=model_returned_mask,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=explanation_result.x_origs,
            y_test=explanation_result.y_origs,
            median_log_prob=log_prob_threshold,
            y_target=explanation_result.y_cf_targets,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_time
        metrics_path = os.path.join(save_folder, "cf_metrics_DiCoFlex.csv")
        df_metrics.to_csv(metrics_path, index=False)
        logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
