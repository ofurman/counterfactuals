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
from torch.utils.data import DataLoader, TensorDataset

from cel.cf_methods.local_methods.tabdce.data import (
    TabularCounterfactualDataset,
    TabularSpec,
)
from cel.cf_methods.local_methods.tabdce.denoise import TabularEpsModel
from cel.cf_methods.local_methods.tabdce.diffusion import (
    MixedTabularDiffusion,
)
from cel.cf_methods.local_methods.tabdce.tabdce import TabDCE
from cel.datasets.method_dataset import MethodDataset
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.full_pipeline.full_pipeline import full_pipeline
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def build_preprocessing_pipeline() -> PreprocessingPipeline:
    """Return a lightweight preprocessing pipeline for the tabular data."""
    return PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )


def prepare_tabular_dataset(
    dataset: MethodDataset, cfg: DictConfig, device: torch.device
) -> TabularCounterfactualDataset:
    """Create the training dataset used by the TabDCE diffusion model."""
    spec = TabularSpec(
        num_idx=list(dataset.numerical_features_indices),
        cat_idx=list(dataset.categorical_features_indices),
    )
    return TabularCounterfactualDataset(
        X=dataset.X_train,
        y=dataset.y_train,
        spec=spec,
        k=cfg.tabdce.k_neighbors,
        search_method=cfg.tabdce.search_method,
        device=device,
    )


def create_diffusion_model(
    tab_dataset: TabularCounterfactualDataset, cfg: DictConfig, device: torch.device
) -> MixedTabularDiffusion:
    """Instantiate the denoiser and diffusion components."""
    eps_model = TabularEpsModel(
        xdim=tab_dataset.X_model.shape[1],
        cat_dims=tab_dataset.cat_cardinalities,
        y_classes=int(tab_dataset.num_classes_target),
        hidden=cfg.tabdce.hidden_dim,
    )
    diffusion_model = MixedTabularDiffusion(
        denoise_fn=eps_model,
        num_numerical=tab_dataset.num_numerical,
        num_classes=tab_dataset.cat_cardinalities,
        T=cfg.tabdce.T,
        device=device,
    )
    return diffusion_model.to(device)


def train_tabdce_diffusion(
    model: MixedTabularDiffusion,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    model_path: Path,
) -> None:
    """Train the TabDCE diffusion model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        last_components = {"num": float("nan"), "cat": float("nan")}
        for batch in dataloader:
            x_neigh = batch["x_neigh"]
            x_orig = batch["x_orig"]
            y_target = batch["y_target"]

            optimizer.zero_grad()
            loss, components = model(x_neigh, x_orig, y_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            last_components = components

        epoch_loss /= max(1, len(dataloader))
        logger.info(
            "Epoch %d | loss %.4f | num_loss %.4f | cat_loss %.4f",
            epoch,
            epoch_loss,
            last_components.get("num", float("nan")),
            last_components.get("cat", float("nan")),
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            logger.info("Saved improved diffusion model to %s", model_path)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=model.betas.device))
        logger.info("Loaded best diffusion weights from %s", model_path)


def search_counterfactuals(
    cfg: DictConfig,
    dataset: MethodDataset,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate counterfactuals using the TabDCE method."""
    _ = gen_model, disc_model
    cf_method_name = "TabDCE"
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
        return (
            np.empty((0, dataset.X_test.shape[1])),
            np.empty((0, dataset.X_test.shape[1])),
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

    time_start = time()
    explanation_result = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        target_class=target_class,
    )
    cf_search_time = time() - time_start
    logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

    Xs_cfs = np.asarray(explanation_result.x_cfs)
    Xs = np.asarray(explanation_result.x_origs)
    ys_orig = np.asarray(explanation_result.y_origs)
    ys_target = np.asarray(explanation_result.y_cf_targets)
    model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

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
    """Calculate evaluation metrics for generated counterfactual explanations."""
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
    logger.info("Metrics:\n%s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run TabDCE with the standard full pipeline interface."""
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)
    preprocessing_pipeline = build_preprocessing_pipeline()
    full_pipeline(
        cfg, preprocessing_pipeline, logger, search_counterfactuals, calculate_metrics
    )


if __name__ == "__main__":
    main()
