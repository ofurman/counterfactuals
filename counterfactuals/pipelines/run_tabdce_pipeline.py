import logging
import os
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.cf_methods.local_methods.tabdce.data import (
    TabularCounterfactualDataset,
    TabularSpec,
)
from counterfactuals.cf_methods.local_methods.tabdce.denoise import TabularEpsModel
from counterfactuals.cf_methods.local_methods.tabdce.diffusion import (
    MixedTabularDiffusion,
)
from counterfactuals.cf_methods.local_methods.tabdce.tabdce import TabDCE
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
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


def generate_counterfactuals(
    cf_method: TabDCE,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    target_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run counterfactual generation on the provided split."""
    mask = y != target_class
    if not np.any(mask):
        logger.warning(
            "All samples already belong to the target class %s", target_class
        )
        return (
            np.empty((0, X.shape[1])),
            np.empty((0, X.shape[1])),
            np.array([]),
            np.array([]),
        )

    filtered_X = X[mask]
    filtered_y = y[mask]
    dataloader = DataLoader(
        TensorDataset(
            torch.from_numpy(filtered_X).float(),
            torch.from_numpy(filtered_y).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    result = cf_method.explain_dataloader(
        dataloader=dataloader,
        target_class=target_class,
    )
    return result.x_cfs, result.x_origs, result.y_cf_targets, result.y_origs


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Train TabDCE diffusion and generate counterfactuals."""
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)

    use_gpu = torch.cuda.is_available() and cfg.tabdce.get("use_gpu", False)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info("Using device: %s", device)

    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)
    file_dataset = instantiate(cfg.dataset)
    preprocessing_pipeline = build_preprocessing_pipeline()
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)

    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)
    if cfg.experiment.get("relabel_with_disc_model", False):
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    tab_dataset = prepare_tabular_dataset(dataset, cfg, device)
    train_loader = DataLoader(
        tab_dataset,
        batch_size=cfg.tabdce.batch_size,
        shuffle=True,
    )
    diffusion_model = create_diffusion_model(tab_dataset, cfg, device)
    train_tabdce_diffusion(
        model=diffusion_model,
        dataloader=train_loader,
        epochs=cfg.tabdce.epochs,
        lr=cfg.tabdce.lr,
        model_path=Path(gen_model_path),
    )

    cf_method = TabDCE(
        diffusion_model=diffusion_model,
        spec=tab_dataset.spec,
        qt=tab_dataset.qt,
        ohe=tab_dataset.ohe,
        device=device,
    )

    x_cfs, x_origs, y_targets, y_origs = generate_counterfactuals(
        cf_method=cf_method,
        X=dataset.X_test,
        y=dataset.y_test,
        batch_size=cfg.counterfactuals_params.batch_size,
        target_class=cfg.counterfactuals_params.target_class,
    )

    if x_cfs.size == 0:
        logger.info("No counterfactuals generated.")
        return

    cf_original_space = dataset.inverse_transform(x_cfs)
    cf_path = Path(save_folder) / "counterfactuals_TabDCE.csv"
    np.savetxt(cf_path, cf_original_space, delimiter=",")
    logger.info("Saved counterfactuals to %s", cf_path)


if __name__ == "__main__":
    main()
