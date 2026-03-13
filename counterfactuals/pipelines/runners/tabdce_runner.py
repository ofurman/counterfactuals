import logging
import os
from pathlib import Path
from time import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from counterfactuals.cf_methods.local_methods.tabdce.data import (
    TabularCounterfactualDataset,
    TabularSpec,
)
from counterfactuals.cf_methods.local_methods.tabdce.denoise import TabularEpsModel
from counterfactuals.cf_methods.local_methods.tabdce.diffusion import (
    MixedTabularDiffusion,
)
from counterfactuals.cf_methods.local_methods.tabdce.tabdce import TabDCE
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
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
    return PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )


def prepare_tabular_dataset(dataset, cfg, device):
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


def create_diffusion_model(tab_dataset, cfg, device):
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


def train_tabdce_diffusion(model, dataloader, epochs, lr, model_path):
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


class TabDCEPipelineRunner(PipelineRunner):
    """Pipeline runner for TabDCE counterfactual generation."""

    cf_method_name = "TabDCE"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        _ = gen_model, disc_model
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        use_gpu = torch.cuda.is_available() and self.cfg.tabdce.get("use_gpu", False)
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cuda" if use_gpu else "cpu")
        logger.info("Using device: %s", device)

        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        if X_test_origin.shape[0] == 0:
            logger.info("All samples already belong to the target class %s", target_class)
            return SearchResult(
                X_cf=np.empty((0, dataset.X_test.shape[1])),
                X_test=np.empty((0, dataset.X_test.shape[1])),
                y_orig=np.array([]),
                y_target=np.array([]),
                model_returned=np.array([], dtype=bool),
                cf_search_time=0.0,
            )

        tab_dataset = prepare_tabular_dataset(dataset, self.cfg, device)
        train_loader = DataLoader(tab_dataset, batch_size=self.cfg.tabdce.batch_size, shuffle=True)
        diffusion_model = create_diffusion_model(tab_dataset, self.cfg, device)
        diffusion_path = Path(save_folder) / "tabdce_diffusion.pt"
        train_tabdce_diffusion(
            model=diffusion_model,
            dataloader=train_loader,
            epochs=self.cfg.tabdce.epochs,
            lr=self.cfg.tabdce.lr,
            model_path=diffusion_path,
        )

        cf_method = TabDCE(
            diffusion_model=diffusion_model,
            spec=tab_dataset.spec,
            qt=tab_dataset.qt,
            ohe=tab_dataset.ohe,
            device=device,
        )

        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
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

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig):
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)
    preprocessing_pipeline = build_preprocessing_pipeline()
    runner = TabDCEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
