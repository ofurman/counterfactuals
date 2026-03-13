"""Pipeline runner for TabDCE pairwise counterfactual generation."""

import logging
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from counterfactuals.cf_methods.local_methods.tabdce.tabdce import TabDCE
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.run_tabdce_pipeline import (
    create_diffusion_model,
    prepare_tabular_dataset,
    train_tabdce_diffusion,
)
from counterfactuals.pipelines.runners.pairwise_mixin import PairwiseMixin

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TabDCEPairwisePipelineRunner(PairwiseMixin, PipelineRunner):
    """Pipeline runner for TabDCE counterfactual generation with multiple CFs per instance."""

    cf_method_name = "TabDCEPairwise"

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
                extras={"Xs_cfs_all": np.empty((0, 0, dataset.X_test.shape[1]))},
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

        cf_per_instance = int(self.cfg.counterfactuals_params.get("cf_samples_per_factual", 5))
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
            save_folder, f"counterfactuals_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(Xs_cfs_all.reshape(-1, Xs_cfs_all.shape[-1])).to_csv(
            counterfactuals_path, index=False
        )
        logger.info("Counterfactuals saved to %s", counterfactuals_path)

        return SearchResult(
            X_cf=Xs_cfs_first,
            X_test=x_origs,
            y_orig=y_origs,
            y_target=y_targets,
            model_returned=model_returned_first,
            cf_search_time=cf_search_time,
            extras={"Xs_cfs_all": Xs_cfs_all},
        )
