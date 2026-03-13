"""Pipeline runner for CCHVAE pairwise counterfactual generation."""

import logging
import os
from time import time

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from counterfactuals.cf_methods.local_methods.c_chvae.c_chvae import CCHVAE
from counterfactuals.cf_methods.local_methods.c_chvae.data import CustomData
from counterfactuals.cf_methods.local_methods.c_chvae.mlmodel import CustomMLModel
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.runners.pairwise_mixin import PairwiseMixin

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class CCHVAEPairwisePipelineRunner(PairwiseMixin, PipelineRunner):
    """Pipeline runner for CCHVAE counterfactual generation with multiple CFs per instance."""

    cf_method_name = "CCHVAE"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]

        logger.info("Filtering out target class data for counterfactual generation")
        target_class = self.cfg.counterfactuals_params.target_class
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating dataset interface")
        custom_dataset = CustomData(dataset)

        logger.info("Creating counterfactual model")
        wrapped_model = CustomMLModel(disc_model, custom_dataset)

        hyperparams = OmegaConf.to_container(
            self.cfg.counterfactuals_params.hyperparams, resolve=True
        )
        if not hyperparams.get("data_name"):
            hyperparams["data_name"] = self.cfg.dataset.config_path.split("/")[-1].split(".")[0]

        input_size = dataset.X_train.shape[1]
        hyperparams["vae_params"]["layers"] = [input_size] + hyperparams["vae_params"]["layers"]

        exp = CCHVAE(wrapped_model, hyperparams)

        logger.info("Handling counterfactual generation")
        cf_per_instance = int(self.cfg.counterfactuals_params.get("num_counterfactuals", 1))
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )

        time_start = time()
        cfs_list: list[np.ndarray] = []
        y_target = np.abs(1 - y_test_origin)
        for _ in range(cf_per_instance):
            explanation_result = exp.explain_dataloader(
                dataloader=cf_dataloader,
                epochs=self.cfg.counterfactuals_params.epochs,
                lr=self.cfg.counterfactuals_params.lr,
                y_target=y_target,
            )
            cfs_list.append(explanation_result.x_cfs)

        cf_search_time = time() - time_start
        logger.info("Counterfactual search time: %.4f seconds", cf_search_time)

        Xs_cfs_all = np.stack(cfs_list, axis=1)
        Xs_cfs_first = Xs_cfs_all[:, 0, :]

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(Xs_cfs_all.reshape(-1, Xs_cfs_all.shape[-1])).to_csv(
            counterfactuals_path, index=False
        )
        logger.info("Counterfactuals saved to %s", counterfactuals_path)

        model_returned_first = np.ones(Xs_cfs_first.shape[0], dtype=bool)

        return SearchResult(
            X_cf=Xs_cfs_first,
            X_test=X_test_origin,
            y_orig=y_test_origin,
            y_target=y_target,
            model_returned=model_returned_first,
            cf_search_time=cf_search_time,
            extras={"Xs_cfs_all": Xs_cfs_all},
        )
