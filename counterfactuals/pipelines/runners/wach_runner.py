"""Pipeline runner for WACH counterfactual generation."""

import logging
import os
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.wach.wach import WACH
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class WACHPipelineRunner(PipelineRunner):
    """Pipeline runner for WACH counterfactual generation.

    WACH does not use a dequantizer, so overrides create_gen_model to call
    the factory without dequantization. The log_prob_threshold is computed
    inside search_counterfactuals instead of in the base run() method.
    """

    cf_method_name = "RPPCEF"

    def create_gen_model(self, dataset, path, dequantizer):
        """Create generative model without dequantizer for WACH.

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the model checkpoint.
            dequantizer: Ignored - WACH does not use dequantization.

        Returns:
            Trained generative model.
        """
        return create_gen_model(self.cfg, dataset, path)

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        cf_method: WACH = WACH(disc_model=disc_model)

        logger.info("Calculating log_prob_threshold")
        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=self.cfg.counterfactuals_params.batch_size, shuffle=False
        )
        log_prob_threshold = torch.quantile(
            gen_model.predict_log_prob(train_dataloader_for_log_prob),
            self.cfg.counterfactuals_params.log_prob_quantile,
        )
        logger.info("log_prob_threshold: %.4f", log_prob_threshold)

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        Xs_cfs, Xs, ys_orig, ys_target, model_returned = cf_method.explain_dataloader(
            dataloader=cf_dataloader, target_class=target_class
        )

        cf_search_time = np.mean(time() - time_start)
        logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_no_plaus_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        logger.info("Counterfactuals saved to %s", counterfactuals_path)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="wach_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = WACHPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
