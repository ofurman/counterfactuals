import logging
import os
from time import time

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.wach.wach_ours import WACH_OURS
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


class WACHOURSPipelineRunner(PipelineRunner):
    """Pipeline runner for WACH_OURS counterfactual generation."""

    cf_method_name = "WACH_OURS"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        disc_model_criterion = instantiate(self.cfg.counterfactuals_params.disc_model_criterion)
        cf_method = WACH_OURS(
            disc_model=disc_model,
            disc_model_criterion=disc_model_criterion,
        )

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        results = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=self.cfg.counterfactuals_params.epochs,
            lr=self.cfg.counterfactuals_params.lr,
            alpha=self.cfg.counterfactuals_params.alpha,
        )
        Xs_cfs = results.x_cfs
        Xs = results.x_origs
        ys_orig = results.y_origs
        ys_target = results.y_cf_targets

        model_returned = (np.ones(Xs_cfs.shape[0]),)

        cf_search_time = np.mean(time() - time_start)
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")
        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_no_plaus_{self.cf_method_name}_{disc_model_name}.csv"
        )

        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        logger.info("Counterfactual deltas saved to %s", counterfactuals_path)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="wach_ours_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = WACHOURSPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
