import logging
from time import time

import hydra
import numpy as np
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.artelt.artelt import Artelt
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


class ArteltPipelineRunner(PipelineRunner):
    """Pipeline runner for Artelt counterfactual generation."""

    cf_method_name = "Artelt"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        _ = gen_model
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        cf_method = Artelt(disc_model=disc_model)

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        cf_method.fit_density_estimators(
            X_train=np.asarray(dataset.X_train),
            y_train=np.asarray(dataset.y_train).reshape(-1),
        )
        explanation_result = cf_method.explain_dataloader(dataloader=cf_dataloader)

        cf_search_time = time() - time_start
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

        Xs_cfs = np.atleast_2d(np.asarray(explanation_result.x_cfs))
        Xs = np.atleast_2d(np.asarray(explanation_result.x_origs))
        ys_orig = np.asarray(explanation_result.y_origs)
        ys_target = np.asarray(explanation_result.y_cf_targets)
        model_returned = ~np.isnan(Xs_cfs).any(axis=1)

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="artelt_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = ArteltPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
