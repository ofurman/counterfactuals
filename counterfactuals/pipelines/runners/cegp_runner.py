import logging
from time import time

import hydra
import numpy as np
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.cegp.cegp import CEGP
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


class CEGPPipelineRunner(PipelineRunner):
    """Pipeline runner for CEGP counterfactual generation."""

    cf_method_name = "CEGP"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        cf_method = CEGP(
            disc_model=disc_model,
            beta=self.cfg.counterfactuals_params.beta,
            c_init=self.cfg.counterfactuals_params.c_init,
            c_steps=self.cfg.counterfactuals_params.c_steps,
            max_iterations=self.cfg.counterfactuals_params.max_iterations,
            feature_range=tuple(self.cfg.counterfactuals_params.feature_range),
            d_type=self.cfg.counterfactuals_params.fit_d_type,
            disc_perc=list(self.cfg.counterfactuals_params.fit_disc_perc),
        )

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )

        time_start = time()
        explanation_result = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            target_class=target_class,
            X_train=np.asarray(dataset.X_train),
        )
        Xs_cfs = explanation_result.x_cfs
        Xs = explanation_result.x_origs
        ys_orig = explanation_result.y_origs
        ys_target = explanation_result.y_cf_targets
        logs = explanation_result.logs or {}
        model_returned = np.asarray(logs.get("model_returned", np.ones(len(Xs_cfs), dtype=bool)))

        cf_search_time = np.mean(time() - time_start)
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="cegp_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = CEGPPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
