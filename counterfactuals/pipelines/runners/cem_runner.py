import logging
from time import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.cem.cem import CEM_CF
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


class CEMPipelineRunner(PipelineRunner):
    """Pipeline runner for CEM counterfactual generation."""

    cf_method_name = "CEM"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        _ = gen_model
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        cf_method = CEM_CF(
            disc_model=disc_model,
            mode=self.cfg.counterfactuals_params.mode,
            kappa=self.cfg.counterfactuals_params.kappa,
            beta=self.cfg.counterfactuals_params.beta,
            c_init=self.cfg.counterfactuals_params.c_init,
            c_steps=self.cfg.counterfactuals_params.c_steps,
            max_iterations=self.cfg.counterfactuals_params.max_iterations,
            learning_rate_init=self.cfg.counterfactuals_params.learning_rate_init,
            no_info_type=self.cfg.counterfactuals_params.fit_no_info_type,
            feature_range=tuple(self.cfg.counterfactuals_params.feature_range),
            clip=tuple(self.cfg.counterfactuals_params.clip_range),
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

        cf_search_time = time() - time_start
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

        Xs_cfs = np.asarray(explanation_result.x_cfs)
        Xs = np.asarray(explanation_result.x_origs)
        ys_orig = np.asarray(explanation_result.y_origs)
        ys_target = np.asarray(explanation_result.y_cf_targets)
        logs = explanation_result.logs or {}
        model_returned = np.asarray(logs.get("model_returned", np.ones(len(Xs_cfs), dtype=bool)))

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="cem_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = CEMPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
