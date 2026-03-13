import logging
from time import time

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.ppcef import PPCEF
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.utils import apply_categorical_discretization
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PPCEFPipelineRunner(PipelineRunner):
    """Pipeline runner for PPCEF counterfactual generation."""

    cf_method_name = "PPCEF"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        disc_model_criterion = instantiate(self.cfg.counterfactuals_params.disc_model_criterion)

        cf_method = PPCEF(
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=disc_model_criterion,
        )

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        explanation_result = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=self.cfg.counterfactuals_params.epochs,
            lr=self.cfg.counterfactuals_params.lr,
            patience=self.cfg.counterfactuals_params.patience,
            alpha=self.cfg.counterfactuals_params.alpha,
            alpha_s=self.cfg.counterfactuals_params.alpha_s,
            alpha_k=self.cfg.counterfactuals_params.alpha_k,
            log_prob_threshold=log_prob_threshold,
            categorical_intervals=get_categorical_intervals(
                self.cfg.counterfactuals_params.use_categorical,
                dataset.categorical_features_lists,
            ),
            plausibility_weight=self.cfg.counterfactuals_params.plausibility_weight,
            plausibility_bias=self.cfg.counterfactuals_params.plausibility_bias,
        )
        Xs = explanation_result.x_origs
        Xs_cfs = explanation_result.x_cfs
        ys_orig = explanation_result.y_origs
        ys_target = explanation_result.y_cf_targets

        cf_search_time = time() - time_start
        logger.info(f"Counterfactual search time: {cf_search_time:.4f} seconds")

        if self.cfg.counterfactuals_params.use_categorical:
            Xs_cfs = apply_categorical_discretization(dataset.categorical_features_lists, Xs_cfs)
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


def get_categorical_intervals(use_categorical, categorical_features_lists):
    return categorical_features_lists if use_categorical else None


@hydra.main(config_path="./conf", config_name="ppcef_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = PPCEFPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
