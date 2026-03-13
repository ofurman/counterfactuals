import logging
from time import time

import hydra
import numpy as np
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.casebased_sace.casebased_sace import (
    CaseBasedSACE,
)
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


class CaseBasedSACEPipelineRunner(PipelineRunner):
    """Pipeline runner for CaseBasedSACE counterfactual generation."""

    cf_method_name = "CaseBasedSACE"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        _ = gen_model
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating counterfactual model")
        cf_method = CaseBasedSACE(
            disc_model=disc_model,
            variable_features=dataset.numerical_features_indices
            + dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            categorical_features_lists=dataset.categorical_features_lists,
            **self.cfg.counterfactuals_params.cf_method,
        )

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        Xs_cfs, Xs, ys_orig, ys_target, model_returned = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            X_train=np.asarray(dataset.X_train),
            y_train=np.asarray(dataset.y_train),
        )

        cf_search_time = time() - time_start
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

        Xs_cfs = np.asarray(Xs_cfs)
        Xs = np.asarray(Xs)
        ys_orig = np.asarray(ys_orig)
        ys_target = np.asarray(ys_target)
        model_returned = np.asarray(model_returned).astype(bool)

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="casebased_sace_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = CaseBasedSACEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
