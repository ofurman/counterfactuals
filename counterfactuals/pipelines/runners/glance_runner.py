import logging
from time import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.glance.glance import GLANCE
from counterfactuals.metrics.metrics import evaluate_cf_for_glance
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


class GLANCEPipelineRunner(PipelineRunner):
    """Pipeline runner for GLANCE counterfactual generation."""

    cf_method_name = "GLANCE"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        _ = gen_model
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]

        target_class = self.cfg.counterfactuals_params.target_class
        if target_class != 1:
            logger.warning(
                "GLANCE assumes target class 1; overriding configured target_class=%s",
                target_class,
            )
            target_class = 1

        logger.info("Filtering out target class data for counterfactual generation")
        Xs = dataset.X_test[dataset.y_test != target_class]
        ys_orig = dataset.y_test[dataset.y_test != target_class]

        logger.info("Creating counterfactual model")
        cf_method_cfg = self.cfg.counterfactuals_params.cf_method
        cf_method = GLANCE(
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            model=disc_model,
            features=list(dataset.features),
            k=int(cf_method_cfg.get("k", -1)),
            s=int(cf_method_cfg.get("s", 4)),
            m=int(cf_method_cfg.get("m", 1)),
            target_class=target_class,
        )

        logger.info("Handling counterfactual generation")
        time_start = time()
        ys_target = np.abs(ys_orig - 1)
        explanation_results = cf_method.explain(
            X=Xs,
            y_origin=ys_orig,
            y_target=ys_target,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
        )
        Xs_cfs = explanation_results.x_cfs
        model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)
        cf_search_time = np.mean(time() - time_start)
        logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        extras = {"cf_group_ids": explanation_results.cf_group_ids}
        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
            extras=extras,
        )

    def calculate_metrics(self, gen_model, disc_model, dataset, result, log_prob_threshold):
        """Calculate evaluation metrics for GLANCE counterfactuals."""
        logger.info("Calculating metrics")
        metrics = evaluate_cf_for_glance(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=result.X_cf,
            model_returned=result.model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=result.X_test,
            y_test=result.y_orig,
            y_target=result.y_target,
            median_log_prob=log_prob_threshold,
            cf_group_ids=result.extras.get("cf_group_ids"),
            metrics_conf_path="counterfactuals/pipelines/conf/metrics/default.yaml",
        )
        logger.info("Metrics calculated: %s", metrics)
        return metrics


@hydra.main(config_path="./conf", config_name="glance_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = GLANCEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
