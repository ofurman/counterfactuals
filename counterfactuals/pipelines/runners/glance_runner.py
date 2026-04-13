import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.glance.glance import GLANCE
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics.metrics import evaluate_cf_for_glance
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult

logger = logging.getLogger(__name__)


class GLANCEPipelineRunner(PipelineRunner):
    """Pipeline runner for GLANCE counterfactual generation."""

    cf_method_name = "GLANCE"

    def search_counterfactuals(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        save_folder: str,
        log_prob_threshold: float,
    ) -> SearchResult:
        """Generate counterfactuals for the current fold.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained generative model.
            disc_model: Trained discriminative model.
            save_folder: Directory for saving generated counterfactuals.
            log_prob_threshold: Plausibility threshold from compute_log_prob_threshold.

        Returns:
            SearchResult with counterfactuals and timing information.
        """
        _ = gen_model
        disc_model_name = self._get_disc_model_name()

        target_class = self._get_target_class()
        if target_class != 1:
            self.logger.warning(
                "GLANCE assumes target class 1; overriding configured target_class=%s",
                target_class,
            )
            target_class = 1

        self.logger.info("Filtering out target class data for counterfactual generation")
        Xs = dataset.X_test[dataset.y_test != target_class]
        ys_orig = dataset.y_test[dataset.y_test != target_class]

        self.logger.info("Creating counterfactual model")
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

        self.logger.info("Handling counterfactual generation")
        with self._timed_search() as timer:
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
        cf_search_time = timer["elapsed"]

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
        self.logger.info("Calculating metrics")
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
        self.logger.info("Metrics calculated: %s", metrics)
        return metrics


@hydra.main(config_path="./conf", config_name="glance_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = GLANCEPipelineRunner(cfg, logger, GLANCEPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
