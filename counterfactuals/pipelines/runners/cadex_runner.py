import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.cadex import CADEX
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.utils import apply_categorical_discretization

logger = logging.getLogger(__name__)


class CADEXPipelineRunner(PipelineRunner):
    """Pipeline runner for CADEX counterfactual generation."""

    cf_method_name = "CADEX"

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

        self.logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)
        y_target = np.full_like(y_test_origin, target_class)

        self.logger.info("Creating counterfactual model")
        cadex_params = self.cfg.counterfactuals_params.get("cadex", {})
        ordinal_attributes = cadex_params.get("ordinal_attributes")
        if ordinal_attributes:
            raise ValueError("cadex.ordinal_attributes requires scale/unscale hooks in CADEX.")

        cf_method = CADEX(
            disc_model=disc_model,
            categorical_attributes=dataset.categorical_features_lists,
            ordinal_attributes=ordinal_attributes,
            device=self.cfg.counterfactuals_params.get("device"),
        )

        self.logger.info("Handling counterfactual generation")
        with self._timed_search() as timer:
            explanation_result = cf_method.explain(
                X_test_origin,
                y_test_origin,
                y_target,
                num_changed_attributes=cadex_params.get("num_changed_attributes"),
                max_epochs=cadex_params.get("max_epochs", 1000),
                skip_attributes=cadex_params.get("skip_attributes", 0),
                categorical_threshold=cadex_params.get("categorical_threshold", 0.0),
                direction_constraints=cadex_params.get("direction_constraints"),
            )

            Xs = explanation_result.x_origs
            Xs_cfs = explanation_result.x_cfs
            ys_orig = explanation_result.y_origs
            ys_target = explanation_result.y_cf_targets
        cf_search_time = timer["elapsed"]

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


@hydra.main(config_path="./conf", config_name="cadex_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = CADEXPipelineRunner(cfg, logger, CADEXPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
