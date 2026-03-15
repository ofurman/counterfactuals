import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.casebased_sace.casebased_sace import (
    CaseBasedSACE,
)
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult

logger = logging.getLogger(__name__)


class CaseBasedSACEPipelineRunner(PipelineRunner):
    """Pipeline runner for CaseBasedSACE counterfactual generation."""

    cf_method_name = "CaseBasedSACE"

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
        return self._default_search_counterfactuals(
            dataset, gen_model, disc_model, save_folder, log_prob_threshold
        )

    def create_cf_method(self, dataset, gen_model, disc_model):
        self.logger.info("Creating counterfactual model")
        return CaseBasedSACE(
            disc_model=disc_model,
            variable_features=dataset.numerical_features_indices
            + dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            categorical_features_lists=dataset.categorical_features_lists,
            **self.cfg.counterfactuals_params.cf_method,
        )

    def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
        self.logger.info("Handling counterfactual generation")
        Xs_cfs, Xs, ys_orig, ys_target, model_returned = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            X_train=np.asarray(dataset.X_train),
            y_train=np.asarray(dataset.y_train),
        )
        return CfMethodOutput(
            x_cfs=np.asarray(Xs_cfs),
            x_origs=np.asarray(Xs),
            y_origs=np.asarray(ys_orig),
            y_targets=np.asarray(ys_target),
            model_returned=np.asarray(model_returned).astype(bool),
        )


@hydra.main(config_path="./conf", config_name="casebased_sace_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = CaseBasedSACEPipelineRunner(
        cfg, logger, CaseBasedSACEPipelineRunner.default_preprocessing()
    )
    runner.run()


if __name__ == "__main__":
    main()
