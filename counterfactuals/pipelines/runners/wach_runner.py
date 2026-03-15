"""Pipeline runner for WACH counterfactual generation."""

import logging
import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.wach.wach import WACH
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model

logger = logging.getLogger(__name__)


class WACHPipelineRunner(PipelineRunner):
    """Pipeline runner for WACH counterfactual generation.

    WACH does not use a dequantizer, so overrides create_gen_model to call
    the factory without dequantization. The log_prob_threshold is computed
    inside search_counterfactuals instead of in the base run() method.
    """

    cf_method_name = "RPPCEF"

    def create_gen_model(self, dataset, path, dequantizer):
        """Create generative model without dequantizer for WACH.

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the model checkpoint.
            dequantizer: Ignored - WACH does not use dequantization.

        Returns:
            Trained generative model.
        """
        return create_gen_model(self.cfg, dataset, path)

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
        return WACH(disc_model=disc_model)

    def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
        self.logger.info("Handling counterfactual generation")
        target_class = self._get_target_class()
        Xs_cfs, Xs, ys_orig, ys_target, model_returned = cf_method.explain_dataloader(
            dataloader=cf_dataloader, target_class=target_class
        )
        return CfMethodOutput(
            x_cfs=Xs_cfs,
            x_origs=Xs,
            y_origs=ys_orig,
            y_targets=ys_target,
            model_returned=model_returned,
        )

    def _save_counterfactuals(self, X_cf, save_folder, cf_method_name, disc_model_name):
        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(X_cf).to_csv(counterfactuals_path, index=False)
        self.logger.info("Counterfactuals saved to %s", counterfactuals_path)
        return counterfactuals_path


@hydra.main(config_path="./conf", config_name="wach_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = WACHPipelineRunner(cfg, logger, WACHPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
