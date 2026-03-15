import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.cegp.cegp import CEGP
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult

logger = logging.getLogger(__name__)


class CEGPPipelineRunner(PipelineRunner):
    """Pipeline runner for CEGP counterfactual generation."""

    cf_method_name = "CEGP"

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
        return CEGP(
            disc_model=disc_model,
            beta=self.cfg.counterfactuals_params.beta,
            c_init=self.cfg.counterfactuals_params.c_init,
            c_steps=self.cfg.counterfactuals_params.c_steps,
            max_iterations=self.cfg.counterfactuals_params.max_iterations,
            feature_range=tuple(self.cfg.counterfactuals_params.feature_range),
            d_type=self.cfg.counterfactuals_params.fit_d_type,
            disc_perc=list(self.cfg.counterfactuals_params.fit_disc_perc),
        )

    def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
        self.logger.info("Handling counterfactual generation")
        target_class = self._get_target_class()
        explanation_result = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            target_class=target_class,
            X_train=np.asarray(dataset.X_train),
        )
        logs = explanation_result.logs or {}
        model_returned = np.asarray(
            logs.get("model_returned", np.ones(len(explanation_result.x_cfs), dtype=bool))
        )
        return CfMethodOutput(
            x_cfs=explanation_result.x_cfs,
            x_origs=explanation_result.x_origs,
            y_origs=explanation_result.y_origs,
            y_targets=explanation_result.y_cf_targets,
            model_returned=model_returned,
        )


@hydra.main(config_path="./conf", config_name="cegp_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = CEGPPipelineRunner(cfg, logger, CEGPPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
