import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.ppcef import PPCEF
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult
from counterfactuals.pipelines.utils import apply_categorical_discretization

logger = logging.getLogger(__name__)


class PPCEFPipelineRunner(PipelineRunner):
    """Pipeline runner for PPCEF counterfactual generation."""

    cf_method_name = "PPCEF"

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
        disc_model_criterion = instantiate(self.cfg.counterfactuals_params.disc_model_criterion)
        return PPCEF(
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=disc_model_criterion,
        )

    def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
        self.logger.info("Handling counterfactual generation")
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
        return CfMethodOutput(
            x_cfs=explanation_result.x_cfs,
            x_origs=explanation_result.x_origs,
            y_origs=explanation_result.y_origs,
            y_targets=explanation_result.y_cf_targets,
        )

    def postprocess_cf_output(self, output, dataset):
        if self.cfg.counterfactuals_params.use_categorical:
            output.x_cfs = apply_categorical_discretization(
                dataset.categorical_features_lists, output.x_cfs
            )
        return output


def get_categorical_intervals(use_categorical, categorical_features_lists):
    return categorical_features_lists if use_categorical else None


@hydra.main(config_path="./conf", config_name="ppcef_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = PPCEFPipelineRunner(cfg, logger, PPCEFPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
