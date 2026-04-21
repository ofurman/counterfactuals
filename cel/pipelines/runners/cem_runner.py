import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from cel.cf_methods.local_methods.cem.cem import CEM_CF
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult

logger = logging.getLogger(__name__)


class CEMPipelineRunner(PipelineRunner):
    """Pipeline runner for CEM counterfactual generation."""

    cf_method_name = "CEM"

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

    def create_cf_method(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
    ) -> object:
        """Instantiate the CEM CF method.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained generative model (unused by CEM).
            disc_model: Trained discriminative model.

        Returns:
            CEM_CF method instance.
        """
        self.logger.info("Creating counterfactual model")
        return CEM_CF(
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

    def run_cf_method(
        self,
        cf_method: object,
        cf_dataloader: torch.utils.data.DataLoader,
        dataset: MethodDataset,
        log_prob_threshold: float,
    ) -> CfMethodOutput:
        """Run CEM CF generation via explain_dataloader.

        Args:
            cf_method: CEM_CF method instance.
            cf_dataloader: DataLoader for the filtered test set.
            dataset: The current fold's dataset providing X_train.
            log_prob_threshold: Plausibility threshold (unused by CEM).

        Returns:
            CfMethodOutput with generated counterfactuals.
        """
        self.logger.info("Handling counterfactual generation")
        target_class = self._get_target_class()
        explanation_result = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            target_class=target_class,
            X_train=np.asarray(dataset.X_train),
        )
        logs = explanation_result.logs or {}
        x_cfs = np.asarray(explanation_result.x_cfs)
        model_returned = np.asarray(logs.get("model_returned", np.ones(len(x_cfs), dtype=bool)))
        return CfMethodOutput(
            x_cfs=x_cfs,
            x_origs=np.asarray(explanation_result.x_origs),
            y_origs=np.asarray(explanation_result.y_origs),
            y_targets=np.asarray(explanation_result.y_cf_targets),
            model_returned=model_returned,
        )


@hydra.main(config_path="./conf", config_name="cem_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = CEMPipelineRunner(cfg, logger, CEMPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
