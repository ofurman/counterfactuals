import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from counterfactuals.cf_methods.local_methods.c_chvae.c_chvae import CCHVAE
from counterfactuals.cf_methods.local_methods.c_chvae.data import CustomData
from counterfactuals.cf_methods.local_methods.c_chvae.mlmodel import CustomMLModel
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult

logger = logging.getLogger(__name__)


class CCHVAEPipelineRunner(PipelineRunner):
    """Pipeline runner for CCHVAE counterfactual generation."""

    cf_method_name = "CCHVAE"

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
        self.logger.info("Creating dataset interface")
        custom_dataset = CustomData(dataset)

        self.logger.info("Creating counterfactual model")
        wrapped_model = CustomMLModel(disc_model, custom_dataset)

        hyperparams = OmegaConf.to_container(
            self.cfg.counterfactuals_params.hyperparams, resolve=True
        )
        input_size = dataset.X_train.shape[1]
        hyperparams["vae_params"]["layers"] = [input_size] + hyperparams["vae_params"]["layers"]

        return CCHVAE(wrapped_model, hyperparams)

    def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
        self.logger.info("Handling counterfactual generation")
        explanation_result = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=self.cfg.counterfactuals_params.epochs,
            lr=self.cfg.counterfactuals_params.lr,
        )
        return CfMethodOutput(
            x_cfs=explanation_result.x_cfs,
            x_origs=explanation_result.x_origs,
            y_origs=explanation_result.y_origs,
            y_targets=explanation_result.y_cf_targets,
        )


@hydra.main(config_path="./conf", config_name="cchvae_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = CCHVAEPipelineRunner(cfg, logger, CCHVAEPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
