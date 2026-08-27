import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cel.cf_methods.local_methods.wach.wach import WACH
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult

logger = logging.getLogger(__name__)


class WACHPipelineRunner(PipelineRunner):
    """Pipeline runner for WACH counterfactual generation."""

    cf_method_name = "WACH"

    def search_counterfactuals(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        save_folder: str,
        log_prob_threshold: float,
    ) -> SearchResult:
        """Generate counterfactuals for the current fold."""
        return self._default_search_counterfactuals(
            dataset, gen_model, disc_model, save_folder, log_prob_threshold
        )

    def create_cf_method(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
    ) -> object:
        """Instantiate the WACH CF method."""
        self.logger.info("Creating counterfactual model")
        disc_model_criterion = instantiate(self.cfg.counterfactuals_params.disc_model_criterion)
        return WACH(
            disc_model=disc_model,
            disc_model_criterion=disc_model_criterion,
        )

    def run_cf_method(
        self,
        cf_method: object,
        cf_dataloader: torch.utils.data.DataLoader,
        dataset: MethodDataset,
        log_prob_threshold: float,
    ) -> CfMethodOutput:
        """Run WACH CF generation via explain_dataloader."""
        self.logger.info("Handling counterfactual generation")
        results = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=self.cfg.counterfactuals_params.epochs,
            lr=self.cfg.counterfactuals_params.lr,
            alpha=self.cfg.counterfactuals_params.alpha,
        )
        return CfMethodOutput(
            x_cfs=results.x_cfs,
            x_origs=results.x_origs,
            y_origs=results.y_origs,
            y_targets=results.y_cf_targets,
            model_returned=np.ones(results.x_cfs.shape[0], dtype=bool),
        )

    def _save_counterfactuals(self, X_cf, save_folder, cf_method_name, disc_model_name):
        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(X_cf).to_csv(counterfactuals_path, index=False)
        self.logger.info("Counterfactual deltas saved to %s", counterfactuals_path)
        return counterfactuals_path


@hydra.main(config_path="./conf", config_name="wach_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    runner = WACHPipelineRunner(cfg, logger, WACHPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
