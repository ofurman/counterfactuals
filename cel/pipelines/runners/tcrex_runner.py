"""Pipeline runner for TCREx counterfactual generation."""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cel.cf_methods.group_methods.tcrex import TCREx
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import PipelineRunner, SearchResult
from cel.pipelines.nodes.gen_model_nodes import create_gen_model
from cel.pipelines.utils import align_counterfactuals_with_factuals

logger = logging.getLogger(__name__)


class TCRExPipelineRunner(PipelineRunner):
    """Pipeline runner for TCREx group counterfactual generation.

    TCREx filters by origin_class (not target_class), uses a surrogate tree
    for explaining groups, and tracks the number of groups as an additional
    metric. It does not use dequantization.
    """

    cf_method_name = "TCREx"

    def create_gen_model(self, dataset, path, dequantizer):
        """Create generative model without dequantizer for TCREx.

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the model checkpoint.
            dequantizer: Ignored - TCREx does not use dequantization.

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
        disc_model_name = self._get_disc_model_name()

        origin_class = self.cfg.counterfactuals_params.origin_class
        target_class = self._get_target_class()

        self.logger.info("Filtering by origin class for counterfactual generation")
        mask = dataset.y_test == origin_class
        X_test_origin = dataset.X_test[mask]
        y_test_origin = dataset.y_test[mask]

        self.logger.info("Creating TCREx counterfactual model")
        cf_method = TCREx(
            target_model=disc_model,
            tau=self.cfg.counterfactuals_params.tau,
            rho=self.cfg.counterfactuals_params.rho,
            surrogate_tree_params=self.cfg.counterfactuals_params.surrogate_tree_params,
        )

        self.logger.info("Fitting the TCREx model")
        with self._timed_search() as timer:
            cf_method.fit(dataset.X_train, dataset.y_train)

            self.logger.info("Generating counterfactuals")
            Xs_cfs = cf_method.explain(X_test_origin)
            Xs_cfs, model_returned = align_counterfactuals_with_factuals(Xs_cfs, X_test_origin)
        cf_search_time = timer["elapsed"]

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        self.logger.info("Counterfactuals saved to %s", counterfactuals_path)

        n_groups = cf_method.n_groups_
        self._n_groups = n_groups  # Store for save_results
        ys_target = np.full_like(y_test_origin, target_class)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=X_test_origin,
            y_orig=y_test_origin,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
            extras={"n_groups": n_groups},
        )

    def save_results(self, metrics, cf_search_time, save_folder):
        """Save metrics to CSV, adding n_groups from the TCREx model."""
        super().save_results(metrics, cf_search_time, save_folder)

        # Add n_groups to the CSV
        csv_path = os.path.join(
            save_folder,
            f"cf_metrics_{self.cfg.disc_model.model._target_.split('.')[-1]}.csv",
        )
        df = pd.read_csv(csv_path)
        df["n_groups"] = self._n_groups
        df.to_csv(csv_path, index=False)


@hydra.main(config_path="./conf", config_name="tcrex_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = TCRExPipelineRunner(cfg, logger, TCRExPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
