"""Pipeline runner for PPCEFR regression counterfactual generation."""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.regression_ppcef.ppcefr import PPCEFR
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics import evaluate_cf_regression
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths

logger = logging.getLogger(__name__)


class PPCEFRPipelineRunner(PipelineRunner):
    """Pipeline runner for PPCEFR regression counterfactual generation.

    Uses raw dataset without MethodDataset preprocessing, no dequantizer,
    single fold (no CV), and regression metrics instead of classification metrics.
    """

    cf_method_name = "PPCEFR"

    def run(self) -> None:
        """Custom run implementation for PPCEFR with single fold and regression metrics."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.logger.info("Loading dataset")
        dataset = instantiate(self.cfg.dataset)

        # Single fold - no CV
        disc_model_path, gen_model_path, save_folder = set_model_paths(self.cfg)

        disc_model = create_disc_model(self.cfg, dataset, disc_model_path, save_folder)

        if self.cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

        gen_model = create_gen_model(self.cfg, dataset, gen_model_path)

        result = self.search_counterfactuals(dataset, gen_model, disc_model, save_folder, None)

        metrics = self.calculate_metrics(gen_model, disc_model, dataset, result, None)

        self.save_results(metrics, result.cf_search_time, save_folder)

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

        self.logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin = dataset.X_test
        y_test_origin = dataset.y_test

        self.logger.info("Creating counterfactual model")
        disc_model_criterion = instantiate(self.cfg.counterfactuals_params.disc_loss)

        cf_method = PPCEFR(
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=disc_model_criterion,
        )

        self.logger.info("Calculating delta threshold")
        train_dataloader_for_delta = dataset.train_dataloader(
            batch_size=self.cfg.counterfactuals_params.batch_size, shuffle=False
        )
        delta = torch.quantile(
            gen_model.predict_log_prob(train_dataloader_for_delta),
            self.cfg.counterfactuals_params.log_prob_quantile,
        )
        self.logger.info("delta: %.4f", delta)
        self._delta = delta  # Store for calculate_metrics

        self.logger.info("Handling counterfactual generation")
        cf_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_test_origin).float(),
                torch.tensor(y_test_origin).float(),
            ),
            batch_size=self.cfg.counterfactuals_params.batch_size,
            shuffle=False,
        )

        with self._timed_search() as timer:
            x_cfs, x_origs, y_origs, y_cf_targets, logs = cf_method.explain_dataloader(
                dataloader=cf_dataloader,
                target_change=self.cfg.counterfactuals_params.target_change,
                epochs=self.cfg.counterfactuals_params.epochs,
                lr=self.cfg.counterfactuals_params.lr,
                alpha=self.cfg.counterfactuals_params.alpha,
                delta=delta,
            )
        cf_search_time = timer["elapsed"]

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_no_plaus_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(x_cfs).to_csv(counterfactuals_path, index=False)
        self.logger.info("Counterfactuals saved to %s", counterfactuals_path)

        model_returned = np.ones(x_cfs.shape[0]).astype(bool)

        return SearchResult(
            X_cf=x_cfs,
            X_test=x_origs,
            y_orig=y_origs,
            y_target=y_cf_targets,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )

    def calculate_metrics(self, gen_model, disc_model, dataset, result, log_prob_threshold):
        """Calculate regression metrics using evaluate_cf_regression."""
        self.logger.info("Calculating metrics")
        metrics = evaluate_cf_regression(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=result.X_cf,
            model_returned=result.model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=result.X_test,
            y_test=result.y_orig,
            median_log_prob=self._delta,
            y_target=result.y_target,
        )
        self.logger.info("Metrics: %s", metrics)
        return metrics


@hydra.main(config_path="./conf", config_name="ppcefr_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = PPCEFRPipelineRunner(cfg, logger, None)
    runner.run()


if __name__ == "__main__":
    main()
