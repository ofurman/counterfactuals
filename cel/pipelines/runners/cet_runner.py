import logging

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cel.cf_methods.local_methods.cet.cet import (
    CounterfactualExplanationTree,
)
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import PipelineRunner, SearchResult

MAX_ITERATION = 50
LAMBDA, GAMMA = 0.02, 1.0

logger = logging.getLogger(__name__)


class CETPipelineRunner(PipelineRunner):
    """Pipeline runner for CET counterfactual generation."""

    cf_method_name = "CET"

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
        disc_model.eval()
        disc_model_name = self._get_disc_model_name()
        X_train = dataset.inverse_transform(dataset.X_train)
        y_train = dataset.y_train
        X_test = dataset.inverse_transform(dataset.X_test)
        target_class = self._get_target_class()

        self.logger.info("Filtering out target class data for counterfactual generation")
        ys_pred = disc_model.predict(dataset.X_test)
        mask = ys_pred != target_class
        Xs = dataset.X_test[mask]
        X_test = X_test[mask]
        ys_orig = ys_pred[mask]

        self.logger.info("Creating counterfactual model")
        X_train_df = pd.DataFrame(X_train, columns=dataset.features)
        columns = X_train_df.columns
        X_train = X_train_df.to_numpy()
        feature_types = ["I" for _ in range(X_train.shape[1])]
        feature_constraints = ["" for _ in range(X_train.shape[1])]
        feature_categories = []

        disc_model_wrapper = DiscModelWrapper(disc_model)

        cet = CounterfactualExplanationTree(
            disc_model_wrapper,
            X_train,
            y_train,
            max_iteration=MAX_ITERATION,
            lime_approximation=False,
            feature_names=columns,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            target_name=dataset.features[-1],
            target_labels=[0, 1],
        )

        self.logger.info("Handling counterfactual generation")
        with self._timed_search() as timer:
            cet = cet.fit(
                X_test,
                max_change_num=3,
                cost_type="MPS",
                C=LAMBDA,
                gamma=GAMMA,
                time_limit=60,
                verbose=True,
            )
            Xs_cfs = cet.predict(X_test)
            ys_target = np.abs(ys_orig - 1)
            model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)
        cf_search_time = timer["elapsed"]

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


class DiscModelWrapper:
    """Simple wrapper adapting torch-based discriminator to CET's numpy API."""

    def __init__(self, disc_model: torch.nn.Module) -> None:
        self.disc_model = disc_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.disc_model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.disc_model.predict_proba(X))


@hydra.main(config_path="./conf", config_name="cet_config", version_base="1.2")
def main(cfg: DictConfig):
    runner = CETPipelineRunner(cfg, logger, CETPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
