import logging

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cel.cf_methods.global_methods.globe_ce import GLOBE_CE
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import PipelineRunner, SearchResult
from cel.pipelines.utils import (
    align_counterfactuals_with_factuals,
    one_hot,
)
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


def compute_bin_widths(
    dataset: MethodDataset, data: pd.DataFrame, n_bins: int = 10
) -> dict[str, float]:
    """Compute the width of the last histogram bin for each continuous feature.

    Skips features listed in ``dataset.categorical_features``. Logs a warning
    and skips any feature where binning fails or produces no categories.

    Args:
        dataset: Dataset object exposing ``categorical_features``.
        data: Raw (unscaled) feature DataFrame.
        n_bins: Number of equal-width bins to use for each feature.

    Returns:
        Mapping from feature name to bin width (midpoint length of last bin).
    """
    bin_widths = {}
    for feature in data.columns:
        if feature in dataset.categorical_features:
            continue

        try:
            categories = pd.cut(data[feature].astype(float), bins=n_bins).cat.categories
        except ValueError as err:
            logger.warning("Skipping bin width computation for feature %s: %s", feature, err)
            continue

        if len(categories) == 0:
            logger.warning(
                "Skipping bin width computation for feature %s: no categories returned",
                feature,
            )
            continue

        bin_widths[feature] = float(categories.length[-1])

    return bin_widths


class GLOBECEPipelineRunner(PipelineRunner):
    """Pipeline runner for GLOBE-CE counterfactual generation."""

    cf_method_name = "GLOBE_CE"

    def load_dataset(self):
        file_dataset = instantiate(self.cfg.dataset)
        return MethodDataset(file_dataset, self.preprocessing_pipeline)

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
        disc_model.eval()
        disc_model_name = self._get_disc_model_name()
        target_class = self._get_target_class()

        minmax_scaler = dataset.preprocessing_pipeline.get_step("minmax")

        X_test_unscaled = minmax_scaler._inverse_transform_array(dataset.X_test)
        data_oh, features = one_hot(
            dataset, pd.DataFrame(X_test_unscaled, columns=dataset.features)
        )

        def predict_fn(x):
            x_array = x.values if isinstance(x, pd.DataFrame) else x
            x_scaled = minmax_scaler._transform_array(x_array)
            return disc_model.predict(x_scaled)

        self.logger.info("Filtering out target class data for counterfactual generation")
        ys_pred = predict_fn(X_test_unscaled)
        mask = ys_pred != target_class
        Xs_unscaled = X_test_unscaled[mask]
        Xs = dataset.X_test[mask]
        ys_orig = ys_pred[mask]

        self.logger.info("Computing bin widths for continuous features")
        bin_widths = compute_bin_widths(
            dataset=dataset,
            data=pd.DataFrame(X_test_unscaled, columns=dataset.features),
            n_bins=10,
        )

        cf_method = GLOBE_CE(
            predict_fn=predict_fn,
            dataset=dataset,
            X=pd.DataFrame(Xs_unscaled, columns=dataset.features),
            bin_widths=bin_widths,
            target_class=target_class,
        )

        self.logger.info("Handling counterfactual generation")
        with self._timed_search() as timer:
            ys_target = np.full_like(ys_orig, target_class)
            explanation_result = cf_method.explain(
                y_origin=ys_orig,
                y_target=ys_target,
            )
            Xs_cfs = explanation_result.x_cfs
            Xs_cfs = minmax_scaler._transform_array(Xs_cfs)
            Xs_cfs, model_returned = align_counterfactuals_with_factuals(Xs_cfs, Xs)
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


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    # GLOBE-CE uses torch_dtype first, then minmax (different order)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("torch_dtype", TorchDataTypeStep()),
            ("minmax", MinMaxScalingStep()),
        ]
    )
    runner = GLOBECEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
