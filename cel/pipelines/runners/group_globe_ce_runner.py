"""Pipeline runner for Group GLOBE-CE counterfactual generation."""

import logging

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from cel.cf_methods import GLOBE_CE
from cel.datasets.method_dataset import MethodDataset
from cel.pipelines.base_runner import PipelineRunner, SearchResult
from cel.pipelines.runners.globe_ce_runner import compute_bin_widths
from cel.pipelines.utils import align_counterfactuals_with_factuals, one_hot
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


class GroupGLOBECEPipelineRunner(PipelineRunner):
    """Pipeline runner for Group GLOBE-CE counterfactual generation.

    Same orchestration as GLOBE-CE, but performs KMeans clustering on the test set
    and runs a per-cluster AReS + GLOBE-CE search.
    """

    cf_method_name = "Group_Globe_CE"

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
        """Generate counterfactuals for the current fold."""
        disc_model.eval()
        disc_model_name = self._get_disc_model_name()
        target_class = self._get_target_class()

        minmax_scaler = dataset.preprocessing_pipeline.get_step("minmax")

        X_test_unscaled = minmax_scaler._inverse_transform_array(dataset.X_test)
        one_hot(dataset, pd.DataFrame(X_test_unscaled, columns=dataset.features))

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

        kmeans = KMeans(n_clusters=self.cfg.counterfactuals_params.n_clusters)
        kmeans.fit(Xs)
        labels = kmeans.labels_

        self.logger.info("Handling counterfactual generation")
        ys_target = np.full_like(ys_orig, target_class)
        with self._timed_search() as timer:
            Xs_cfs_unscaled = np.empty_like(Xs_unscaled)
            for i in range(kmeans.n_clusters):
                self.logger.info("Creating counterfactual model for cluster %d", i)
                cluster_mask = labels == i
                cluster_X = pd.DataFrame(Xs_unscaled[cluster_mask], columns=dataset.features)

                cf_method = GLOBE_CE(
                    predict_fn=predict_fn,
                    dataset=dataset,
                    X=cluster_X,
                    bin_widths=bin_widths,
                    target_class=target_class,
                )
                explanation_result = cf_method.explain(
                    y_origin=ys_orig[cluster_mask],
                    y_target=ys_target[cluster_mask],
                )
                Xs_cfs_unscaled[cluster_mask] = explanation_result.x_cfs
                cluster_cf_preds = predict_fn(explanation_result.x_cfs)
                self.logger.info(
                    "Cluster %d: %d/%d CFs flip via predict_fn",
                    i,
                    int((cluster_cf_preds == target_class).sum()),
                    len(cluster_cf_preds),
                )

            Xs_cfs = minmax_scaler._transform_array(Xs_cfs_unscaled)
            scaled_preds = disc_model.predict(Xs_cfs)
            scaled_preds_np = (
                scaled_preds.detach().cpu().numpy()
                if hasattr(scaled_preds, "detach")
                else np.asarray(scaled_preds)
            ).flatten()
            self.logger.info(
                "After rescale: %d/%d CFs flip via disc_model.predict",
                int((scaled_preds_np == target_class).sum()),
                len(scaled_preds_np),
            )
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


@hydra.main(config_path="./conf", config_name="group_globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("torch_dtype", TorchDataTypeStep()),
            ("minmax", MinMaxScalingStep()),
        ]
    )
    runner = GroupGLOBECEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
