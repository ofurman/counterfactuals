"""Pipeline runner for DiCE pairwise counterfactual generation."""

import logging
import os
from time import time

import dice_ml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.runners.pairwise_mixin import PairwiseMixin

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DiscWrapper(nn.Module):
    """Wrap a discriminative model with a PyTorch-style `forward` method for DiCE."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class DiCEPairwisePipelineRunner(PairwiseMixin, PipelineRunner):
    """Pipeline runner for DiCE counterfactual generation with multiple CFs per instance."""

    cf_method_name = "DiceExplainerWrapper"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)
        X_test_origin = X_test_origin.astype(np.float64)
        y_test_origin = y_test_origin.astype(np.float64)

        logger.info("Creating dataset interface")
        X_train, y_train = dataset.X_train, dataset.y_train

        features = list(range(dataset.X_train.shape[1])) + ["label"]
        features = list(map(str, features))

        logger.info("Combining train and test data for DiCE range establishment")
        X_combined = np.concatenate([X_train, X_test_origin], axis=0)
        y_combined = np.concatenate([y_train, y_test_origin], axis=0)

        combined_dataframe = pd.DataFrame(
            np.concatenate((X_combined, y_combined.reshape(-1, 1)), axis=1),
            columns=features,
        )

        dice = dice_ml.Data(
            dataframe=combined_dataframe,
            continuous_features=list(map(str, dataset.numerical_features_indices)),
            outcome_name=features[-1],
        )

        logger.info("Creating counterfactual model")
        disc_model_w = DiscWrapper(disc_model)
        model = dice_ml.Model(disc_model_w, backend=self.cfg.counterfactuals_params.backend)
        exp = dice_ml.Dice(dice, model, method=self.cfg.counterfactuals_params.method)

        logger.info("Handling counterfactual generation")
        cf_per_instance = int(self.cfg.counterfactuals_params.generation_params.total_CFs)
        query_instance = pd.DataFrame(X_test_origin, columns=features[:-1])
        time_start = time()

        generation_params = dict(self.cfg.counterfactuals_params.generation_params)
        generation_params["total_CFs"] = cf_per_instance

        cfs = exp.generate_counterfactuals(query_instance, **generation_params)
        cf_search_time = np.mean(time() - time_start)
        logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{self.cf_method_name}_{disc_model_name}.csv"
        )

        # Store first CF per instance (for original metrics)
        Xs_cfs_first_list: list[np.ndarray] = []
        model_returned_first_list: list[bool] = []

        # Store all CFs as 3D array (n_instances, cf_per_instance, n_features)
        Xs_cfs_all_list: list[np.ndarray] = []

        for orig, cf in zip(X_test_origin, cfs.cf_examples_list):
            cf_df = cf.final_cfs_df
            if cf_df is None or cf_df.empty:
                Xs_cfs_first_list.append(orig)
                model_returned_first_list.append(False)
                cf_block = np.repeat(orig[None, :], cf_per_instance, axis=0)
            else:
                cf_array = cf_df.to_numpy()[:, :-1]
                Xs_cfs_first_list.append(cf_array[0])
                model_returned_first_list.append(True)

                cf_block = cf_array[:cf_per_instance]
                if cf_block.shape[0] < cf_per_instance:
                    deficit = cf_per_instance - cf_block.shape[0]
                    padding = np.repeat(orig[None, :], deficit, axis=0)
                    cf_block = np.vstack([cf_block, padding])

            Xs_cfs_all_list.append(cf_block)

        Xs_cfs_first = np.array(Xs_cfs_first_list)
        model_returned_first = np.array(model_returned_first_list)
        Xs_cfs_all = np.stack(Xs_cfs_all_list)  # Shape: (n_instances, cf_per_instance, n_features)
        ys_target = np.abs(1 - y_test_origin)

        # Save all CFs to file (flatten for CSV)
        pd.DataFrame(Xs_cfs_all.reshape(-1, Xs_cfs_all.shape[-1])).to_csv(
            counterfactuals_path, index=False
        )
        logger.info("Counterfactuals saved to %s", counterfactuals_path)

        return SearchResult(
            X_cf=Xs_cfs_first,
            X_test=X_test_origin,
            y_orig=y_test_origin,
            y_target=ys_target,
            model_returned=model_returned_first,
            cf_search_time=cf_search_time,
            extras={"Xs_cfs_all": Xs_cfs_all},
        )
