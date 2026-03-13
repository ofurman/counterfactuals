import logging
from time import time

import dice_ml
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig

from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

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


class DiCEPipelineRunner(PipelineRunner):
    """Pipeline runner for DiCE counterfactual generation."""

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
        query_instance = pd.DataFrame(X_test_origin, columns=features[:-1])
        time_start = time()

        generation_params = dict(self.cfg.counterfactuals_params.generation_params)

        cfs = exp.generate_counterfactuals(query_instance, **generation_params)

        cf_search_time = np.mean(time() - time_start)
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

        Xs_cfs = []
        for orig, cf in zip(X_test_origin, cfs.cf_examples_list):
            if cf.final_cfs_df is None:
                Xs_cfs.append(orig)
                continue
            out = cf.final_cfs_df.to_numpy()
            if out.shape[0] > 0:
                Xs_cfs.append(out[0][:-1])
            else:
                Xs_cfs.append(orig)

        Xs_cfs = np.array(Xs_cfs)
        ys_target = np.abs(1 - y_test_origin)
        model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=X_test_origin,
            y_orig=y_test_origin,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="dice_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = DiCEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
