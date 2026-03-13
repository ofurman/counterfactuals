import logging
from time import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from counterfactuals.cf_methods.local_methods.c_chvae.c_chvae import CCHVAE
from counterfactuals.cf_methods.local_methods.c_chvae.data import CustomData
from counterfactuals.cf_methods.local_methods.c_chvae.mlmodel import CustomMLModel
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


class CCHVAEPipelineRunner(PipelineRunner):
    """Pipeline runner for CCHVAE counterfactual generation."""

    cf_method_name = "CCHVAE"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        logger.info("Filtering out target class data for counterfactual generation")
        X_test_origin, y_test_origin = self._filter_test_data(dataset, target_class)

        logger.info("Creating dataset interface")
        custom_dataset = CustomData(dataset)

        logger.info("Creating counterfactual model")
        wrapped_model = CustomMLModel(disc_model, custom_dataset)

        hyperparams = OmegaConf.to_container(
            self.cfg.counterfactuals_params.hyperparams, resolve=True
        )

        input_size = dataset.X_train.shape[1]
        hyperparams["vae_params"]["layers"] = [input_size] + hyperparams["vae_params"]["layers"]

        exp = CCHVAE(wrapped_model, hyperparams)

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        explanation_result = exp.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=self.cfg.counterfactuals_params.epochs,
            lr=self.cfg.counterfactuals_params.lr,
        )
        Xs = explanation_result.x_origs
        Xs_cfs = explanation_result.x_cfs
        ys_orig = explanation_result.y_origs
        ys_target = explanation_result.y_cf_targets

        cf_search_time = np.mean(time() - time_start)
        logger.info(f"Counterfactual search time: {cf_search_time:.4f} seconds")

        model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)
        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="cchvae_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = CCHVAEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
