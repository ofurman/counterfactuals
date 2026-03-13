"""Pipeline runner for Regional GLOBE-CE counterfactual generation."""

import logging
import os
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from counterfactuals.cf_methods import GLOBE_CE, AReS
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.pipelines.utils import one_hot

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class RegionalGLOBECEPipelineRunner(PipelineRunner):
    """Pipeline runner for Regional GLOBE-CE counterfactual generation.

    Uses raw dataset without MethodDataset preprocessing, no dequantizer,
    and performs KMeans clustering followed by per-cluster GLOBE-CE with
    shared bin widths from AReS.
    """

    cf_method_name = "GLOBE_CE"

    def run(self) -> None:
        """Custom run implementation for Regional GLOBE-CE with raw dataset."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.logger.info("Loading dataset")
        dataset = instantiate(self.cfg.dataset, shuffle=False)

        for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
            disc_model_path, gen_model_path, save_folder = set_model_paths(self.cfg, fold=fold_n)
            self.logger.info("Processing fold %d", fold_n)
            disc_model = create_disc_model(self.cfg, dataset, disc_model_path, save_folder)

            if self.cfg.experiment.relabel_with_disc_model:
                dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
                dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

            gen_model = create_gen_model(self.cfg, dataset, gen_model_path)

            result = self.search_counterfactuals(dataset, gen_model, disc_model, save_folder, None)

            metrics = self.calculate_metrics(gen_model, disc_model, dataset, result, None)

            self.save_results(metrics, result.cf_search_time, save_folder)

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model.eval()
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]

        X_test_unscaled = dataset.feature_transformer.inverse_transform(dataset.X_test)
        data_oh, features = one_hot(
            dataset, pd.DataFrame(X_test_unscaled, columns=dataset.features[:-1])
        )

        def predict_fn(x):
            x_scaled = dataset.feature_transformer.transform(x)
            return disc_model.predict(x_scaled).detach().numpy().flatten()

        logger.info("Filtering out target class data for counterfactual generation")
        target_class = 1
        ys_pred = predict_fn(X_test_unscaled)
        Xs = dataset.X_test[ys_pred != target_class]
        X_test_unscaled = X_test_unscaled[ys_pred != target_class]
        ys_orig = ys_pred[ys_pred != target_class]

        ares_helper = AReS(
            predict_fn=predict_fn,
            dataset=dataset,
            X=pd.DataFrame(X_test_unscaled, columns=dataset.features[:-1]),
            dropped_features=[],
            n_bins=10,
            ordinal_features=[],
            normalise=False,
            constraints=[20, 7, 10],
        )
        bin_widths = ares_helper.bin_widths

        logger.info("Calculating log_prob_threshold")
        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=self.cfg.counterfactuals_params.batch_size, shuffle=False
        )
        log_prob_threshold = torch.quantile(
            gen_model.predict_log_prob(train_dataloader_for_log_prob),
            self.cfg.counterfactuals_params.log_prob_quantile,
        )
        logger.info("log_prob_threshold: %.4f", log_prob_threshold)

        time_start = time()
        k_means = KMeans(n_clusters=10)
        clusters_id = k_means.fit_predict(Xs)
        Xs_cfs = np.empty_like(Xs)
        for label in range(10):
            logger.info("Creating counterfactual model for cluster %d", label)
            cf_method = GLOBE_CE(
                predict_fn=predict_fn,
                dataset=dataset,
                X=pd.DataFrame(
                    X_test_unscaled[clusters_id == label], columns=dataset.features[:-1]
                ),
                bin_widths=bin_widths,
            )

            logger.info("Handling counterfactual generation for cluster %d", label)
            Xs_cfs[clusters_id == label] = cf_method.explain()
            Xs_cfs[clusters_id == label] = dataset.feature_transformer.transform(
                Xs_cfs[clusters_id == label]
            )

        ys_target = np.abs(ys_orig - 1)
        model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)
        cf_search_time = np.mean(time() - time_start)
        logger.info("Counterfactual search completed in %.4f seconds", cf_search_time)

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{self.cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        logger.info("Counterfactuals saved to %s", counterfactuals_path)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="regional_globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = RegionalGLOBECEPipelineRunner(cfg, logger, None)
    runner.run()


if __name__ == "__main__":
    main()
