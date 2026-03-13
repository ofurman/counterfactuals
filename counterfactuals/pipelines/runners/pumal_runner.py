import logging
from time import time

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.pumal import PUMAL
from counterfactuals.metrics.metrics import evaluate_cf_for_pumal
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


class PUMALPipelineRunner(PipelineRunner):
    """Pipeline runner for PUMAL counterfactual generation."""

    cf_method_name = "PUMAL"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]

        logger.info("Filtering out target class data for counterfactual generation")
        origin_class = self.cfg.counterfactuals_params.origin_class
        target_class = self.cfg.counterfactuals_params.target_class
        y_test = dataset.y_test
        y_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test.reshape(-1)
        y_indices = y_labels.astype(int)
        mask_origin = y_indices == origin_class
        X_test_origin = dataset.X_test[mask_origin]
        if y_test.ndim > 1:
            y_test_origin = y_test[mask_origin]
        else:
            n_classes = int(np.max(y_indices)) + 1
            y_test_origin = np.eye(n_classes)[y_indices][mask_origin]
        actionable_features = getattr(dataset, "actionable_features", None)
        not_actionable_features = None
        if actionable_features:
            not_actionable_features = [
                idx
                for idx, feature in enumerate(dataset.features)
                if feature not in actionable_features
            ]

        logger.info("Creating counterfactual model")
        disc_model_criterion = instantiate(self.cfg.counterfactuals_params.disc_model_criterion)
        cf_method = PUMAL(
            cf_method_type=self.cfg.counterfactuals_params.cf_method.cf_method_type,
            K=self.cfg.counterfactuals_params.cf_method.K,
            X=X_test_origin,
            gen_model=gen_model,
            disc_model=disc_model,
            disc_model_criterion=disc_model_criterion,
            not_actionable_features=not_actionable_features,
        )

        logger.info("Handling counterfactual generation")
        cf_dataloader = self._create_cf_dataloader(
            X_test_origin, y_test_origin, self.cfg.counterfactuals_params.batch_size
        )
        time_start = time()
        delta, Xs, _, _ = cf_method.explain_dataloader(
            dataloader=cf_dataloader,
            target_class=target_class,
            epochs=self.cfg.counterfactuals_params.epochs,
            lr=self.cfg.counterfactuals_params.lr,
            patience=self.cfg.counterfactuals_params.patience,
            alpha_dist=self.cfg.counterfactuals_params.alpha_dist,
            alpha_plaus=self.cfg.counterfactuals_params.alpha_plaus,
            alpha_class=self.cfg.counterfactuals_params.alpha_class,
            alpha_s=self.cfg.counterfactuals_params.alpha_s,
            alpha_k=self.cfg.counterfactuals_params.alpha_k,
            alpha_d=self.cfg.counterfactuals_params.alpha_d,
            log_prob_threshold=log_prob_threshold,
            decrease_loss_patience=self.cfg.counterfactuals_params.decrease_loss_patience,
        )

        cf_search_time = np.mean(time() - time_start)
        Xs_cfs = Xs + delta().detach().numpy()
        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        ys_orig = y_indices[mask_origin]
        ys_target = np.full_like(ys_orig, target_class)
        model_returned = np.ones(Xs_cfs.shape[0], dtype=bool)

        _, S_matrix, D_matrix = cf_method.delta.get_matrices()
        extras = {
            "S_matrix": S_matrix.detach().cpu().numpy()
            if hasattr(S_matrix, "detach")
            else np.asarray(S_matrix),
            "D_matrix": D_matrix.detach().cpu().numpy()
            if hasattr(D_matrix, "detach")
            else np.asarray(D_matrix),
        }

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
            extras=extras,
        )

    def calculate_metrics(self, gen_model, disc_model, dataset, result, log_prob_threshold):
        """Calculate evaluation metrics for generated counterfactuals."""
        logger.info("Calculating metrics")
        metrics = evaluate_cf_for_pumal(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=result.X_cf,
            model_returned=result.model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=result.X_test,
            y_test=result.y_orig,
            median_log_prob=log_prob_threshold,
            y_target=result.y_target,
            S_matrix=result.extras.get("S_matrix"),
            D_matrix=result.extras.get("D_matrix"),
            metrics_conf_path="counterfactuals/pipelines/conf/metrics/group_metrics.yaml",
        )
        logger.info(f"Metrics:\n{metrics}")
        return metrics


@hydra.main(config_path="./conf", config_name="pumal_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = PUMALPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
