import logging
import os
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.group_methods.tcrex import TCREx
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
):
    """
    Create a counterfactual model using TCREx method
    """

    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    origin_class = cfg.counterfactuals_params.origin_class
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test == origin_class]
    y_test_origin = dataset.y_test[dataset.y_test == origin_class]

    logger.info("Creating TCREx counterfactual model")
    # Create the TCREx instance with configuration parameters
    cf_method = TCREx(
        target_model=disc_model,
        tau=cfg.counterfactuals_params.tau,
        rho=cfg.counterfactuals_params.rho,
        surrogate_tree_params=cfg.counterfactuals_params.surrogate_tree_params,
    )

    # Fit the TCREx model on training data
    logger.info("Fitting the TCREx model")
    time_start = time()

    # Use training data with labels for fitting the surrogate tree
    # cf_method.fit(dataset.X_train, np.argmax(dataset.y_train, axis=1))
    cf_method.fit(dataset.X_train, dataset.y_train)

    # Generate counterfactuals for the test instances
    logger.info("Generating counterfactuals")
    Xs_cfs = cf_method.explain(X_test_origin)

    cf_search_time = np.mean(time() - time_start)

    # Save the counterfactuals
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)

    # For compatibility with the metrics calculation, calculate log_prob_threshold
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )

    n_groups = cf_method.n_groups_

    # Return counterfactuals and related info
    return (
        Xs_cfs,
        X_test_origin,
        log_prob_threshold,
        y_test_origin,
        np.full_like(y_test_origin, target_class),
        cf_search_time,
        n_groups,
    )


@hydra.main(config_path="./conf", config_name="tcrex_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            # ("label_onehot", LabelOneHotEncodingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )

    file_dataset = instantiate(cfg.dataset)
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    label_onehot_encoder = dataset.preprocessing_pipeline.get_step("label_onehot")

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, cf_search_time, n_groups = (
            search_counterfactuals(cfg, dataset, gen_model, disc_model, save_folder)
        )

        logger.info("Calculating metrics")
        metrics = evaluate_cf(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics["time"] = cf_search_time
        df_metrics["n_groups"] = n_groups
        logger.info(f"Metrics:\n{metrics}")
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
