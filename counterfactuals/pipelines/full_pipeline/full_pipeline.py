import os

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.dequantization.utils import DequantizationWrapper
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths


def full_pipeline(
    cfg, preprocessing_pipeline, logger, search_counterfactuals, calculate_metrics
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info("Loading dataset")
    file_dataset = instantiate(cfg.dataset)
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    dequantizer = GroupDequantizer(dataset.categorical_features_lists)
    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        dequantizer.fit(dataset.X_train)
        gen_model = create_gen_model(cfg, dataset, gen_model_path, dequantizer)

        # Custom code
        dataset.X_train = dequantizer.transform(dataset.X_train)
        log_prob_threshold = get_log_prob_threshold(
            gen_model,
            dataset,
            cfg.counterfactuals_params.batch_size,
            cfg.counterfactuals_params.log_prob_quantile,
            logger,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)
        Xs_cfs, Xs, ys_orig, ys_target, cf_search_time = search_counterfactuals(
            cfg, dataset, gen_model, disc_model, save_folder
        )

        gen_model = DequantizationWrapper(gen_model, dequantizer)

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


def get_log_prob_threshold(
    gen_model: torch.nn.Module,
    dataset: DictConfig,
    batch_size: int,
    log_prob_quantile: float,
    logger,
) -> float:
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
    return log_prob_threshold
