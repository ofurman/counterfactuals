import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler

from cel.cf_methods.local_methods.cearm import CEARM
from cel.metrics import evaluate_cf_regression
from cel.pipelines.nodes.disc_model_nodes import create_disc_model
from cel.pipelines.nodes.gen_model_nodes import create_gen_model
from cel.pipelines.nodes.helper_nodes import set_model_paths
from cel.datasets.method_dataset import MethodDataset
from cel.preprocessing import (
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Generate counterfactuals using the PPCEFR (regression) method.

    Prepares the PPCEFR method with the generative and discriminative models, computes
    a delta threshold from the generative model outputs, and generates counterfactuals
    for the regression setting.

    Args:
        cfg: Hydra configuration with experiment parameters
        dataset: Dataset object with features/targets and metadata
        gen_model: Trained generative model (used for delta threshold)
        disc_model: Trained discriminative/regression model
        save_folder: Directory where counterfactuals CSV will be saved

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactuals
            - Xs: Original instances
            - ys_orig: Original targets
            - ys_target: Target values for counterfactuals
            - delta: Threshold derived from generative model
            - cf_search_time: Average time taken for CF search
    """
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    X_test_origin = dataset.X_test
    y_test_origin = dataset.y_test

    logger.info("Creating counterfactual model")
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_loss)

    cf_method = CEARM(
        disc_model=disc_model,
    )

    logger.info("Calculating delta threshold")
    train_dataloader_for_delta = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    delta = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_delta),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    logger.info(f"delta: {delta:.4f}")

    logger.info("Handling counterfactual generation")
    cf_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )

    time_start = time()
    x_cfs, x_origs, y_origs, y_cf_targets, logs = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        target_change=cfg.counterfactuals_params.target_change,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        alpha=cfg.counterfactuals_params.alpha,
        delta=delta,
    )

    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_wach_{cf_method_name}_{disc_model_name}.csv"
    )

    pd.DataFrame(x_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)
    return x_cfs, x_origs, y_origs, y_cf_targets, delta, cf_search_time


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    delta: float,
    y_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for regression counterfactuals.

    Uses the regression evaluation suite to assess the quality of counterfactuals
    with respect to the discriminative and generative models.
    """
    logger.info("Calculating metrics")
    metrics = evaluate_cf_regression(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        median_log_prob=delta,
        y_target=y_target,
    )
    logger.info(f"Metrics:\n{metrics}")
    return metrics


@hydra.main(config_path="./conf", config_name="cearm_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline for CEARM counterfactual generation and evaluation (regression).

    Loads dataset and models, generates regression counterfactuals with CEARM, and
    computes evaluation metrics. Results are logged and saved to CSV files.

    Args:
        cfg: Hydra configuration including dataset, model, and CF parameters
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    file_dataset = instantiate(cfg.dataset)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    cv_splits = (
        dataset.get_cv_splits(cfg.experiment.num_folds)
        if cfg.experiment.num_folds > 1
        else [None]
    )
    for fold_n, split in enumerate(cv_splits):
        if split is not None:
            dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test = split
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        if dataset.y_train.ndim == 1:
            dataset.y_train = dataset.y_train.reshape(-1, 1)
        if dataset.y_test.ndim == 1:
            dataset.y_test = dataset.y_test.reshape(-1, 1)
        target_scaler = MinMaxScaler()
        dataset.y_train = target_scaler.fit_transform(dataset.y_train).astype(
            np.float32
        )
        dataset.y_test = target_scaler.transform(dataset.y_test).astype(np.float32)
        dataset.y_train = np.clip(dataset.y_train, 0.0, 1.0)
        dataset.y_test = np.clip(dataset.y_test, 0.0, 1.0)
        dataset.target_scaler = target_scaler

        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = np.clip(
                np.asarray(disc_model.predict(dataset.X_train), dtype=np.float32),
                0.0,
                1.0,
            )
            dataset.y_test = np.clip(
                np.asarray(disc_model.predict(dataset.X_test), dtype=np.float32),
                0.0,
                1.0,
            )

        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        Xs_cfs, Xs, ys_orig, ys_target, delta, cf_search_time = search_counterfactuals(
            cfg, dataset, gen_model, disc_model, save_folder
        )

        # # Create 2D dataset plot with model predictions, contours, and counterfactuals
        # dataset_plot_path = os.path.join(
        #     save_folder, f"dataset_2d_plot_{cf_method_name}_{disc_model_name}.pdf"
        # )
        # plot_2d_regression_dataset(
        #     gen_model=gen_model,
        #     disc_model=disc_model,
        #     X_test=dataset.X_test,
        #     X_cfs=Xs_cfs,
        #     X_origs=Xs,
        #     y_origs=ys_orig,
        #     y_targets=ys_target,
        #     delta=delta,
        #     num_points=3,
        #     save_path=dataset_plot_path,
        # )
        #
        # # # Create 3D plot if data has exactly 2 features
        # # plot_path = os.path.join(
        # #     save_folder, f"cf_3d_plot_{cf_method_name}_{disc_model_name}.pdf"
        # # )
        # # plot_3d_regression_cfs(
        # #     gen_model=gen_model,
        # #     X_cfs=Xs_cfs,
        # #     X_origs=Xs,
        # #     y_origs=ys_orig,
        # #     y_targets=ys_target,
        # #     delta=delta,
        # #     num_points=3,
        # #     save_path=plot_path,
        # # )
        #
        # # Create conditional density contour plot
        # contour_path = os.path.join(
        #     save_folder, f"cf_contour_plot_{cf_method_name}_{disc_model_name}.pdf"
        # )
        # plot_conditional_density_contours(
        #     gen_model=gen_model,
        #     disc_model=disc_model,
        #     X_cfs=Xs_cfs,
        #     X_origs=Xs,
        #     y_origs=ys_orig,
        #     y_targets=ys_target,
        #     delta=delta,
        #     num_points=3,
        #     save_path=contour_path,
        # )
        #
        # # Create plausibility comparison plot
        # plausibility_path = os.path.join(
        #     save_folder, f"cf_plausibility_plot_{cf_method_name}_{disc_model_name}.pdf"
        # )
        # plot_plausibility_comparison(
        #     gen_model=gen_model,
        #     X_cfs=Xs_cfs,
        #     X_origs=Xs,
        #     y_origs=ys_orig,
        #     y_targets=ys_target,
        #     delta=delta,
        #     save_path=plausibility_path,
        # )

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            delta=delta,
        )
        logger.info(f"Final metrics: {metrics}")
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )

if __name__ == "__main__":
    main()