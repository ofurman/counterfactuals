from __future__ import annotations

import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from counterfactuals.cf_methods import AReS
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


def one_hot(dataset: Any, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode categorical features and record metadata on ``dataset``."""
    label_encoder = LabelEncoder()
    data_encode = data.copy()
    dataset.bins = {}
    dataset.bins_tree = {}
    dataset.features_tree = {}
    dataset.n_bins = None

    # Assign encoded features to one hot columns
    data_oh, features = [], []
    for x in data.columns:
        dataset.features_tree[x] = []
        categorical = x in dataset.categorical_features
        if categorical:
            data_encode[x] = label_encoder.fit_transform(data_encode[x])
            cols = label_encoder.classes_
        elif dataset.n_bins is not None:
            data_encode[x] = pd.cut(data_encode[x].apply(lambda x: float(x)), bins=dataset.n_bins)
            cols = data_encode[x].cat.categories
            dataset.bins_tree[x] = {}
        else:
            data_oh.append(data[x])
            features.append(x)
            continue

        one_hot = pd.get_dummies(data_encode[x])
        data_oh.append(one_hot)
        for col in cols:
            feature_value = x + " = " + str(col)
            features.append(feature_value)
            dataset.features_tree[x].append(feature_value)
            if not categorical:
                dataset.bins[feature_value] = col.mid
                dataset.bins_tree[x][feature_value] = col.mid

    data_oh = pd.concat(data_oh, axis=1, ignore_index=True)
    data_oh.columns = features
    return data_oh, features


def _feature_columns(dataset: Any) -> List[str]:
    """Return feature columns, excluding target if it is part of the list."""
    columns: List[str] = list(dataset.features)
    target: Optional[str] = None
    if hasattr(dataset, "target"):
        target = dataset.target
    elif hasattr(dataset, "config"):
        target = getattr(dataset.config, "target", None)
    if target is not None and target in columns:
        columns = [col for col in columns if col != target]
    return columns


def _get_feature_transformer(dataset: Any):
    """Retrieve a feature transformer if available (e.g., MinMax scaler)."""
    transformer = getattr(dataset, "feature_transformer", None)
    if transformer is not None:
        return transformer
    if hasattr(dataset, "preprocessing_pipeline"):
        return dataset.preprocessing_pipeline.get_step("minmax")
    return None


def _ensure_numpy(array: Any) -> np.ndarray:
    """Convert model outputs to a 1D numpy array."""
    if hasattr(array, "detach"):
        return array.detach().numpy().flatten()
    return np.asarray(array).flatten()


def _build_dataset(cfg: DictConfig) -> Any:
    """Instantiate the configured dataset and fall back to legacy preprocessing."""
    dataset = instantiate(cfg.dataset)
    if hasattr(dataset, "train_dataloader"):
        return dataset

    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    return MethodDataset(dataset, preprocessing_pipeline)


def search_counterfactuals(
    cfg: DictConfig,
    dataset: Any,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Generate regional GLOBE-CE counterfactuals with KMeans clustering.

    This routine clusters eligible test samples, computes bin widths using
    AReS as a helper, then generates counterfactuals per-cluster using
    GLOBE-CE. A plausibility threshold is derived from the generative model.

    Args:
        cfg: Hydra configuration with experiment and method parameters.
        dataset: Dataset object with train/test arrays and metadata.
        gen_model: Trained generative model used for plausibility thresholding.
        disc_model: Trained classifier providing ``predict`` for decision labels.
        save_folder: Output directory for artifacts (e.g., counterfactuals CSV).

    Returns:
        Tuple of:
        - Xs_cfs: Generated counterfactuals (scaled feature space)
        - Xs: Original inputs for which CFs were generated (scaled)
        - log_prob_threshold: Quantile-based log-probability threshold
        - ys_orig: Original predicted labels for Xs
        - ys_target: Target labels for counterfactuals (flipped)
        - model_returned: Boolean mask of successful generations
        - cf_search_time: Average CF search time in seconds
    """
    cf_method_name = "GLOBE_CE"
    disc_model.eval()
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    feature_transformer = _get_feature_transformer(dataset)
    minmax_scaler = None
    if feature_transformer is None:
        minmax_scaler = dataset.preprocessing_pipeline.get_step("minmax")
        X_test_unscaled = minmax_scaler._inverse_transform_array(dataset.X_test)
    else:
        if hasattr(feature_transformer, "_inverse_transform_array"):
            X_test_unscaled = feature_transformer._inverse_transform_array(
                dataset.X_test
            )
        else:
            X_test_unscaled = feature_transformer.inverse_transform(dataset.X_test)
    feature_columns = _feature_columns(dataset)
    data_oh, features = one_hot(
        dataset, pd.DataFrame(X_test_unscaled, columns=feature_columns)
    )

    def predict_fn_raw(x):
        x_array = x.values if isinstance(x, pd.DataFrame) else x
        if feature_transformer is not None:
            if hasattr(feature_transformer, "_transform_array"):
                x_scaled = feature_transformer._transform_array(x_array)
            else:
                x_scaled = feature_transformer.transform(x_array)
        else:
            x_scaled = minmax_scaler._transform_array(x_array)
        preds = disc_model.predict(x_scaled)
        return _ensure_numpy(preds)

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = getattr(cfg.counterfactuals_params, "target_class", 1)
    ys_pred = predict_fn_raw(X_test_unscaled)
    mask = ys_pred != target_class
    Xs_unscaled = X_test_unscaled[mask]
    Xs = dataset.X_test[mask]
    ys_orig = ys_pred[mask]

    # Align AReS expectation (negative class == 0) with configurable target class
    predict_fn_for_cf = (
        (lambda x: 1 - predict_fn_raw(x)) if target_class == 0 else predict_fn_raw
    )

    logger.info("Creating counterfactual model")
    cf_method = AReS(
        predict_fn=predict_fn_for_cf,
        dataset=dataset,
        X=pd.DataFrame(Xs_unscaled, columns=feature_columns),
        dropped_features=[],
        n_bins=10,
        ordinal_features=[],
        normalise=False,
        constraints=[20, 7, 10],
    )
    bin_widths = ares_helper.bin_widths

    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    time_start = time()
    Xs_cfs = cf_method.explain()
    if Xs_cfs.shape[0] > 0:
        if feature_transformer is not None:
            if hasattr(feature_transformer, "_transform_array"):
                Xs_cfs = feature_transformer._transform_array(Xs_cfs)
            else:
                Xs_cfs = feature_transformer.transform(Xs_cfs)
        else:
            Xs_cfs = minmax_scaler._transform_array(Xs_cfs)
    ys_target = np.full_like(ys_orig, target_class)
    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)
    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info("Counterfactuals saved to %s", counterfactuals_path)

    return (
        Xs_cfs,
        Xs,
        log_prob_threshold,
        ys_orig,
        ys_target,
        model_returned,
        cf_search_time,
    )


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: List[int] | List[str],
    continuous_features: List[int] | List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute evaluation metrics for generated counterfactuals.

    Args:
        gen_model: Generative model used for density and plausibility checks.
        disc_model: Discriminative model used for predictions.
        Xs_cfs: Generated counterfactuals in scaled feature space.
        model_returned: Boolean mask indicating for which rows CFs were produced.
        categorical_features: Indices or names of categorical features.
        continuous_features: Indices or names of continuous features.
        X_train: Training features used for metrics referencing the dataset.
        y_train: Training labels aligned to ``X_train``.
        X_test: Original instances for which the CFs were generated.
        y_test: Original labels for ``X_test``.
        median_log_prob: Plausibility threshold for valid CFs.
        y_target: Optional target labels for ``X_test``.

    Returns:
        Mapping from metric names to their computed values.
    """
    logger.info("Calculating metrics")
    metrics = evaluate_cf(
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
        median_log_prob=median_log_prob,
        y_target=y_target,
    )
    logger.info("Metrics computed: %s", metrics)
    return metrics


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Regional GLOBE-CE pipeline: generate and evaluate clustered CFs.

    Orchestrates a 5-fold CV process: loads dataset, trains/loads models,
    clusters eligible test points via KMeans, generates GLOBE-CE CFs per
    cluster, evaluates them with ``evaluate_cf``, and writes metrics/CFs to
    disk.

    Args:
        cfg: Hydra configuration with dataset, model, and CF method settings.
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing pipeline")
    dataset = _build_dataset(cfg)
    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        logger.info("Processing fold %d", fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = _ensure_numpy(disc_model.predict(dataset.X_train))
            dataset.y_test = _ensure_numpy(disc_model.predict(dataset.X_test))

        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        (
            Xs_cfs,
            Xs,
            log_prob_threshold,
            ys_orig,
            ys_target,
            model_returned,
            cf_search_time,
        ) = search_counterfactuals(cfg, dataset, gen_model, disc_model, save_folder)

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )

        logger.info("Metrics: %s", metrics)
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
