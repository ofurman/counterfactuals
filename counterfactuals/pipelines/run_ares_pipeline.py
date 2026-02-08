from __future__ import annotations

import copy
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
from sklearn.preprocessing import LabelEncoder

from counterfactuals.cf_methods import AReS
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.pipelines.utils import align_counterfactuals_with_factuals
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _set_dataset_attribute(dataset: Any, attribute: str, value: Any) -> None:
    """Set an attribute on a dataset or its underlying file dataset (MethodDataset)."""
    try:
        setattr(dataset, attribute, value)
        return
    except AttributeError:
        pass

    if hasattr(dataset, "file_dataset"):
        setattr(dataset.file_dataset, attribute, value)
        return

    raise


def _infer_one_hot_category(base_feature: str, column: str) -> str:
    """Infer the category label from a one-hot column name."""
    if not column.startswith(base_feature):
        return column

    suffix = column[len(base_feature) :]
    for sep in (" = ", "__", "=", "_"):
        if suffix.startswith(sep):
            return suffix[len(sep) :]
    return suffix.lstrip(" _=")


def _build_features_tree_from_one_hot(
    dataset: Any, data: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a features tree when data is already one-hot encoded.

    Uses ``one_hot_feature_groups`` (produced by initial transforms) to group one-hot columns
    under their original base categorical feature. Columns are renamed to follow
    the ``"<feature> = <value>"`` convention used by AReS.
    """
    groups = getattr(dataset, "one_hot_feature_groups", None)
    if groups is None and hasattr(dataset, "file_dataset"):
        groups = getattr(dataset.file_dataset, "one_hot_feature_groups", None)

    dataset.bins = {}
    dataset.bins_tree = {}
    dataset.features_tree = {}
    dataset.n_bins = None

    columns = list(data.columns)
    if not groups:
        dataset.features_tree = {col: [] for col in columns}
        return data.copy(), columns

    group_lookup = {
        column: base_feature
        for base_feature, group_columns in groups.items()
        for column in group_columns
    }

    data_transformed = data.copy()
    transformed_columns: list[str] = []
    for column in columns:
        base_feature = group_lookup.get(column)
        if base_feature is None:
            dataset.features_tree[column] = []
            transformed_columns.append(column)
            continue

        category = _infer_one_hot_category(base_feature, column)
        feature_value = f"{base_feature} = {category}" if category else column
        dataset.features_tree.setdefault(base_feature, []).append(feature_value)
        transformed_columns.append(feature_value)

    data_transformed.columns = transformed_columns
    _set_dataset_attribute(dataset, "features", transformed_columns)
    _set_dataset_attribute(
        dataset,
        "categorical_features",
        [feature for feature, values in dataset.features_tree.items() if values],
    )
    return data_transformed, transformed_columns


def one_hot(dataset: Any, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode categorical features and record metadata on ``dataset``."""
    if getattr(dataset, "one_hot_feature_groups", None) or (
        hasattr(dataset, "file_dataset")
        and getattr(dataset.file_dataset, "one_hot_feature_groups", None)
    ):
        return _build_features_tree_from_one_hot(dataset, data)

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
            data_encode[x] = pd.cut(
                data_encode[x].apply(lambda x: float(x)), bins=dataset.n_bins
            )
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
    _set_dataset_attribute(dataset, "features", features)
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
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactuals using the AReS method.

    This function applies the AReS counterfactual explanation method to generate
    counterfactuals for instances that don't belong to the target class.

    Args:
        cfg: Hydra configuration containing counterfactual parameters
        dataset: Dataset containing training and test data
        gen_model: Pre-trained generative model
        disc_model: Pre-trained discriminative model
        save_folder: Directory path where counterfactuals will be saved

    Returns:
        tuple: A tuple containing:
            - Xs_cfs (np.ndarray): Generated counterfactual examples
            - Xs (np.ndarray): Original examples used for counterfactual generation
            - log_prob_threshold (float): Calculated log probability threshold
            - ys_orig (np.ndarray): Original predicted labels
            - ys_target (np.ndarray): Target labels for counterfactuals
            - model_returned (np.ndarray): Boolean array indicating successful generation
            - cf_search_time (float): Time taken for counterfactual search in seconds
    """
    cf_method_name = "ARES"
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
    ares_dataset = copy.deepcopy(dataset)
    X_test_for_ares, _ = one_hot(
        ares_dataset, pd.DataFrame(X_test_unscaled, columns=feature_columns)
    )

    def predict_fn_raw(x: pd.DataFrame | np.ndarray) -> np.ndarray:
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
    Xs_for_ares = X_test_for_ares.loc[mask].reset_index(drop=True)
    Xs = dataset.X_test[mask]
    ys_orig = ys_pred[mask]

    # Align AReS expectation (negative class == 0) with configurable target class
    predict_fn_for_cf = (
        (lambda x: 1 - predict_fn_raw(x)) if target_class == 0 else predict_fn_raw
    )

    logger.info("Creating counterfactual model")
    apriori_threshold = float(
        getattr(cfg.counterfactuals_params, "apriori_threshold", 0.6)
    )
    n_bins = int(getattr(cfg.counterfactuals_params, "n_bins", 10))
    max_triples_eval = int(
        getattr(cfg.counterfactuals_params, "max_triples_eval", 5000)
    )
    cf_method = AReS(
        predict_fn=predict_fn_for_cf,
        dataset=ares_dataset,
        X=Xs_for_ares,
        dropped_features=[],
        n_bins=n_bins,
        ordinal_features=[],
        normalise=False,
        constraints=[20, 7, 10],
    )
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    logger.info("Handling counterfactual generation")
    time_start = time()
    ys_target = np.full_like(ys_orig, target_class)
    explanation_result = cf_method.explain(
        apriori_threshold=apriori_threshold,
        max_triples_eval=max_triples_eval,
        y_origin=ys_orig,
        y_target=ys_target,
    )
    Xs_cfs = explanation_result.x_cfs
    if Xs_cfs.shape[0] > 0:
        if feature_transformer is not None:
            if hasattr(feature_transformer, "_transform_array"):
                Xs_cfs = feature_transformer._transform_array(Xs_cfs)
            else:
                Xs_cfs = feature_transformer.transform(Xs_cfs)
        else:
            Xs_cfs = minmax_scaler._transform_array(Xs_cfs)
    Xs_cfs, model_returned = align_counterfactuals_with_factuals(Xs_cfs, Xs)
    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search time: {cf_search_time:.2f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to {counterfactuals_path}")

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
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for generated counterfactuals.

    Evaluates the quality of counterfactuals using various metrics including validity,
    plausibility, proximity, and diversity measures.

    Args:
        gen_model: Generative model used for plausibility assessment
        disc_model: Discriminative model used for validity assessment
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean array indicating successful counterfactual generation
        categorical_features: List of categorical feature indices
        continuous_features: List of continuous feature indices
        X_train: Training data features
        y_train: Training data labels
        X_test: Original test examples
        y_test: Original test labels
        median_log_prob: Log probability threshold for plausibility
        y_target: Target labels for counterfactuals (optional)

    Returns:
        dict: Dictionary containing computed metrics
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
    logger.info(f"Metrics:\n{metrics}")
    return metrics


@hydra.main(config_path="./conf", config_name="ares_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline function for AReS counterfactual generation and evaluation.

    This function orchestrates the complete counterfactual analysis pipeline using
    the AReS method including:
    - Dataset loading and cross-validation setup
    - Discriminative and generative model creation/loading
    - Counterfactual generation using AReS method
    - Metrics calculation and results saving

    The pipeline processes multiple CV folds and saves results for each fold.

    Args:
        cfg: Hydra configuration containing all pipeline parameters including
             dataset configuration, model parameters, and counterfactual generation settings

    Returns:
        None: Results are saved to files and logged
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing pipeline")
    dataset = _build_dataset(cfg)
    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
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
        logger.info(
            "Fold %s counterfactual search time: %.4f seconds", fold_n, cf_search_time
        )

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

        logger.info(f"Metrics for fold {fold_n}: {metrics}")
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics["cf_search_time"] = cf_search_time
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
