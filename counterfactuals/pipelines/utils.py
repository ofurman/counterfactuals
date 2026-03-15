"""Shared pipeline utilities."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def apply_categorical_discretization(
    categorical_features_lists: List[List[int]], samples: np.ndarray
) -> np.ndarray:
    """Discretize categorical feature groups into one-hot values.

    Args:
        categorical_features_lists: List of index groups for categorical features.
        samples: Array of samples with shape (n_samples, n_features).

    Returns:
        Copy of samples with categorical groups snapped to valid one-hot values.
    """
    if not categorical_features_lists:
        return samples

    discretized = samples.copy()
    for interval in categorical_features_lists:
        if not interval:
            continue
        max_indices = np.argmax(discretized[:, interval], axis=1)
        one_hot = np.zeros((discretized.shape[0], len(interval)), dtype=discretized.dtype)
        one_hot[np.arange(discretized.shape[0]), max_indices] = 1.0
        discretized[:, interval] = one_hot

    return discretized


def align_counterfactuals_with_factuals(
    x_cfs: np.ndarray, x_factuals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align generated counterfactuals to factual rows.

    When a method returns fewer rows than requested, the remaining rows are
    filled with the original factual instances. This preserves row alignment and
    allows metrics to report failed searches through zero validity.

    Args:
        x_cfs: Generated counterfactuals, shape ``(n_returned, n_features)``.
        x_factuals: Queried factual instances, shape ``(n_expected, n_features)``.

    Returns:
        Tuple of:
            - Aligned counterfactual array of shape ``(n_expected, n_features)``.
            - Boolean mask of shape ``(n_expected,)`` indicating rows originally
              returned by the model.
    """
    if x_factuals.ndim != 2:
        raise ValueError(
            f"x_factuals must be 2D with shape (n_samples, n_features), got {x_factuals.shape}"
        )

    if x_cfs.ndim == 1:
        if x_factuals.shape[1] == 1:
            x_cfs = x_cfs.reshape(-1, 1)
        else:
            x_cfs = x_cfs.reshape(1, -1)
    elif x_cfs.ndim != 2:
        raise ValueError(
            f"x_cfs must be 1D or 2D with shape (n_samples, n_features), got {x_cfs.shape}"
        )

    if x_cfs.shape[1] != x_factuals.shape[1]:
        raise ValueError(
            "x_cfs and x_factuals must have the same number of features. "
            f"Got {x_cfs.shape[1]} and {x_factuals.shape[1]}."
        )

    n_expected = x_factuals.shape[0]
    n_returned = min(x_cfs.shape[0], n_expected)

    aligned = x_factuals.copy()
    if n_returned > 0:
        aligned[:n_returned] = x_cfs[:n_returned]

    model_returned = np.zeros(n_expected, dtype=bool)
    model_returned[:n_returned] = True
    return aligned, model_returned


def one_hot(dataset: Any, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Apply one-hot encoding to categorical features in the dataset.

    Mirrors the AReS preprocessing utility: encodes categoricals and optionally
    bins continuous features, updating dataset metadata for later use.

    Args:
        dataset: Dataset object that will be updated with encoding metadata.
        data: DataFrame with raw features to be encoded.

    Returns:
        Tuple containing:
            - data_oh: One-hot encoded DataFrame.
            - features: List of feature names after encoding.
    """
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


# ---------------------------------------------------------------------------
# One-hot feature-tree builders (shared between AReS and GLOBE-CE runners)
# ---------------------------------------------------------------------------


def _set_dataset_attribute(dataset: Any, attribute: str, value: Any) -> None:
    """Set an attribute on a dataset, falling back to file_dataset if needed.

    Args:
        dataset: Dataset object to update.
        attribute: Name of the attribute to set.
        value: Value to assign.

    Raises:
        AttributeError: If neither dataset nor its file_dataset supports the attribute.
    """
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
    """Infer a human-readable category name from a one-hot column name.

    Args:
        base_feature: The base feature name (e.g. ``"color"``).
        column: The one-hot column name (e.g. ``"color = red"`` or ``"color__red"``).

    Returns:
        The inferred category string (e.g. ``"red"``), or the full column name
        if no separator is recognised.
    """
    if not column.startswith(base_feature):
        return column

    suffix = column[len(base_feature) :]
    for sep in (" = ", "__", "=", "_"):
        if suffix.startswith(sep):
            return suffix[len(sep) :]
    return suffix.lstrip(" _=")


def _build_features_tree_from_one_hot(
    dataset: Any, data: pd.DataFrame, rename_columns: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a features tree from one-hot encoded column groups.

    When ``rename_columns=True`` (AReS behavior), column names are rewritten to
    human-readable ``"feature = category"`` strings and dataset metadata attributes
    (``features``, ``categorical_features``) are updated via
    :func:`_set_dataset_attribute`.

    When ``rename_columns=False`` (GLOBE-CE behavior), original column names are
    preserved and no metadata attributes are updated.

    Args:
        dataset: Dataset object whose ``bins``, ``bins_tree``, ``features_tree``,
            and ``n_bins`` attributes will be initialised.
        data: DataFrame of unscaled feature values.
        rename_columns: If ``True``, apply AReS-style column renaming and metadata
            updates. If ``False``, use GLOBE-CE's lighter behavior.

    Returns:
        Tuple of (transformed_data, column_names).
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
        if rename_columns:
            dataset.features_tree = {col: [] for col in columns}
        return data.copy(), columns

    group_lookup = {
        column: base_feature
        for base_feature, group_columns in groups.items()
        for column in group_columns
    }

    if not rename_columns:
        # GLOBE-CE behavior: build features_tree without renaming
        added_groups: set[str] = set()
        for column in columns:
            base = group_lookup.get(column)
            if base is None:
                dataset.features_tree[column] = []
                continue
            if base in added_groups:
                continue
            grouped_columns = [c for c in columns if group_lookup.get(c) == base]
            dataset.features_tree[base] = grouped_columns
            added_groups.add(base)
        return data.copy(), columns

    # AReS behavior: rename columns using _infer_one_hot_category
    data_transformed = data.copy()
    transformed_columns: List[str] = []
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
