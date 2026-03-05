"""Input validation utilities for metrics."""

import logging
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def validate_metric_inputs(
    X_cf: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_target: np.ndarray,
    continuous_features: list[int],
    categorical_features: list[int],
    ratio_cont: float | None = None,
) -> None:
    """
    Validate all input data for metrics computation.

    Args:
        X_cf: Counterfactual instances.
        X_train: Training instances.
        X_test: Test instances.
        y_train: Training labels.
        y_test: Test labels.
        y_target: Target labels.
        continuous_features: List of continuous feature indices.
        categorical_features: List of categorical feature indices.
        ratio_cont: Optional ratio of continuous features.

    Raises:
        AssertionError: If any validation check fails.
    """
    # Shape validations
    assert X_cf.shape[1] == X_train.shape[1] == X_test.shape[1], (
        f"All input data should have the same number of features. "
        f"Got X_cf: {X_cf.shape[1]}, X_train: {X_train.shape[1]}, X_test: {X_test.shape[1]}"
    )

    assert X_train.shape[0] == y_train.shape[0], (
        f"X_train and y_train should have the same number of samples. "
        f"Got X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"
    )

    assert X_test.shape[0] == y_test.shape[0], (
        f"X_test and y_test should have the same number of samples. "
        f"Got X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}"
    )

    assert X_cf.shape[0] == y_test.shape[0], (
        f"X_cf and y_test should have the same number of samples. "
        f"Got X_cf: {X_cf.shape[0]}, y_test: {y_test.shape[0]}"
    )

    assert X_cf.shape[0] == y_target.shape[0], (
        f"X_cf and y_target should have the same number of samples. "
        f"Got X_cf: {X_cf.shape[0]}, y_target: {y_target.shape[0]}"
    )

    # Feature validations
    n_features = X_cf.shape[1]
    assert len(continuous_features) + len(categorical_features) == n_features, (
        f"The sum of continuous and categorical features should equal the number of features. "
        f"Got {len(continuous_features)} + {len(categorical_features)} = "
        f"{len(continuous_features) + len(categorical_features)}, expected {n_features}"
    )

    # Check for duplicate feature indices
    feature_set = set(continuous_features) | set(categorical_features)
    assert len(feature_set) == n_features, (
        f"Features contain duplicates or don't cover all indices. "
        f"Got {len(feature_set)} unique features, expected {n_features}"
    )

    # Check feature indices are within bounds
    assert all(0 <= f < n_features for f in continuous_features), (
        f"All continuous feature indices must be in range [0, {n_features})"
    )
    assert all(0 <= f < n_features for f in categorical_features), (
        f"All categorical feature indices must be in range [0, {n_features})"
    )

    # Ratio validation
    if ratio_cont is not None:
        assert 0 <= ratio_cont <= 1, f"ratio_cont should be between 0 and 1, got {ratio_cont}"

    # Type validations
    assert isinstance(X_cf, np.ndarray), f"X_cf should be numpy array, got {type(X_cf)}"
    assert isinstance(X_train, np.ndarray), f"X_train should be numpy array, got {type(X_train)}"
    assert isinstance(X_test, np.ndarray), f"X_test should be numpy array, got {type(X_test)}"

    # Dimensionality validations
    assert X_cf.ndim == 2, f"X_cf should be 2D array, got {X_cf.ndim}D"
    assert X_train.ndim == 2, f"X_train should be 2D array, got {X_train.ndim}D"
    assert X_test.ndim == 2, f"X_test should be 2D array, got {X_test.ndim}D"

    logger.debug("All metric inputs validated successfully")


def convert_to_numpy(X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert input data to numpy array.

    Args:
        X: Input data.

    Returns:
        Converted numpy array.

    Raises:
        ValueError: If X is neither a numpy array nor a torch tensor.
    """
    if isinstance(X, np.ndarray):
        return X
    elif isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    else:
        raise ValueError(f"X should be either a numpy array or a torch tensor, got {type(X)}")
