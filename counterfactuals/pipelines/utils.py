"""Shared pipeline utilities."""

from __future__ import annotations

from typing import List

import numpy as np


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
        one_hot = np.zeros(
            (discretized.shape[0], len(interval)), dtype=discretized.dtype
        )
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
