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
        one_hot = np.zeros((discretized.shape[0], len(interval)), dtype=discretized.dtype)
        one_hot[np.arange(discretized.shape[0]), max_indices] = 1.0
        discretized[:, interval] = one_hot

    return discretized
