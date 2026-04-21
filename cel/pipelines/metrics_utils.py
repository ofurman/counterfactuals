"""Utility functions for computing metrics across counterfactual pipelines."""

import numpy as np
from scipy.spatial.distance import pdist


def compute_pairwise_mean_distance(cfs: np.ndarray) -> float:
    """Average minimum pairwise distance across counterfactual sets.

    Args:
        cfs: Array of shape (n_instances, cfs_per_instance, n_features) containing
            multiple counterfactuals per instance.

    Returns:
        Mean of pairwise distances across all instances. Returns NaN if the input
        is empty or has fewer than 2 counterfactuals per instance.
    """
    if cfs.size == 0 or cfs.shape[1] < 2:
        return float("nan")

    mean_dists: list[float] = []
    for group in cfs:
        distances = pdist(group, metric="euclidean")
        if distances.size > 0:
            mean_dists.append(float(distances.mean()))

    return float(np.mean(mean_dists)) if mean_dists else float("nan")
