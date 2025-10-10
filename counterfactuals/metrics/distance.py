"""Distance metrics for counterfactual evaluation."""

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import _validate_vector, cdist

from counterfactuals.metrics.base import Metric
from counterfactuals.metrics.utils import register_metric

logger = logging.getLogger(__name__)


def _median_abs_deviation(data: np.ndarray, axis=None) -> np.ndarray:
    """
    Calculate the Median Absolute Deviation (MAD) of a dataset along a specified axis.

    Args:
        data: The input data for which the MAD is to be computed.
        axis: The axis along which the median should be computed.

    Returns:
        The MAD of the data along the given axis.
    """
    median = np.median(data, axis=axis)
    if axis is None:
        deviations = np.abs(data - median)
    else:
        deviations = np.abs(data - np.expand_dims(median, axis=axis))
    mad = np.median(deviations, axis=axis)
    return mad


def _mad_cityblock(u: np.ndarray, v: np.ndarray, mad: np.ndarray) -> float:
    """
    Calculate the MAD cityblock distance between two vectors.

    Args:
        u: The first vector.
        v: The second vector.
        mad: The Median Absolute Deviation (MAD) values.

    Returns:
        The MAD cityblock distance between the two vectors.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


class BaseDistanceMetric(Metric):
    """Base class for distance metrics."""

    name: str

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {
            "X_test",
            "X_cf",
            "y_test",
            "y_target",
            "y_cf_pred",
            "continuous_features",
            "categorical_features",
        }

    def _filter_valid(
        self,
        X_test: np.ndarray,
        X_cf: np.ndarray,
        y_test: np.ndarray,
        y_target: np.ndarray,
        y_cf_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter only valid counterfactuals."""
        valid_mask = y_cf_pred == y_target
        return X_test[valid_mask], X_cf[valid_mask]


@register_metric("proximity_categorical_hamming")
class CategoricalHammingDistance(BaseDistanceMetric):
    """Categorical distance using Hamming metric."""

    name = "proximity_categorical_hamming"

    def __call__(self, **inputs: Any) -> float:
        """Compute categorical Hamming distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        categorical_features = inputs["categorical_features"]
        dist = cdist(
            X_test_valid[:, categorical_features],
            X_cf_valid[:, categorical_features],
            metric="hamming",
        ).diagonal()
        return np.mean(dist)


@register_metric("proximity_categorical_jaccard")
class CategoricalJaccardDistance(BaseDistanceMetric):
    """Categorical distance using Jaccard metric."""

    name = "proximity_categorical_jaccard"

    def __call__(self, **inputs: Any) -> float:
        """Compute categorical Jaccard distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        categorical_features = inputs["categorical_features"]
        dist = cdist(
            X_test_valid[:, categorical_features],
            X_cf_valid[:, categorical_features],
            metric="jaccard",
        ).diagonal()
        return np.mean(dist)


@register_metric("proximity_continuous_manhattan")
class ContinuousManhattanDistance(BaseDistanceMetric):
    """Continuous distance using Manhattan (cityblock) metric."""

    name = "proximity_continuous_manhattan"

    def __call__(self, **inputs: Any) -> float:
        """Compute continuous Manhattan distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        continuous_features = inputs["continuous_features"]
        dist = cdist(
            X_test_valid[:, continuous_features],
            X_cf_valid[:, continuous_features],
            metric="cityblock",
        ).diagonal()
        return np.mean(dist)


@register_metric("proximity_continuous_euclidean")
class ContinuousEuclideanDistance(BaseDistanceMetric):
    """Continuous distance using Euclidean metric."""

    name = "proximity_continuous_euclidean"

    def __call__(self, **inputs: Any) -> float:
        """Compute continuous Euclidean distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        continuous_features = inputs["continuous_features"]
        dist = cdist(
            X_test_valid[:, continuous_features],
            X_cf_valid[:, continuous_features],
            metric="euclidean",
        ).diagonal()
        return np.mean(dist)


@register_metric("proximity_continuous_mad")
class ContinuousMADDistance(BaseDistanceMetric):
    """Continuous distance using Median Absolute Deviation (MAD) metric."""

    name = "proximity_continuous_mad"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        base_inputs = super().required_inputs()
        base_inputs.add("X_train")
        return base_inputs

    def __call__(self, **inputs: Any) -> float:
        """Compute continuous MAD distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        continuous_features = inputs["continuous_features"]
        X_train = inputs["X_train"]

        # Calculate MAD for normalization
        mad = _median_abs_deviation(X_train[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def mad_cityblock(u, v):
            return _mad_cityblock(u, v, mad)

        dist = cdist(
            X_test_valid[:, continuous_features],
            X_cf_valid[:, continuous_features],
            metric=mad_cityblock,
        ).diagonal()
        return np.mean(dist)


@register_metric("proximity_l2_jaccard")
class CombinedL2JaccardDistance(BaseDistanceMetric):
    """Combined distance using Euclidean (L2) and Jaccard metrics."""

    name = "proximity_l2_jaccard"

    def __call__(self, **inputs: Any) -> float:
        """Compute combined L2 and Jaccard distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        continuous_features = inputs["continuous_features"]
        categorical_features = inputs["categorical_features"]
        ratio_cont = inputs.get("ratio_cont")

        # Calculate continuous distance
        dist_cont = cdist(
            X_test_valid[:, continuous_features],
            X_cf_valid[:, continuous_features],
            metric="euclidean",
        ).diagonal()

        # Calculate categorical distance
        dist_cate = cdist(
            X_test_valid[:, categorical_features],
            X_cf_valid[:, categorical_features],
            metric="jaccard",
        ).diagonal()

        # Calculate ratios
        n_features = X_test_valid.shape[1]
        if ratio_cont is None:
            ratio_continuous = len(continuous_features) / n_features
        else:
            ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_continuous

        # Combined distance
        dist = ratio_continuous * np.mean(dist_cont) + ratio_categorical * np.mean(
            dist_cate
        )
        return dist


@register_metric("proximity_mad_hamming")
class CombinedMADHammingDistance(BaseDistanceMetric):
    """Combined distance using MAD and Hamming metrics."""

    name = "proximity_mad_hamming"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        base_inputs = super().required_inputs()
        base_inputs.add("X_train")
        return base_inputs

    def __call__(self, **inputs: Any) -> float:
        """Compute combined MAD and Hamming distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        continuous_features = inputs["continuous_features"]
        categorical_features = inputs["categorical_features"]
        X_train = inputs["X_train"]
        ratio_cont = inputs.get("ratio_cont")

        # Calculate MAD for normalization
        mad = _median_abs_deviation(X_train[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def mad_cityblock(u, v):
            return _mad_cityblock(u, v, mad)

        # Calculate continuous distance with MAD
        dist_cont = cdist(
            X_test_valid[:, continuous_features],
            X_cf_valid[:, continuous_features],
            metric=mad_cityblock,
        ).diagonal()

        # Calculate categorical distance
        dist_cate = cdist(
            X_test_valid[:, categorical_features],
            X_cf_valid[:, categorical_features],
            metric="hamming",
        ).diagonal()

        # Calculate ratios
        n_features = X_test_valid.shape[1]
        if ratio_cont is None:
            ratio_continuous = len(continuous_features) / n_features
        else:
            ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_continuous

        # Combined distance
        dist = ratio_continuous * np.mean(dist_cont) + ratio_categorical * np.mean(
            dist_cate
        )
        return dist


@register_metric("target_distance")
class TargetDistance(Metric):
    """Distance between predicted and target values (for regression)."""

    name = "target_distance"

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        return {"y_target", "y_cf_pred"}

    def __call__(self, **inputs: Any) -> float:
        """Compute distance between targets."""
        y_target = inputs["y_target"]
        y_cf_pred = inputs["y_cf_pred"]

        # Handle 1D arrays for regression targets
        if y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)
        if y_cf_pred.ndim == 1:
            y_cf_pred = y_cf_pred.reshape(-1, 1)

        dist = cdist(y_target, y_cf_pred, metric="euclidean").diagonal()
        return np.mean(dist)
