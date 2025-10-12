"""Distance metrics for counterfactual evaluation."""

import logging
from typing import Any, Callable, Union

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


class CombinedDistanceMetric(BaseDistanceMetric):
    """
    Base class for combined distance metrics.

    Computes a weighted combination of continuous and categorical distances.
    Subclasses should specify the continuous and categorical metrics to use.
    """

    continuous_metric: Union[str, Callable]
    categorical_metric: str
    requires_X_train: bool = False

    def required_inputs(self) -> set[str]:
        """Return the set of required input keys."""
        base_inputs = super().required_inputs()
        if self.requires_X_train:
            base_inputs.add("X_train")
        return base_inputs

    def _compute_combined_distance(
        self,
        X_test_valid: np.ndarray,
        X_cf_valid: np.ndarray,
        continuous_features: list[int],
        categorical_features: list[int],
        ratio_cont: float | None,
        continuous_metric: Union[str, Callable],
        categorical_metric: str,
    ) -> float:
        """
        Compute combined distance as weighted average of continuous and categorical.

        Args:
            X_test_valid: Valid test instances.
            X_cf_valid: Valid counterfactual instances.
            continuous_features: Indices of continuous features.
            categorical_features: Indices of categorical features.
            ratio_cont: Ratio for continuous features (if None, computed from feature counts).
            continuous_metric: Metric for continuous features.
            categorical_metric: Metric for categorical features.

        Returns:
            Combined distance value.
        """
        # Calculate continuous distance
        dist_cont = cdist(
            X_test_valid[:, continuous_features],
            X_cf_valid[:, continuous_features],
            metric=continuous_metric,
        ).diagonal()

        # Calculate categorical distance
        dist_cate = cdist(
            X_test_valid[:, categorical_features],
            X_cf_valid[:, categorical_features],
            metric=categorical_metric,
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


@register_metric("proximity_categorical_hamming")
class EuclideanHammingCombinedDistance(CombinedDistanceMetric):
    """
    Combined distance using Euclidean (continuous) and Hamming (categorical) metrics.

    Registered as: proximity_categorical_hamming (legacy compatibility name)

    Note: This replicates the buggy behavior from CFMetrics.feature_distance()
    where calling with categorical_metric="hamming" still had continuous_metric="euclidean"
    as default, resulting in a combined distance calculation.
    """

    name = "proximity_categorical_hamming"
    continuous_metric = "euclidean"
    categorical_metric = "hamming"

    def __call__(self, **inputs: Any) -> float:
        """Compute combined Euclidean and Hamming distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=self.continuous_metric,
            categorical_metric=self.categorical_metric,
        )


@register_metric("proximity_categorical_jaccard")
class EuclideanJaccardCombinedDistance(CombinedDistanceMetric):
    """
    Combined distance using Euclidean (continuous) and Jaccard (categorical) metrics.

    Registered as: proximity_categorical_jaccard (legacy compatibility name)

    Note: This replicates the buggy behavior from CFMetrics.feature_distance()
    where calling with categorical_metric="jaccard" still had continuous_metric="euclidean"
    as default, resulting in a combined distance calculation.
    """

    name = "proximity_categorical_jaccard"
    continuous_metric = "euclidean"
    categorical_metric = "jaccard"

    def __call__(self, **inputs: Any) -> float:
        """Compute combined Euclidean and Jaccard distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=self.continuous_metric,
            categorical_metric=self.categorical_metric,
        )


@register_metric("proximity_continuous_manhattan")
class ManhattanJaccardCombinedDistance(CombinedDistanceMetric):
    """
    Combined distance using Manhattan/Cityblock (continuous) and Jaccard (categorical) metrics.

    Registered as: proximity_continuous_manhattan (legacy compatibility name)

    Note: This replicates the buggy behavior from CFMetrics.feature_distance()
    where calling with continuous_metric="cityblock" still had categorical_metric="jaccard"
    as default, resulting in a combined distance calculation.
    """

    name = "proximity_continuous_manhattan"
    continuous_metric = "cityblock"
    categorical_metric = "jaccard"

    def __call__(self, **inputs: Any) -> float:
        """Compute combined Manhattan and Jaccard distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=self.continuous_metric,
            categorical_metric=self.categorical_metric,
        )


@register_metric("proximity_continuous_euclidean")
class EuclideanJaccardCombinedDistanceAlt(CombinedDistanceMetric):
    """
    Combined distance using Euclidean (continuous) and Jaccard (categorical) metrics.

    Registered as: proximity_continuous_euclidean (legacy compatibility name)

    Note: This replicates the buggy behavior from CFMetrics.feature_distance()
    where calling with continuous_metric="euclidean" still had categorical_metric="jaccard"
    as default, resulting in a combined distance calculation.
    This is actually the same as proximity_l2_jaccard but registered under a different name.
    """

    name = "proximity_continuous_euclidean"
    continuous_metric = "euclidean"
    categorical_metric = "jaccard"

    def __call__(self, **inputs: Any) -> float:
        """Compute combined Euclidean and Jaccard distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=self.continuous_metric,
            categorical_metric=self.categorical_metric,
        )


@register_metric("proximity_continuous_mad")
class MADJaccardCombinedDistance(CombinedDistanceMetric):
    """
    Combined distance using MAD (continuous) and Jaccard (categorical) metrics.

    Registered as: proximity_continuous_mad (legacy compatibility name)

    Note: This replicates the buggy behavior from CFMetrics.feature_distance()
    where calling with continuous_metric="mad" still had categorical_metric="jaccard"
    as default, resulting in a combined distance calculation.
    """

    name = "proximity_continuous_mad"
    categorical_metric = "jaccard"
    requires_X_train = True

    def __call__(self, **inputs: Any) -> float:
        """Compute combined MAD and Jaccard distance."""
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

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=mad_cityblock,
            categorical_metric=self.categorical_metric,
        )


@register_metric("proximity_l2_jaccard")
class L2JaccardCombinedDistance(CombinedDistanceMetric):
    """
    Combined distance using Euclidean/L2 (continuous) and Jaccard (categorical) metrics.

    Registered as: proximity_l2_jaccard

    Note: This is an intentionally combined metric (not a legacy bug).
    """

    name = "proximity_l2_jaccard"
    continuous_metric = "euclidean"
    categorical_metric = "jaccard"

    def __call__(self, **inputs: Any) -> float:
        """Compute combined L2 and Jaccard distance."""
        X_test_valid, X_cf_valid = self._filter_valid(
            inputs["X_test"],
            inputs["X_cf"],
            inputs["y_test"],
            inputs["y_target"],
            inputs["y_cf_pred"],
        )

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=self.continuous_metric,
            categorical_metric=self.categorical_metric,
        )


@register_metric("proximity_mad_hamming")
class MADHammingCombinedDistance(CombinedDistanceMetric):
    """
    Combined distance using MAD (continuous) and Hamming (categorical) metrics.

    Registered as: proximity_mad_hamming

    Note: This is an intentionally combined metric (not a legacy bug).
    """

    name = "proximity_mad_hamming"
    categorical_metric = "hamming"
    requires_X_train = True

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
        X_train = inputs["X_train"]

        # Calculate MAD for normalization
        mad = _median_abs_deviation(X_train[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def mad_cityblock(u, v):
            return _mad_cityblock(u, v, mad)

        return self._compute_combined_distance(
            X_test_valid=X_test_valid,
            X_cf_valid=X_cf_valid,
            continuous_features=inputs["continuous_features"],
            categorical_features=inputs["categorical_features"],
            ratio_cont=inputs.get("ratio_cont"),
            continuous_metric=mad_cityblock,
            categorical_metric=self.categorical_metric,
        )


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
