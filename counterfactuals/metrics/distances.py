import logging
from typing import Optional

import numpy as np
from scipy.spatial.distance import _validate_vector, cdist

logger = logging.getLogger(__name__)


def _median_abs_deviation(data: np.ndarray, axis=None):
    """
    Calculate the Median Absolute Deviation (MAD) of a dataset along a specified axis.

    Args:
        data (list or numpy array): The input data for which the MAD is to be computed.
        axis (int, optional): The axis along which the median should be computed.
                              The default is None, which computes the MAD of the flattened array.

    Returns:
        numpy array or float: The MAD of the data along the given axis.
    """
    logger.debug("Calculating Median Absolute Deviation (MAD)")
    median = np.median(data, axis=axis)
    if axis is None:
        deviations = np.abs(data - median)
    else:
        deviations = np.abs(data - np.expand_dims(median, axis=axis))
    mad = np.median(deviations, axis=axis)
    return mad


def _mad_cityblock(u, v, mad):
    """
    Calculate the Median Absolute Deviation (MAD) cityblock distance between two vectors.

    Args:
        u (numpy array): The first vector.
        v (numpy array): The second vector.
        mad (numpy array): The Median Absolute Deviation (MAD) values.

    Returns:
        float: The MAD cityblock distance between the two vectors.
    """
    logger.debug("Calculating MAD cityblock distance")
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(
    X_test: np.ndarray,
    X_cf: np.ndarray,
    continuous_features: list[int],
    metric: str = "euclidean",
    X_all: Optional[np.ndarray] = None,
    agg: str = "mean",
) -> float:
    """
    Calculate the continuous distance between two datasets.

    Args:
        X_test (numpy array): The test dataset.
        X_cf (numpy array): The counterfactual dataset.
        continuous_features (list of int): The indices of the continuous features.
        metric (str, optional): The distance metric to be used. Defaults to "euclidean".
        X_all (numpy array, optional): The train dataset. Required if metric is "mad".
        agg (str, optional): The aggregation function to be used. Defaults to "mean".

    Returns:
        float: The continuous distance between the two datasets.
    """
    logger.info("Calculating continuous distance")
    allowed_metrics = ["cityblock", "euclidean", "mad"]
    agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "no": lambda x: x}

    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array"
    assert isinstance(X_cf, np.ndarray), "X_cf should be a numpy array"
    assert X_test.ndim in [1, 2], "X_test should be a 1D or 2D array"
    assert metric in allowed_metrics, f"Metric should be one of: {allowed_metrics}"
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    assert X_test.shape == X_cf.shape, (
        f"Shapes should be the same: {X_test.shape} - {X_cf.shape}"
    )

    # used if distance is calculated for regression target
    if X_cf.ndim == 1:
        logger.warning(
            "X_cf is 1D array, reshaping to 2D and continuous_features would be ignored"
        )
        X_cf = X_cf.reshape(1, -1)
        X_test = X_test.reshape(1, -1)
        continuous_features = [0]

    if metric == "mad":
        assert X_all is not None, "X_all should be provided for MAD distance"
        mad = _median_abs_deviation(X_all[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def mad_cityblock(u, v):
            return _mad_cityblock(u, v, mad)

        metric = mad_cityblock

    dist = cdist(
        X_test[:, continuous_features], X_cf[:, continuous_features], metric=metric
    ).diagonal()
    return agg_funcs[agg](dist)


def categorical_distance(
    X_test: np.ndarray,
    X_cf: np.ndarray,
    categorical_features: list[int],
    metric: str = "jaccard",
    agg: Optional[str] = "mean",
) -> float:
    """
    Calculate the categorical distance between two datasets.

    Args:
        X_test (numpy array): The test dataset.
        X_cf (numpy array): The counterfactual dataset.
        categorical_features (list of int): The indices of the categorical features.
        metric (str, optional): The distance metric to be used. Defaults to "jaccard".
        agg (str, optional): The aggregation function to be used. Defaults to "mean".

    Returns:
        float: The categorical distance between the two datasets.
    """
    logger.info("Calculating categorical distance")
    allowed_metrics = ["hamming", "jaccard"]
    agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "no": lambda x: x}

    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array"
    assert isinstance(X_cf, np.ndarray), "X_cf should be a numpy array"
    assert metric in allowed_metrics, f"Metric should be one of: {allowed_metrics}"
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    assert X_test.shape == X_cf.shape, (
        f"Shapes should be the same: {X_test.shape} - {X_cf.shape}"
    )

    dist = cdist(
        X_test[:, categorical_features], X_cf[:, categorical_features], metric=metric
    ).diagonal()
    return agg_funcs[agg](dist)


def distance_combined(
    X_test: np.ndarray,
    X_cf: np.ndarray,
    continuous_metric: str,
    categorical_metric: str,
    continuous_features: list[int],
    categorical_features: list[int],
    X_all: Optional[np.ndarray] = None,
    ratio_cont: Optional[float] = None,
) -> float:
    """
    Calculate the combined distance between two datasets.

    Args:
        X_test (numpy array): The test dataset.
        X_cf (numpy array): The counterfactual dataset.
        continuous_metric (str): The distance metric to be used for continuous features.
        categorical_metric (str): The distance metric to be used for categorical features.
        continuous_features (list of int): The indices of the continuous features.
        categorical_features (list of int): The indices of the categorical features.
        X_all (numpy array, optional): The combined dataset of X_test and X_cf. Required if continuous_metric is "mad".
        ratio_cont (float, optional): The ratio of continuous features. Defaults to None.

    Returns:
        float: The combined distance between the two datasets.
    """
    logger.info("Calculating combined distance")
    number_features = X_cf.shape[1]
    dist_cont = continuous_distance(
        X_test,
        X_cf,
        continuous_features,
        metric=continuous_metric,
        X_all=X_all,
        agg="mean",
    )
    dist_cate = categorical_distance(
        X_test, X_cf, categorical_features, metric=categorical_metric, agg="mean"
    )
    ratio_continuous = (
        len(continuous_features) / number_features if ratio_cont is None else ratio_cont
    )
    ratio_categorical = 1.0 - ratio_continuous
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_cf = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 9]])
    continuous_features = [0, 1]
    categorical_features = [2]

    logger.info(
        continuous_distance(
            X_test,
            X_cf,
            continuous_features,
            metric="euclidean",
            X_all=None,
            agg="mean",
        )
    )
    logger.info(
        categorical_distance(
            X_test, X_cf, categorical_features, metric="jaccard", agg="mean"
        )
    )
    logger.info(
        distance_combined(
            X_test,
            X_cf,
            "euclidean",
            "jaccard",
            continuous_features,
            categorical_features,
            X_all=None,
            ratio_cont=None,
        )
    )
