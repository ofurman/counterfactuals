import pytest
import numpy as np
from counterfactuals.metrics.distances import (
    _median_abs_deviation,
    _mad_cityblock,
    continuous_distance,
    categorical_distance,
    distance_combined,
)


# Test cases for _median_abs_deviation
def test_median_abs_deviation_typical():
    data = np.array([1, 2, 3, 4, 5])
    expected = 1.0
    assert _median_abs_deviation(data) == expected


def test_median_abs_deviation_multidimensional():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    expected = np.array([2, 2])
    np.testing.assert_array_equal(_median_abs_deviation(data, axis=0), expected)


def test_median_abs_deviation_empty():
    assert np.isnan(_median_abs_deviation(np.array([])))


# Test cases for _mad_cityblock
def test_mad_cityblock_typical():
    u = np.array([1, 2])
    v = np.array([2, 4])
    mad = np.array([1, 2])
    expected = 2.0
    assert _mad_cityblock(u, v, mad) == expected


def test_mad_cityblock_zero_mad():
    u = np.array([1, 2])
    v = np.array([1, 2])
    mad = np.array([0, 0])  # Should not use zero MAD without adjustment
    assert np.isnan(_mad_cityblock(u, v, mad))


def test_mad_cityblock_large_values():
    u = np.array([1000, 2000])
    v = np.array([1000, 2001])
    mad = np.array([1, 1])
    expected = 1.0
    assert _mad_cityblock(u, v, mad) == expected


# Test cases for continuous_distance
def test_continuous_distance_valid_input():
    X_test = np.array([[1, 2], [3, 4]])
    X_cf = np.array([[1, 3], [3, 5]])
    features = [0, 1]
    X_all = np.array([[1, 2], [2, 3], [3, 4]])
    assert continuous_distance(X_test, X_cf, features, "euclidean", X_all) >= 0


def test_continuous_distance_invalid_metric():
    X_test = np.array([[1, 2], [3, 4]])
    X_cf = np.array([[1, 3], [3, 5]])
    features = [0, 1]
    with pytest.raises(AssertionError):
        continuous_distance(X_test, X_cf, features, "invalid_metric")


def test_continuous_distance_shape_mismatch():
    X_test = np.array([[1, 2]])
    X_cf = np.array([[1, 2], [3, 4]])
    features = [0, 1]
    with pytest.raises(AssertionError):
        continuous_distance(X_test, X_cf, features, "euclidean")


# Test cases for categorical_distance
def test_categorical_distance_valid():
    X_test = np.array([[0, 1], [1, 0]])
    X_cf = np.array([[1, 1], [0, 0]])
    features = [1]
    result = categorical_distance(X_test, X_cf, features)
    assert isinstance(result, float)


def test_categorical_distance_invalid_metric():
    X_test = np.array([[0, 1], [1, 0]])
    X_cf = np.array([[1, 1], [0, 0]])
    features = [1]
    with pytest.raises(AssertionError):
        categorical_distance(X_test, X_cf, features, metric="invalid")


def test_categorical_distance_shape_mismatch():
    X_test = np.array([[0, 1]])
    X_cf = np.array([[0, 1], [1, 0]])
    features = [0, 1]
    with pytest.raises(AssertionError):
        categorical_distance(X_test, X_cf, features)


# Test cases for distance_combined
def test_distance_combined_valid_input():
    X_test = np.array([[0, 1, 1], [1, 0, 0]])
    X_cf = np.array([[1, 1, 1], [0, 0, 0]])
    continuous_features = [0]
    categorical_features = [1, 2]
    result = distance_combined(
        X_test, X_cf, "euclidean", "jaccard", continuous_features, categorical_features
    )
    assert isinstance(result, float)


def test_distance_combined_invalid_continuous_features():
    X_test = np.array([[0, 1, 1], [1, 0, 0]])
    X_cf = np.array([[1, 1, 1], [0, 0, 0]])
    with pytest.raises(AssertionError):
        distance_combined(X_test, X_cf, "invalid", "jaccard", [], [1, 2])


def test_distance_combined_invalid_categorical_features():
    X_test = np.array([[0, 1, 1], [1, 0, 0]])
    X_cf = np.array([[1, 1, 1], [0, 0, 0]])
    with pytest.raises(AssertionError):
        distance_combined(X_test, X_cf, "euclidean", "invalid", [0], [])
