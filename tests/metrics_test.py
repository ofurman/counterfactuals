import pytest
import numpy as np
import torch
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.ensemble import IsolationForest

# from counterfactuals.metrics.distances import continuous_distance, categorical_distance, distance_combined

from counterfactuals.metrics import CFMetrics


# Mock models
class MockGenModel:
    def __init__(self):
        pass

    def __call__(self, X, y_target):
        return np.random.rand(X.shape[0])  # Mock log probabilities


class MockDiscModel:
    def __init__(self):
        pass

    def predict(self, X):
        return np.random.randint(0, 2, X.shape[0])  # Mock predictions


# Dummy data for testing
@pytest.fixture
def dummy_data():
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    X_cf = np.random.rand(20, 10)
    y_target = np.random.randint(0, 2, 20)

    gen_model = MockGenModel()
    disc_model = MockDiscModel()

    continuous_features = list(range(5))
    categorical_features = list(range(5, 10))
    prob_plausibility_threshold = 0.2

    return (
        X_cf,
        y_target,
        X_train,
        y_train,
        X_test,
        y_test,
        gen_model,
        disc_model,
        continuous_features,
        categorical_features,
        prob_plausibility_threshold,
    )


@pytest.fixture
def cfmetrics_instance(dummy_data):
    (
        X_cf,
        y_target,
        X_train,
        y_train,
        X_test,
        y_test,
        gen_model,
        disc_model,
        continuous_features,
        categorical_features,
        prob_plausibility_threshold,
    ) = dummy_data
    return CFMetrics(
        X_cf,
        y_target,
        X_train,
        y_train,
        X_test,
        y_test,
        gen_model,
        disc_model,
        continuous_features,
        categorical_features,
        prob_plausibility_threshold=prob_plausibility_threshold,
    )


def test_init(cfmetrics_instance):
    assert cfmetrics_instance.X_cf.shape[1] == cfmetrics_instance.X_train.shape[1]
    assert cfmetrics_instance.X_train.shape[0] == cfmetrics_instance.y_train.shape[0]
    assert cfmetrics_instance.X_test.shape[0] == cfmetrics_instance.y_test.shape[0]
    assert cfmetrics_instance.X_cf.shape[0] == cfmetrics_instance.y_test.shape[0]
    assert (
        len(cfmetrics_instance.continuous_features)
        + len(cfmetrics_instance.categorical_features)
        == cfmetrics_instance.X_cf.shape[1]
    )


def test_convert_to_numpy(cfmetrics_instance):
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(cfmetrics_instance._convert_to_numpy(tensor), array)
    assert np.array_equal(cfmetrics_instance._convert_to_numpy(array), array)
    with pytest.raises(ValueError):
        cfmetrics_instance._convert_to_numpy("invalid input")


def test_coverage(cfmetrics_instance):
    assert 0 <= cfmetrics_instance.coverage() <= 1


def test_validity(cfmetrics_instance):
    assert 0 <= cfmetrics_instance.validity() <= 1


def test_actionability(cfmetrics_instance):
    assert 0 <= cfmetrics_instance.actionability() <= 1


def test_sparsity(cfmetrics_instance):
    assert 0 <= cfmetrics_instance.sparsity() <= 1


def test_prob_plausibility(cfmetrics_instance):
    assert 0 <= cfmetrics_instance.prob_plausibility() <= 1


def test_log_density(cfmetrics_instance):
    assert isinstance(cfmetrics_instance.log_density(), float)


def test_lof_scores(cfmetrics_instance):
    assert isinstance(cfmetrics_instance.lof_scores(), float)


def test_isolation_forest_scores(cfmetrics_instance):
    assert isinstance(cfmetrics_instance.isolation_forest_scores(), float)


def test_feature_distance(cfmetrics_instance):
    assert isinstance(cfmetrics_instance.feature_distance(), float)


def test_target_distance(cfmetrics_instance):
    assert isinstance(cfmetrics_instance.target_distance(), float)


def test_calc_all_metrics(cfmetrics_instance):
    metrics = cfmetrics_instance.calc_all_metrics()
    assert isinstance(metrics, dict)
    assert "coverage" in metrics
    assert "validity" in metrics
    assert "actionability" in metrics
    assert "sparsity" in metrics
    assert "prob_plausibility" in metrics
    assert "log_density_cf" in metrics
    assert "lof_scores_cf" in metrics
    assert "isolation_forest_scores_cf" in metrics
