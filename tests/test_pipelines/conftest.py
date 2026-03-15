"""Shared fixtures for pipeline integration tests."""

import copy
import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.models.classifier import MLPClassifier
from counterfactuals.models.generative.maf.maf import MaskedAutoregressiveFlow
from counterfactuals.pipelines.base_runner import SearchResult


@pytest.fixture(scope="session")
def synthetic_dataset():
    """Create a tiny synthetic dataset with deterministic seed.

    Shape: 80 train rows, 20 test rows, 6 features (4 continuous + 2 categorical one-hot group).
    Binary classification: y in {0, 1}, balanced.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 80, 20, 6

    X_train = rng.uniform(0, 1, (n_train, n_feat)).astype(np.float32)
    X_test = rng.uniform(0, 1, (n_test, n_feat)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.float32)
    y_test = rng.integers(0, 2, n_test).astype(np.float32)
    # Guarantee at least one sample per class in test set
    y_test[:10] = 0
    y_test[10:] = 1

    class _Dataset:
        def __init__(self):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.features = [f"f{i}" for i in range(n_feat)]
            self.numerical_features_indices = [0, 1, 2, 3]
            self.categorical_features_indices = [4, 5]
            self.categorical_features = ["cat_a"]
            self.categorical_features_lists = [[4, 5]]  # one one-hot group of size 2
            self.numerical_features = [f"f{i}" for i in range(4)]
            self.actionable_features = None

        def train_dataloader(self, batch_size=32, shuffle=False, noise_lvl=0):
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=batch_size,
                shuffle=shuffle,
            )

        def inverse_transform(self, X):
            # Identity transform for tests (data is already unscaled)
            return X

    return _Dataset()


@pytest.fixture(scope="session")
def regression_dataset():
    """Create a tiny synthetic regression dataset with deterministic seed.

    Continuous target values for regression runners like PPCEFR.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 80, 20, 6

    X_train = rng.uniform(0, 1, (n_train, n_feat)).astype(np.float32)
    X_test = rng.uniform(0, 1, (n_test, n_feat)).astype(np.float32)
    y_train = rng.uniform(0, 10, n_train).astype(np.float32)
    y_test = rng.uniform(0, 10, n_test).astype(np.float32)

    class _RegressionDataset:
        def __init__(self):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.features = [f"f{i}" for i in range(n_feat)]
            self.numerical_features_indices = [0, 1, 2, 3]
            self.categorical_features_indices = [4, 5]
            self.categorical_features = ["cat_a"]
            self.categorical_features_lists = [[4, 5]]
            self.numerical_features = [f"f{i}" for i in range(4)]
            self.actionable_features = None

        def train_dataloader(self, batch_size=32, shuffle=False, noise_lvl=0):
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=batch_size,
                shuffle=shuffle,
            )

        def inverse_transform(self, X):
            return X

    return _RegressionDataset()


@pytest.fixture(scope="session")
def tiny_disc_model():
    """MLPClassifier with random weights — fast, no training needed."""
    model = MLPClassifier(num_inputs=6, num_targets=1, hidden_layer_sizes=[8], dropout=0.0)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_gen_model():
    """Small MAF with random weights — provides predict_log_prob interface."""
    model = MaskedAutoregressiveFlow(
        features=6,
        context_features=1,
        hidden_features=4,
        num_blocks_per_layer=1,
        num_layers=2,
    )
    model.eval()
    return model


@pytest.fixture
def base_cfg():
    """Base config with keys every runner accesses."""
    return OmegaConf.create(
        {
            "disc_model": {
                "model": {"_target_": "counterfactuals.models.classifier.MLPClassifier"},
            },
            "counterfactuals_params": {
                "target_class": 1,
                "batch_size": 16,
                "epochs": 1,
                "lr": 0.01,
                "log_prob_quantile": 0.5,
            },
            "experiment": {"relabel_with_disc_model": False},
        }
    )


@pytest.fixture(scope="session")
def globe_ce_dataset(synthetic_dataset):
    """Extended dataset for Globe-CE / AReS runners with preprocessing pipeline."""
    ds = copy.copy(synthetic_dataset)
    scaler = MinMaxScaler()
    scaler.fit(ds.X_train)

    class _MinMaxStep:
        """Minimal interface matching MinMaxScalingStep."""

        def _transform_array(self, X):
            return scaler.transform(X)

        def _inverse_transform_array(self, X):
            return scaler.inverse_transform(X)

    class _PreprocessingPipeline:
        def get_step(self, name):
            return _MinMaxStep() if name == "minmax" else None

    class _FeatureTransformer:
        def transform(self, X):
            return scaler.transform(X)

        def inverse_transform(self, X):
            return scaler.inverse_transform(X)

    ds.preprocessing_pipeline = _PreprocessingPipeline()
    ds.feature_transformer = _FeatureTransformer()
    ds.one_hot_feature_groups = None  # no one-hot groups in synthetic data
    return ds


@pytest.fixture(scope="session")
def test_logger():
    """Test logger instance."""
    _logger = logging.getLogger("test")
    _logger.setLevel(logging.WARNING)  # Suppress most logging in tests
    return _logger


def _assert_valid_result(result: SearchResult, dataset) -> None:
    """Shared assertion function used by all tests.

    Verifies SearchResult structure is valid: shapes, dtypes, timing ≥ 0.
    """
    n = result.X_test.shape[0]
    n_feat = dataset.X_test.shape[1]
    assert isinstance(result, SearchResult)
    assert result.X_cf.shape == (n, n_feat), f"X_cf shape mismatch: {result.X_cf.shape}"
    assert result.X_test.shape == (n, n_feat)

    # Handle both 1D and 2D arrays for y_orig and y_target
    y_orig = result.y_orig.flatten() if result.y_orig.ndim > 1 else result.y_orig
    y_target = result.y_target.flatten() if result.y_target.ndim > 1 else result.y_target
    assert y_orig.shape == (n,), f"y_orig shape mismatch: {y_orig.shape}"
    assert y_target.shape == (n,), f"y_target shape mismatch: {y_target.shape}"

    model_returned = result.model_returned
    assert model_returned.shape == (n,), f"model_returned shape mismatch: {model_returned.shape}"

    # Some runners may return int dtype instead of bool
    assert float(result.cf_search_time) >= 0.0
    assert isinstance(result.extras, dict)


def make_runner(runner_cls, cfg, logger=None):
    """Helper to instantiate a PipelineRunner with minimal dependencies."""
    if logger is None:
        logger = logging.getLogger("test")
    return runner_cls(cfg, logger, preprocessing_pipeline=None)
